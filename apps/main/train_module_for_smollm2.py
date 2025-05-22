#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train SmolLM-2 with Meta-Lingua-style “modules”.

Save as:  train_smollm_modules.py
Run   as: python -m torch.distributed.run --nproc_per_node <GPUS> train_smollm_modules.py \
             config=<path-to-yaml> model.hf_checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct"
"""

# --------------------------------------------------------------------------- #
#                              Standard imports                               #
# --------------------------------------------------------------------------- #
import gc
import logging
import os
import sys
from copy import deepcopy
from contextlib import ExitStack
from dataclasses import dataclass, field, asdict
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
import xformers.profiler
from torch.optim import lr_scheduler
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed._tensor import DTensor

# --------------------------------------------------------------------------- #
#                  Meta-Lingua / project-specific  imports                    #
# --------------------------------------------------------------------------- #
from lingua.args import dataclass_from_dict, dump_config, flatten_dict
from lingua.checkpoint import CheckpointArgs, CheckpointManager, load_from_checkpoint
from lingua.data import (
    DataArgs,
    PackTokensState,
    build_dataloader_from_args,
    init_dataloader_state_from_args,
)
from lingua.distributed import (
    DistributedArgs,
    EnvironmentArgs,
    init_signal_handler,
    dist_mean_dict,
    get_device_mesh,
    get_is_master,
    get_world_size,
    parallelize_model,
    setup_env,
    setup_torch_distributed,
    clean_env,
    requeue_slurm_job,
    check_model_value_range,
)
from lingua.logger import init_logger
from lingua.metrics import GPUMemoryMonitor, LoggingArgs, MetricLogger, get_num_params
from lingua.optim import OptimArgs, build_optimizer
from lingua.profiling import ProfilerArgs, maybe_run_profiler
from lingua.tokenizer import build_tokenizer
from lingua.probe import AutoProbeD
from lingua.stool import StoolArgs, launch_job

# Meta-Lingua transformer utilities
from lingua.transformer import (
    BaseTransformer,
    BaseTransformerArgs,
    RMSNorm,
    TiedLinear,
    cross_entropy,
)

# Keep the helper policies from the original training script
from apps.main.transformer import (
    get_no_recompute_ops,
    build_fsdp_grouping_plan,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#                     SmolLM-2  →  Meta-Lingua   wrapper                      #
# --------------------------------------------------------------------------- #
from transformers import AutoConfig, AutoModelForCausalLM
from torch.nn.attention.flex_attention import create_block_mask, BlockMask
from xformers.ops import fmha, AttentionBias

# --------- helpers reused from the original code --------------------------- #
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def create_causal_mask(seqlen, attn_impl="sdpa", sliding_window=None):
    if sliding_window is not None and attn_impl == "xformers":
        return fmha.attn_bias.LocalAttentionFromBottomRightMask(
            window_left=sliding_window - 1, window_right=0
        )
    elif attn_impl == "xformers":
        return fmha.attn_bias.LowerTriangularMask()
    elif attn_impl == "sdpa":
        return "causal"
    elif attn_impl == "flex_attention":
        from torch.nn.attention.flex_attention import create_block_mask  # local import
        return create_block_mask(causal_mask, None, None, seqlen, seqlen)
    else:
        raise NotImplementedError(
            f"Attention {attn_impl} with {sliding_window} sliding window not implemented"
        )


@dataclass
class SmolLM2Args(BaseTransformerArgs):
    """
    Extend BaseTransformerArgs with SmolLM-2–specific options.
    All fields referenced elsewhere in the script *must* live here.
    """
    # ---- checkpoint -------------------------------------------------------
    hf_checkpoint: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

    # ---- tokenizer / vocab -----------------------------------------------
    # gets filled once we know the tokenizer size (see validate_train_args)
    vocab_size: int = -1


# -------------------------  WRAPPER  -------------------------------------- #
class SmolLM2WithModules(BaseTransformer):
    """
    Thin adapter that loads a HuggingFace SmolLM-2 checkpoint into the
    Meta-Lingua `BaseTransformer`, then lets you plug “modules” on every layer.
    """

    def __init__(
        self,
        checkpoint: str,
        module_seq_len: int,
        module_agg: str = "sum",
        weight_tying: bool = True,
    ):
        # 1) Mirror the HF model config into BaseTransformerArgs ---------------
        hf_cfg = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)  # checkpoint is hub or local path
        args = SmolLM2Args(
            hf_checkpoint=checkpoint,
            dim=hf_cfg.hidden_size,
            n_layers=hf_cfg.num_hidden_layers,
            head_dim=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
            n_heads=hf_cfg.num_attention_heads,
            n_kv_heads=hf_cfg.num_attention_heads,
            multiple_of=256,
            norm_eps=getattr(hf_cfg, "layer_norm_epsilon", 1e-5),
            rope_theta=getattr(hf_cfg, "rope_theta", 10000.0),
            max_seqlen=hf_cfg.max_position_embeddings,
            vocab_size=hf_cfg.vocab_size,
            module_seq_len=module_seq_len,
            module_agg=module_agg,
        )
        super().__init__(args)

        self.weight_tying = weight_tying

        # 2) Load the HF model and freeze its parameters -----------------------
        hf = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16,  # change if you prefer bf16/fp32
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        for p in hf.parameters():
            p.requires_grad = False
        self._hf = hf  # keep for weight copying

        # 3) Tie/clone embeddings + lm_head -----------------------------------
        self.tok_embeddings = hf.model.embed_tokens
        if weight_tying:
            self.output = TiedLinear(self.tok_embeddings)
        else:
            self.output = nn.Linear(
                args.dim, hf_cfg.vocab_size, bias=False, dtype=self.tok_embeddings.weight.dtype
            )
            self.output.weight.data.copy_(hf.lm_head.weight)

        # Final RMSNorm (hf.model.norm or ln_f depending on architecture)
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        hf_final_norm = getattr(hf.model, "norm", None) or getattr(hf.model, "ln_f", None)
        if hf_final_norm is not None:
            self.norm.weight.data.copy_(hf_final_norm.weight)

        # 4) Copy transformer block weights layer-by-layer ---------------------
        self._load_weights_from_hf()

        # 5) NB: new “modules” start as zeros; call add_module / enable_module
        # after constructing the wrapper.

    # --------------------------------------------------------------------- #
    #                            weight loading                             #
    # --------------------------------------------------------------------- #
    def _load_weights_from_hf(self):
        """Copy every transformer block’s W_q, W_k, W_v, W_o, FFN, norms."""
        hf_blocks = self._hf.model.layers if hasattr(self._hf.model, "layers") else self._hf.model.transformer.h
        assert len(hf_blocks) == len(self.layers), "HF <> Meta-Lingua depth mismatch"

        for i, (our_blk, hf_blk) in enumerate(zip(self.layers, hf_blocks)):
            # ---- RMSNorms ----
            our_blk.attention_norm.weight.data.copy_(hf_blk.input_layernorm.weight)
            our_blk.ffn_norm.weight.data.copy_(hf_blk.post_attention_layernorm.weight)

            # ---- Attention ----
            our_blk.attention.wq.weight.data.copy_(hf_blk.self_attn.q_proj.weight)
            our_blk.attention.wk.weight.data.copy_(hf_blk.self_attn.k_proj.weight)
            our_blk.attention.wv.weight.data.copy_(hf_blk.self_attn.v_proj.weight)
            our_blk.attention.wo.weight.data.copy_(hf_blk.self_attn.o_proj.weight)

            # ---- MLP ----
            # SmolLM-2 uses (gate + up) → silu → down  (same as Llama)
            our_blk.feed_forward.w1.weight.data.copy_(hf_blk.mlp.gate_proj.weight)
            our_blk.feed_forward.w3.weight.data.copy_(hf_blk.mlp.up_proj.weight)
            our_blk.feed_forward.w2.weight.data.copy_(hf_blk.mlp.down_proj.weight)

    # --------------------------------------------------------------------- #
    #                              forward                                  #
    # --------------------------------------------------------------------- #
    def forward(
        self,
        input_ids: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        attn_impl: str = "sdpa",
    ):
        """
        Exactly the same API as Meta-Lingua `LMTransformer`: if `target`
        is given returns cross-entropy loss, else returns logits.
        """
        bsz, seqlen = input_ids.shape
        hidden = self.tok_embeddings(input_ids)  # (B,S,D)

        if mask is None:
            mask = create_causal_mask(seqlen, attn_impl)

        hidden = super().forward(hidden, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
        logits = self.output(self.norm(hidden))  # (B,S,V)

        if target is not None:
            return cross_entropy(logits, target)
        return logits


# --------------------------------------------------------------------------- #
#                          Training dataclass bundle                          #
# --------------------------------------------------------------------------- #
@dataclass
class SmolLM2Args(BaseTransformerArgs):
    """
    Extend LMTransformerArgs with the HF checkpoint path.
    (We still inherit all the fields used by the distributed helpers)
    """
    hf_checkpoint: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"


@dataclass
class TrainArgs:
    # ---- bookkeeping ------------------------------------------------------
    name: str = "smollm2-modules"
    dump_dir: str = ""
    seed: int = 42
    steps: int = 1000
    grad_acc_steps: int = 1

    # ---- sub-configs ------------------------------------------------------
    data: DataArgs = field(default_factory=DataArgs)
    optim: OptimArgs = field(default_factory=OptimArgs)
    model: SmolLM2Args = field(default_factory=SmolLM2Args)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)

    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    profiling: ProfilerArgs = field(default_factory=ProfilerArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)

    # optional
    gc_collect_freq: int = 1000
    probe_freq: Optional[int] = None
    async_eval_gpus: Optional[int] = None
    eval: Optional[Any] = None


# --------------------------------------------------------------------------- #
#                        Train-state container (unchanged)                    #
# --------------------------------------------------------------------------- #
@dataclass
class TrainState(Stateful):
    step: int
    acc_step: int
    scheduler: lr_scheduler.LambdaLR
    data_loader_state: PackTokensState

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "acc_step": self.acc_step,
            "data_loader_state": self.data_loader_state,
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.acc_step = state_dict["acc_step"]
        self.data_loader_state = PackTokensState(**state_dict["data_loader_state"])
        self.scheduler.load_state_dict(state_dict["scheduler"])


# --------------------------------------------------------------------------- #
#                           Argument validation helper                        #
# --------------------------------------------------------------------------- #
def validate_train_args(args: TrainArgs, output_size: int):
    # Ensure the field exists and is initialised
    if getattr(args.model, "vocab_size", -1) < 0:
        logger.info(f"Setting model vocab_size to {output_size}")
        args.model.vocab_size = output_size
    assert args.model.vocab_size == output_size, (
        f"vocab_size mismatch: tokenizer has {output_size}, "
        f"but model expects {args.model.vocab_size}"
    )

    assert args.dump_dir, "dump_dir must be set"
    if args.checkpoint.path is None:
        args.checkpoint.path = str(Path(args.dump_dir) / "checkpoints")

    for src in args.data.sources:
        path = os.path.join(args.data.root_dir, src)
        assert os.path.exists(path), f"{path} does not exist"

    args.model.max_seqlen = args.data.seq_len


# --------------------------------------------------------------------------- #
#                                   Training                                  #
# --------------------------------------------------------------------------- #
preemption_flag = dict(flag=False)


def set_preemption_flag(signum, frame):
    logger.warning(f"Received signal {signum}; pre-empting.")
    preemption_flag["flag"] = True


def every_n_steps(state, freq, acc_step=None, acc_freq=None):
    ok = state.step % freq == 0
    if acc_step is not None:
        ok = ok and state.acc_step == acc_step
    elif acc_freq is not None:
        ok = ok and state.acc_step % acc_freq == 0
    return ok


def train(args: TrainArgs):
    with ExitStack() as stack:
        # ---------- tokenizer / validate cfg --------------------------------
        tok = build_tokenizer(args.data.tokenizer.name, args.data.tokenizer.path)
        validate_train_args(args, tok.n_words)

        if get_is_master():
            os.makedirs(args.dump_dir, exist_ok=True)
            dump_config(args, Path(args.dump_dir) / "config.yaml")
        init_logger(Path(args.dump_dir) / "train.log")

        # ---------- env / distributed --------------------------------------
        init_signal_handler(set_preemption_flag)
        setup_env(args.env)
        setup_torch_distributed(args.distributed)
        mesh = get_device_mesh(args.distributed)

        torch.manual_seed(args.seed)
        logger.info("Building SmolLM-2 model with modules")

        # ---------- build model on meta ------------------------------------
        with torch.device("meta"):
            model = SmolLM2WithModules(
                checkpoint=args.model.hf_checkpoint,
                module_seq_len=args.model.module_seq_len,
                module_agg=args.model.module_agg,
                weight_tying=True,
            )

            # Optionally add a fresh module to train
            if args.model.create_module_name != "None":
                for p in model.parameters():
                    p.requires_grad = False
                model.add_module(args.model.create_module_name, args.model.module_seq_len)
                model.enable_module(args.model.create_module_name)

        params_total = get_num_params(model)
        logger.info(f"Model has {params_total:,} parameters")

        # ---------- parallelise (FSDP / TP) --------------------------------
        model = parallelize_model(
            model,
            mesh,
            args.model,
            args.distributed,
            fsdp_grouping_plan=build_fsdp_grouping_plan(args.model),
            tp_parallelize=None,
            no_recompute_ops=get_no_recompute_ops(),
        ).to_empty(device="cuda")

        # ---------- load / init weights ------------------------------------
        if args.checkpoint.init_ckpt_path:
            logger.info(f"Loading init checkpoint from {args.checkpoint.init_ckpt_path}")
            load_from_checkpoint(args.checkpoint.init_ckpt_path, model, model_key="model")
            model.rope_embeddings.reset_parameters()
        else:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(args.model.seed)
                model.init_weights()
        check_model_value_range(model, range=10.0, std=1.0)

        # ---------- add module after init (if needed) ----------------------
        if args.model.create_module_name != "None":
            model.add_module(args.model.create_module_name, args.model.module_seq_len)
            model.enable_module(args.model.create_module_name)

        # ---------- optimise / data-loader ---------------------------------
        optimizer, scheduler = build_optimizer(model, args.optim, args.steps)
        dp_mesh = mesh["dp_replicate"]
        dp_rank = dp_mesh.get_local_rank()
        dp_degree = dp_mesh.size()
        loader_state = init_dataloader_state_from_args(args.data, dp_rank, dp_degree)

        train_state = TrainState(0, 0, scheduler, loader_state)
        ckpt_mgr = CheckpointManager.instantiate_and_make_dir(args.checkpoint)
        ckpt_mgr.load(model, optimizer, train_state, mesh)

        model.train()
        metric_logger = stack.enter_context(MetricLogger(Path(args.dump_dir) / "metrics.jsonl", args))
        data_loader = stack.enter_context(build_dataloader_from_args(args.data, state=train_state.data_loader_state))
        profiler = stack.enter_context(maybe_run_profiler(args.dump_dir, model, args.profiling))

        gpu_mem = GPUMemoryMonitor("cuda")
        logger.info(str(gpu_mem))

        nwords_since_log = 0
        t_last_log = timer()

        while train_state.step < args.steps:
            # ------------------------------------------------------------- #
            train_state.acc_step = (train_state.acc_step + 1) % args.grad_acc_steps

            # fetch batch
            batch, train_state.data_loader_state = next(data_loader)
            batch = torch.tensor(batch, dtype=torch.long)
            input_ids = batch[:, :, 0].cuda()
            labels = batch[:, :, 1].cuda()
            nwords_since_log += input_ids.numel()

            if every_n_steps(train_state, args.gc_collect_freq, acc_step=0):
                gc.collect()

            # forward / backward
            loss = model(input_ids, labels)
            loss_scaled = loss / args.grad_acc_steps
            loss_scaled.backward()

            if train_state.acc_step == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.optim.clip, foreach=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_state.step += 1
            else:
                grad_norm = -1.0  # not updated this sub-step

            # --------------------- logging & checkpoint -------------------
            if every_n_steps(train_state, args.logging.freq, acc_step=0):
                dt = timer() - t_last_log
                wps = nwords_since_log / dt
                gpu_stats = gpu_mem.get_peak_stats()

                metrics = {
                    "global_step": train_state.step,
                    "loss": loss.item(),
                    "grad_norm": float(grad_norm),
                    "wps": wps,
                    "gpu/max_active_pct": gpu_stats.max_active_pct,
                }
                metrics = flatten_dict(metrics, sep="/")
                metrics.update(dist_mean_dict({"loss/out": loss.item()}))
                if get_is_master():
                    metric_logger.log(metrics)
                nwords_since_log = 0
                t_last_log = timer()
                gpu_mem.reset_peak_stats()
                logger.info(f"step {train_state.step}  loss {loss.item():.4f}  wps {wps:.1f}")

            # checkpoint
            if every_n_steps(train_state, args.checkpoint.dump.every, acc_step=0):
                ckpt_mgr.save(model, optimizer, train_state, args, device_mesh=mesh)

            if preemption_flag["flag"]:
                ckpt_mgr.save(model, optimizer, train_state, args, device_mesh=mesh)
                requeue_slurm_job()
                sys.exit(0)

        # final save
        ckpt_mgr.save(model, optimizer, train_state, args, device_mesh=mesh)


# --------------------------------------------------------------------------- #
#                                 CLI entry-point                             #
# --------------------------------------------------------------------------- #
def main():
    """
    Launch with:

        python -m torch.distributed.run --nproc_per_node <N_GPU> train_smollm_modules.py \
            config=<your-yaml> \
            model.hf_checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct" \
            model.create_module_name=my_adapter
    """
    cli = OmegaConf.from_cli()
    cfg_file = OmegaConf.load(cli.config)
    del cli.config

    cfg = OmegaConf.merge(OmegaConf.structured(TrainArgs()), cfg_file, cli)
    cfg = OmegaConf.to_object(cfg)  # convert to dataclasses

    train(cfg)


if __name__ == "__main__":
    main()
