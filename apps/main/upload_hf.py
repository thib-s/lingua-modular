from dataclasses import dataclass
import logging
from pathlib import Path
from huggingface_hub import PyTorchModelHubMixin
from typing import Optional
from omegaconf import OmegaConf
import torch
from apps.main.generate import load_consolidated_model_and_tokenizer
from apps.main.transformer import LMTransformer, LMTransformerArgs
from lingua.args import dump_config
from lingua.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints
# from lingua.distributed import (
#     DistributedArgs,
#     get_global_rank,
#     setup_torch_distributed,
# )

EVAL_FOLDER_NAME = "{:010d}"

logger = logging.getLogger()


@dataclass
class UploadArgs:
    username: str = ""
    repo_name: str = ""

    name: str = "upload"
    dump_dir: Optional[str] = None
    ckpt_dir: str = ""


class LinguaModelHub(torch.nn.Module,
                     PyTorchModelHubMixin):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def upload(cfg: UploadArgs):
    # if not torch.distributed.is_initialized():
    #     setup_torch_distributed(DistributedArgs())
    # if (
    #     Path(cfg.ckpt_dir).exists()
    #     and (Path(cfg.ckpt_dir) / "params.json").exists()
    #     and next(Path(cfg.ckpt_dir).glob("*.pth"), None) is not None
    # ):
    #     consolidate_path = Path(cfg.ckpt_dir)
    # else:
    #     consolidate_path = Path(cfg.ckpt_dir) / CONSOLIDATE_FOLDER
    #     if not consolidate_path.exists() and get_global_rank() == 0:
    #         consolidate_path = consolidate_checkpoints(cfg.ckpt_dir)
    consolidate_path = Path(cfg.ckpt_dir) / CONSOLIDATE_FOLDER

    Path(cfg.dump_dir).mkdir(parents=True, exist_ok=True)
    dump_config(cfg, Path(cfg.dump_dir) / "config.yaml", log_config=False)

    consolidate_path = str(consolidate_path)
    # torch.distributed.barrier()
    logger.info("Loading model")
    model, tokenizer, train_cfg = load_consolidated_model_and_tokenizer(
        consolidate_path,
        model_cls=LMTransformer,
        model_args_cls=LMTransformerArgs,
    )
    logger.info("Model loaded")

    lingua_model = LinguaModelHub(model)
    lingua_model.push_to_hub(f"{cfg.username}/{cfg.repo_name}",
                             private=True)


def main():
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    So if the dataclass looks like
    @dataclass
    class DummyArgs:
        name: str
        model: LMTransformerArgsgs
    @dataclass
    class LMTransformerArgsgs:
        dim: int
    Then you can pass model.dim=32 to change values in LMTransformerArgsgs
    or just name=tictac for top level attributes.
    The behavior here is as follows:
    1. We instantiate UploadArgs with its default values
    2. We override those default values with the ones in the provided config file
    3. We override the result with the additional arguments provided through command line
    For example, if the config is the following
    model:
        dim: 128
        n_layers: 4
    and you call upload.py with upload.py model.dim=64
    Then the final UploadArgs will have
    model:
        dim: 64
        n_layers: 4
    Plus all the default values in UploadArgs dataclass.
    """
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(UploadArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    upload(cfg)


if __name__ == "__main__":
    main()
