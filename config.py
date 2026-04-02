# new data class for arguments
from dataclasses import dataclass
import argparse
import yaml
import os

@dataclass
class Config:
    input: str = None
    result: str = None
    config: str = "../configs/config.yaml"
    gpu_ids: str = None
    batch: int = 50
    epochs: int = 25
    lr: float = 1e-4
    wd: float = 0.0
    val_ratio: float = 0.1
    apc: int = None
    cos_anneal: str = None
    cos_lr: str = None
    exp_lr: str = None
    plateau_lr: str = None
    dry_run: bool = False
    reset_early_stopping: bool = False
    enable_shuffle: bool = True
    mini_epoch: int = None
    early_stopping: int = 1
    checkpoint_save: int = 10
    subsetpdbs: str = "ok_list.txt"
    energy_matching: bool = False
    energy_weight: float = 0.0
    force_weight: float = 1.0
    term_def: str = None
    embedding: str = None
    chunk_dataset: int = None
    npfile: bool = False
    yaml: str = None

def parse_config(args: argparse.ArgumentParser) -> Config:
    config = Config()
    if args.yaml is not None:
        if args.yaml:
            with open(args.yaml, "r", encoding="utf-8") as f:
                yml = yaml.safe_load(f)
                config.input = yml["input"]
                config.result = yml["result"]
                config.config = yml["config"]
                config.wd = yml["wd"]
                config.lr = yml["lr"]
                gpus = yml["gpus"]
                config.batch = yml["batch"]
                config.epochs = yml["epochs"]
                config.val_ratio = yml["val_ratio"]
                config.apc = yml["apc"]
                config.cos_anneal = yml["cos_anneal"]
                config.cos_lr = yml["cos_lr"]
                config.exp_lr = yml["exp_lr"]
                config.plateau_lr = yml["plateau_lr"]
                config.dry_run = yml["dry_run"]
                config.reset_early_stopping = yml["reset_early_stopping"]
                config.enable_shuffle = not yml["no_shuffle"]
                config.mini_epoch = yml["mini_epoch"]
                config.early_stopping = yml["early_stopping"]
                config.checkpoint_save = yml["checkpoint_save"]
                config.subsetpdbs = yml["subsetpdbs"]
                config.energy_weight = yml["energy_weight"]
                config.force_weight = yml["force_weight"]
                config.term_def = yml["term_def"]
                config.embedding = yml["embedding"]
                config.chunk_dataset = yml["chunk_dataset"]
                config.npfile = yml["npfile"]
        config.input = args.input
        config.result = args.result
        config.config = args.config
        config.wd = args.wd
        config.lr = args.lr
        gpus = args.gpus
        config.batch = args.batch
        config.epochs = args.epochs
        config.val_ratio = args.val_ratio
        config.apc = args.apc
        config.cos_anneal = args.cos_anneal
        config.cos_lr = args.cos_lr
        config.exp_lr = args.exp_lr
        config.plateau_lr = args.plateau_lr
        config.dry_run = args.dry_run
        config.reset_early_stopping = args.reset_early_stopping
        config.enable_shuffle = not args.enable_shuffle
        config.mini_epoch = args.mini_epoch
        config.early_stopping = args.early_stopping
        config.checkpoint_save = args.checkpoint_save
        config.subsetpdbs = args.subsetpdbs
        config.energy_weight = args.energy_weight
        config.force_weight = args.force_weight
        config.term_def = args.term_def
        config.embedding = args.embedding
        config.chunk_dataset = args.chunk_dataset
        config.npfile = args.npfile
        config.yaml = args.yaml

        config.energy_matching = config.energy_weight != 0.0
        
        if gpus:
            if gpus == "cpu":
                config.gpu_ids = "cpu"
            else:
                config.gpu_ids = [int(i) for i in gpus.strip().split(",")]
        else:
            config.gpu_ids = "cpu"

        assert config.checkpoint_save >= 0

        assert os.path.isfile(config.result), f"Result directory does not exist: {config.result}"
        assert os.path.isfile(config.config), f"Config file does not exist: {config.config}"
        assert os.path.isdir(config.input), f"Input directory does not exist: {config.input}"
        
        return config
        
        