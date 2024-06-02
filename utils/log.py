import os
from datetime import datetime
from pytz import timezone

import wandb
from yacs.config import CfgNode as CN

from utils.misc import mkdir


def init_wandb(cfg):
    wandb.init(
        project=cfg.WANDB.PROJECT,
        name=cfg.WANDB.NAME,
        job_type=cfg.WANDB.JOB_TYPE,
        notes=cfg.WANDB.NOTES,
        dir=cfg.WANDB.DIR,
        config=cfg
    )
    # save checkpoints and results in the wandb log directory
    cfg.TRAIN.CHECKPOINT_DIR = str(mkdir(os.path.join(cfg.TRAIN.CHECKPOINT_DIR, wandb.run.dir)))
    cfg.RESULT_DIR = str(mkdir(os.path.join(cfg.RESULT_DIR, wandb.run.dir)))
    

def set_time_to_log_dir(cfg: CN):
    formatted_time = datetime.now(timezone('Asia/Seoul')).strftime("%y%m%d-%H%M%S")
    cfg.TRAIN.CHECKPOINT_DIR = str(mkdir(os.path.join(cfg.TRAIN.CHECKPOINT_DIR, formatted_time)))
    cfg.RESULT_DIR = str(mkdir(os.path.join(cfg.RESULT_DIR, formatted_time)))
