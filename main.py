import os

from models.build import build_model
from utils.parser import parse_args, load_config
from utils.log import init_wandb, set_time_to_log_dir
from datasets.build import update_cfg_from_dataset
from trainer import build_trainer
from predictor import Predictor
from utils.misc import set_seeds, set_devices


def main():
    args = parse_args()
    cfg = load_config(args)
    update_cfg_from_dataset(cfg, cfg.DATA.NAME)

    # select cuda devices
    set_devices(cfg.VISIBLE_DEVICES)

    # set wandb logger
    if cfg.WANDB.ENABLE:
        init_wandb(cfg)
    if cfg.LOG_TIME:
        set_time_to_log_dir(cfg)
        
    with open(os.path.join(cfg.RESULT_DIR, 'config.txt'), 'w') as f:
        f.write(cfg.dump())

    # set random seed
    set_seeds(cfg.SEED)

    # build model
    model = build_model(cfg)

    # build trainer
    trainer = build_trainer(cfg, model)

    if cfg.TRAIN.ENABLE:
        trainer.train()
        
    if cfg.TEST.ENABLE:
        model = trainer.load_best_model()
        
    ############################    MUC    ############################
        if not cfg.TRAIN.ENABLE:
            cfg.TRAIN.MACs_weight = 0.0
            cfg.TRAIN.LASSO_weight = 0.0
            cfg.SOLVER.MAX_EPOCH = 3
            cfg.SOLVER.BASE_LR = 1e-4
            cfg.SOLVER.WARMUP_EPOCHS = 0    
            
            trainer = build_trainer(cfg, model)
            trainer.train()
    ############################    MUC    ############################
            
        predictor = Predictor(cfg, model)
        predictor.predict()
        if cfg.TEST.VIS_ERROR or cfg.TEST.VIS_DATA:
            predictor.visualize()


if __name__ == '__main__':
    main()
