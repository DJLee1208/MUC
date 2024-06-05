# reference: https://github.com/facebookresearch/SlowFast
import torch
import utils.lr_policy as lr_policy
from utils.sam import SAM

def get_optimizer(optim_params, cfg):
    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == 'adam':
        return torch.optim.Adam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == 'Radam':
        return torch.optim.RAdam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == 'adamW':
        return torch.optim.AdamW(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == 'sam_adam':
        base_optimizer = torch.optim.Adam 
        optimizer = SAM(
            optim_params, base_optimizer, 
            lr=cfg.SOLVER.BASE_LR, 
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            rho=0.05,
            adaptive=False
        )
        return optimizer
    elif cfg.SOLVER.OPTIMIZING_METHOD == 'sam_Radam':
        base_optimizer = torch.optim.RAdam 
        optimizer = SAM(
            optim_params, base_optimizer, 
            lr=cfg.SOLVER.BASE_LR, 
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            rho=0.05,
            adaptive=False
        )
        return optimizer
    elif cfg.SOLVER.OPTIMIZING_METHOD == 'sam_adamW':
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(
            optim_params, base_optimizer, 
            lr=cfg.SOLVER.BASE_LR, 
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            rho=0.05,
            adaptive=False
        )
        return optimizer
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )


def construct_optimizer(model, cfg):
    optim_params = get_param_groups(model)
    optimizer = get_optimizer(optim_params, cfg)

    return optimizer


def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decay.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    if hasattr(cfg.SOLVER, 'LR_POLICY'):
        return lr_policy.get_lr_at_epoch(cfg, cur_epoch)
    else:
        return cfg.SOLVER.BASE_LR


def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for idx, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = new_lr


def get_param_groups(model):
    param_groups = [p for n, p in model.named_parameters() if p.requires_grad]

    return param_groups
