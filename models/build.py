import torch

from models import Autoformer, iTransformer, Crossformer, iTransformer_MUC

def build_model(cfg):
    assert cfg.MODEL_NAME in globals(), f"model {cfg.MODEL_NAME} is not defined"
    model_class = getattr(globals()[cfg.MODEL_NAME], "Model")
    model = model_class(cfg.MODEL)

    if torch.cuda.is_available():
        model = model.cuda()

    return model
