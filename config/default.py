from yacs.config import CfgNode as CN
import os

_C = CN()
_C.OUTPUT_DIR = os.path.join(os.environ.get("HOME"), "Downloads")
_C.EXP_NAME = ''
_C.VISDOM = True
## Loss functions
_C.LOSS = CN()
_C.LOSS.REC_LOSS = 'L1'
_C.LOSS.REC_LOSS_COEFF = 1
_C.LOSS.PRIOR_LOSS = 'KLD'
_C.LOSS.PRIOR_LOSS_COEFF = 0.01
_C.LOSS.ADV_LOSS = 'BCE'
_C.LOSS.ADV_LOSS_COEFF = 1

_C.MODEL = CN()
_C.MODEL.IN_CHANNELS = 3
_C.MODEL.N_LAYERS = 3
_C.MODEL.N_FEATURES = 64

_C.OPTIM = CN()
_C.OPTIM.OPTIMIZER = "AdamW"
_C.OPTIM.BASE_LR = 0.001
_C.OPTIM.DISC_LR = 0.0001
_C.OPTIM.WEIGHT_DECAY = 0.01
_C.OPTIM.EPOCH=10
_C.OPTIM.BATCH_SIZE = 16

_C.DATASET = CN()
_C.DATASET.INPUT_SIZE = (64, 64)

_C.PLOT_EVERY = 20

def update_config(cfg, args):
    cfg.defrost()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()