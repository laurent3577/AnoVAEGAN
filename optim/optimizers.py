from torch import optim


def build_opt(config, model):
    if config.OPTIM.OPTIMIZER == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.OPTIM.BASE_LR,
            weight_decay=config.OPTIM.WEIGHT_DECAY
        )
    elif config.OPTIM.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.OPTIM.BASE_LR,
            weight_decay=config.OPTIM.WEIGHT_DECAY,
            momentum=0.9,
            nesterov=True
        )
    elif config.OPTIM.OPTIMIZER == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.OPTIM.BASE_LR,
            weight_decay=config.OPTIM.WEIGHT_DECAY
        )
    else:
        raise ValueError("{} unknown optimizer type".format(config.OPTIM.OPTIMIZER))
    return optimizer
