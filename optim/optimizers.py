from torch import optim


def build_opt(config, model, discriminator=False):
    lr = config.OPTIM.DISC_LR if discriminator else config.OPTIM.BASE_LR
    weight_decay = config.OPTIM.WEIGHT_DECAY
    if config.OPTIM.OPTIMIZER == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif config.OPTIM.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
            nesterov=True
        )
    elif config.OPTIM.OPTIMIZER == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError("{} unknown optimizer type".format(config.OPTIM.OPTIMIZER))
    return optimizer
