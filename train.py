import argparse
from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader
from optim import build_loss, build_opt, rec_loss_map
from model import AnoVAEGAN, Discriminator
from config import config, update_config
from data import MNISTDataset
from utils import ExpAvgMeter, Plotter, save_batch_output


def parse_args():
    parser = argparse.ArgumentParser(description='Train  AnoVAEGAN')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
    print(config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = AnoVAEGAN(config.MODEL.IN_CHANNELS, config.MODEL.N_LAYERS, config.MODEL.N_FEATURES)
    discriminator = Discriminator(config.MODEL.IN_CHANNELS, config.DATASET.INPUT_SIZE, config.MODEL.N_LAYERS,  config.MODEL.N_FEATURES)
    model.to(device)
    discriminator.to(device)
    model.train()
    discriminator.train()
    print(model)

    dataset = MNISTDataset(
        data_dir=config.DATASET.DIR,
        split='train',
        input_size=config.DATASET.INPUT_SIZE,
        transforms=[
            ("HorizontalFlip", None),
            ("Rotation", {"degrees":30})
        ])

    train_loader =DataLoader(dataset, batch_size=config.OPTIM.BATCH_SIZE, shuffle=True)

    model_opt, model_scheduler = build_opt(config, model, len(train_loader))
    disc_opt, disc_scheduler = build_opt(config, discriminator, len(train_loader), discriminator=True)

    rec_loss = build_loss(config.LOSS.REC_LOSS)
    prior_loss = build_loss(config.LOSS.PRIOR_LOSS)
    adv_loss = build_loss(config.LOSS.ADV_LOSS)
    rec_key = rec_loss_map[config.LOSS.REC_LOSS]

    loss_meter = ExpAvgMeter(0.98)
    rec_loss_meter = ExpAvgMeter(0.98)
    prior_loss_meter = ExpAvgMeter(0.98)
    adv_loss_meter = ExpAvgMeter(0.98)
    if config.VISDOM:
        plotter = Plotter(log_to_filename=os.path.join(config.OUTPUT_DIR, "logs.viz"))

    step = 0
    for e in range(config.OPTIM.EPOCH):
        pbar = tqdm(train_loader)
        for img in pbar:
            step += 1
            img = img.to(device)
            valid = torch.full((img.shape[0],1), 1, dtype=torch.float, device=device)
            fake = torch.full((img.shape[0],1), 0, dtype=torch.float, device=device)
            out = model(img)
            rec_l = rec_loss(out[rec_key], img)
            prior_l = prior_loss(out['mu'], out['logvar'])
            adv_l = adv_loss(discriminator(out['rec']), valid)
            loss =  config.LOSS.REC_LOSS_COEFF * rec_l \
                    + config.LOSS.PRIOR_LOSS_COEFF * prior_l \
                    + config.LOSS.ADV_LOSS_COEFF * adv_l

            model_opt.zero_grad()
            loss.backward()
            model_opt.step()
            if model_scheduler.update_on_step:
                model_scheduler.step()

            real_loss = adv_loss(discriminator(img), valid)
            fake_loss = adv_loss(discriminator(out['rec'].detach()), fake)
            disc_loss = 0.5 * (real_loss + fake_loss)

            disc_opt.zero_grad()
            disc_loss.backward()
            disc_opt.step()
            if disc_scheduler.update_on_step:
                disc_scheduler.step()

            loss_meter.update(float(loss.data))
            rec_loss_meter.update(float(rec_l.data))
            prior_loss_meter.update(float(prior_l.data))
            adv_loss_meter.update(float(adv_l.data))

            pbar.set_description('Train Epoch : {0}/{1} Loss : {2:.4f} '.format(e+1, config.OPTIM.EPOCH, loss_meter.value))

            if config.VISDOM and step%config.PLOT_EVERY == 0:
                plotter.plot("Loss", step, loss_meter.value, "Loss", "Step", "Value")
                plotter.plot("Loss", step, rec_loss_meter.value, "Rec loss", "Step", "Value")
                plotter.plot("Loss", step, prior_loss_meter.value, "Prior loss", "Step", "Value")
                plotter.plot("Loss", step, adv_loss_meter.value, "Adv loss", "Step", "Value")
                model_lr = model_opt.param_groups[0]['lr']
                disc_lr = disc_opt.param_groups[0]['lr']
                plotter.plot("LR", step, model_lr, "Model LR", "Step", "Value")
                plotter.plot("LR", step, disc_lr, "Discr LR", "Step", "Value")
            if config.DEBUG.USE and step%config.DEBUG.DEBUG_EVERY == 0:
                save_batch_output(
                    img,
                    out['rec'],
                    config.DEBUG.DETECT_THRESH,
                    config.DEBUG.SAVE_SIZE,
                    config.OUTPUT_DIR,
                    'batch_{}'.format(step))
        if not model_scheduler.update_on_step:
            model_scheduler.step()
        if not disc_scheduler.update_on_step:
            disc_scheduler.step()
        save_path = os.path.join(config.OUTPUT_DIR, config.EXP_NAME + "_checkpoint.pth")
        torch.save({
            'cfg':config,
            'params':model.state_dict()}, save_path)


if __name__ == '__main__':
    main()