import argparse
from model import AnoVAEGAN
from data import MNISTDataset
from utils import combine_detect_image, save_compare
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Test  AnoVAEGAN')

    parser.add_argument('--model-path',
                        help='Saved model path',
                        required=True,
                        type=str)
    parser.add_argument('--output-dir',
                        help="Output directory",
                        default='/tmp/',
                        type=str)
    parser.add_argument('--batch-size',
                        help="Batch size",
                        default=32,
                        type=int)
    parser.add_argument('--detect-thresh',
                        help='Detection threshold',
                        default=0.85,
                        type=float)
    parser.add_argument('--n-plot',
                        help='Number of images to plot and save',
                        default=12,
                        type=int)
    parser.add_argument('--save-size',
                        help='Size of saved images',
                        default=128,
                        type=int)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.model_path)
    config = checkpoint['cfg']
    model = AnoVAEGAN(config.MODEL.IN_CHANNELS, config.MODEL.N_LAYERS, config.MODEL.N_FEATURES)
    model.load_state_dict(checkpoint['params'])
    model.to(device)
    model.eval()

    dataset = MNISTDataset(
        split='test',
        input_size=config.DATASET.INPUT_SIZE)

    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    pbar = tqdm(test_loader)
    comp = []
    for img in pbar:
        img = img.to(device)
        out = model(img)
        detection = (torch.abs(out['rec']-img)>args.detect_thresh).int()
        img, detect_image, rec = combine_detect_image(img, detection, out['rec'])
        comp += [(i,d, r) for i,d, r in zip(img, detect_image, rec)]

    save_compare(args.save_size, comp[:args.n_plot], args.output_dir)

if __name__ == '__main__':
    main()