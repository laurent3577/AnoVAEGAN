import argparse
from model import AnoVAEGAN
from data import MNISTDataset
from utils import save_batch_output
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import random


def parse_args():
    parser = argparse.ArgumentParser(description='Test  AnoVAEGAN')

    parser.add_argument('--model-path',
                        help='Saved model path',
                        required=True,
                        type=str)
    parser.add_argument('--dataset-dir',
                        help="Dataset directory",
                        default='/tmp/',
                        type=str)
    parser.add_argument('--output-dir',
                        help="Output directory",
                        default='/tmp/',
                        type=str)
    parser.add_argument('--batch-size',
                        help="Batch size",
                        default=16,
                        type=int)
    parser.add_argument('--detect-thresh',
                        help='Detection threshold',
                        default=0.3,
                        type=float)
    parser.add_argument('--n-plot',
                        help='Number of batches to plot and save',
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
        data_dir=args.dataset_dir,
        split='test',
        input_size=config.DATASET.INPUT_SIZE)
    random.shuffle(dataset.data)
    dataset.data = dataset.data[:args.n_plot*args.batch_size]
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    pbar = tqdm(test_loader)
    for i, img in enumerate(pbar):
        img = img.to(device)
        out = model(img)
        save_batch_output(img, out['rec'], args.detect_thresh, args.save_size, args.output_dir, 'test_batch_{}'.format(i))

if __name__ == '__main__':
    main()