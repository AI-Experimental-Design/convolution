#!/usr/bin/env python3
import argparse
import os
import numpy as np
import conv_utils
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',
                        '--out_dir',
                        required=True)
    parser.add_argument('--x_path',
                        default='data/x.txt',
                        help='Base X example to perturb')
    parser.add_argument('--o_path',
                        default='data/o.txt',
                        help='Base O example to perturb')
    parser.add_argument('--n_per_class',
                        type=int,
                        default=64)
    parser.add_argument('--max_shift',
                        type=int,
                        default=2)
    parser.add_argument('--noise_p',
                        type=float,
                        default=0.02)
    parser.add_argument('--seed',
                        type=int)
    return parser.parse_args()
def translate(img, max_shift=2):
    pad = max_shift
    h, w = img.shape
    padded = np.pad(img, pad, mode='constant', constant_values=0.0)
    sx = np.random.randint(0, 2 * pad + 1)
    sy = np.random.randint(0, 2 * pad + 1)
    return padded[sy:sy+h, sx:sx+w]
def add_noise(img, p=0.02):
    noise = (np.random.rand(*img.shape) < p).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)
def main():
    args = get_args()
    np.random.seed(args.seed)
    os.makedirs(args.out_dir,exist_ok=True)
    base_x = conv_utils.read_matrix(args.x_path, dtype=np.float32)
    base_o = conv_utils.read_matrix(args.o_path, dtype=np.float32)
    train_tsv = os.path.join(args.out_dir, 'train.tsv')
    with open(train_tsv, 'w') as f:
        f.write('# label\tpath\n')
        # label: 1 = X, 0 = O
        for i in range(args.n_per_class):
            img = add_noise(translate(base_x, args.max_shift), args.noise_p)
            path = os.path.join(args.out_dir, f'x_{i:03d}.txt')
            conv_utils.save_matrix(img, path)
            f.write(f'1\t{path}\n')
        for i in range(args.n_per_class):
            img = add_noise(translate(base_o, args.max_shift), args.noise_p)
            path = os.path.join(args.out_dir, f'o_{i:03d}.txt')
            conv_utils.save_matrix(img, path)
            f.write(f'0\t{path}\n')
if __name__ == '__main__':
    main()
