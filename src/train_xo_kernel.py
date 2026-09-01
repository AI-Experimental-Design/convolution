#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import conv_utils

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',
                        required=True,
                        help='TSV: label<TAB>path/to/image.txt (label 0/1)')

    parser.add_argument('--kernel',
                        type=int,
                        default=3,
                        help='kernel size (square)')

    parser.add_argument('--pool',
                        choices=['max', 'mean'],
                        default='max')

    parser.add_argument('--epochs',
                        type=int,
                        default=200)

    parser.add_argument('--lr',
                        type=float,
                        default=0.1)

    parser.add_argument('--seed',
                        type=int,
                        default=0)

    parser.add_argument('--out_prefix',
                        required=True,
                        help='Prefix for outputs (kernel txt/png, logs)')

    return parser.parse_args()

def load_tsv(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            y_s, img_path = line.split('\t')
            y = float(y_s)
            img = conv_utils.read_matrix(img_path, dtype=float).astype(np.float32)
            items.append((y, img_path, img))
    return items

class OneKernelNet(nn.Module):
    def __init__(self, k=3, pool='max'):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=k, padding=k//2, bias=True)
        self.pool = pool

    def forward(self, x):
        z = self.conv(x)           # (N,1,H,W)
        #z = torch.relu(z)
        if self.pool == 'max':
            s = torch.amax(z, dim=(2,3))   # (N,1)
        else:
            s = torch.mean(z, dim=(2,3))   # (N,1)
        return s.squeeze(1), z.squeeze(1)  # logits (N,), convmap (N,H,W)

def main():
    args = get_args()

    torch.manual_seed(args.seed)

    np.random.seed(args.seed)

    data = load_tsv(args.train)
    ys = torch.tensor([y for y, _, _ in data], dtype=torch.float32)
    X = torch.tensor(np.stack([img for _, _, img in data]),dtype=torch.float32)[:, None, :, :]  # (N,1,H,W)

    model = OneKernelNet(k=args.kernel, pool=args.pool)
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()

        logits, _ = model(X)
        loss = loss_fn(logits, ys)
        loss.backward()
        opt.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                acc = (preds == ys).float().mean().item()
                # also report mean prob per class (nice sanity check)
                x_mean = probs[ys == 1].mean().item()
                o_mean = probs[ys == 0].mean().item()
            print(f'epoch {epoch:04d} loss={loss.item():.4f} acc={acc:.3f}  meanP(X)={x_mean:.3f} meanP(O)={o_mean:.3f}')

    # Save learned kernel
    w = model.conv.weight.detach().cpu().numpy()[0, 0]   # (k,k)
    b = float(model.conv.bias.detach().cpu().numpy()[0])
    conv_utils.save_matrix(w,
                           args.out_prefix + '.kernel.txt')
    conv_utils.save_image(w,
                          args.out_prefix + '.kernel.png',
                          height=3,
                          width=3,
                          vmin=-np.max(np.abs(w)),
                          vmax=np.max(np.abs(w)))
    print('learned bias:', b)

if __name__ == '__main__':
    main()
