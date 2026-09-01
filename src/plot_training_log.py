#!/usr/bin/env python3
"""Plot loss/acc/meanP(X)/meanP(O) from a train_xo_kernel.py log file."""
import argparse
import re
import matplotlib
import matplotlib.pyplot
from matplotlib import rcParams


rcParams['font.family'] = 'Arial'
rcParams['legend.numpoints'] = 1

LINE_RE = re.compile(
    r'epoch\s+(\d+)\s+loss=([\d.]+)\s+acc=([\d.]+)\s+'
    r'meanP\(X\)=([\d.]+)\s+meanP\(O\)=([\d.]+)'
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--log_file', required=True,
                        help='Log file with train_xo_kernel.py epoch lines')
    parser.add_argument('-o', '--output', required=True,
                        help='Output image path')
    parser.add_argument('--colors', default='blue,green,red,magenta',
                        help='Color CSV for loss,acc,meanP(X),meanP(O)')
    parser.add_argument('--plot_width', type=float, default=4)
    parser.add_argument('--plot_height', type=float, default=3)
    parser.add_argument('--title')
    return parser.parse_args()


def parse_log(path):
    epochs, loss, acc, p_x, p_o = [], [], [], [], []
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            e, l, a, px, po = m.groups()
            epochs.append(int(e))
            loss.append(float(l))
            acc.append(float(a))
            p_x.append(float(px))
            p_o.append(float(po))
    return epochs, loss, acc, p_x, p_o


def main():
    args = get_args()
    epochs, loss, acc, p_x, p_o = parse_log(args.log_file)
    if not epochs:
        raise SystemExit(f'No epoch lines found in {args.log_file}')

    colors = args.colors.split(',')
    series = [
        ('loss', loss, colors[0]),
        ('acc', acc, colors[1]),
        ('meanP(X)', p_x, colors[2]),
        ('meanP(O)', p_o, colors[3]),
    ]

    fig = matplotlib.pyplot.figure(figsize=(args.plot_width, args.plot_height), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=8, width=0.5, length=2)

    plts = []
    for label, values, color in series:
        p, = ax.plot(epochs, values, '-', color=color, linewidth=1)
        plts.append(p)

    ax.legend(plts, [s[0] for s in series], frameon=False, fontsize=8)
    if args.title:
        ax.set_title(args.title, fontsize=8)
    ax.set_xlabel('epoch', fontsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    matplotlib.pyplot.savefig(args.output, bbox_inches='tight')


if __name__ == '__main__':
    main()
