import argparse
import numpy as np
import conv_utils

BASES = ['A', 'C', 'G', 'T']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--seq',
                        required=True,
                        help='Text file with a single DNA sequence (A/C/G/T)')
    parser.add_argument('-k',
                        '--kernel',
                        required=True,
                        help='Text file with kernel matrix (4 rows: A,C,G,T)')
    parser.add_argument('-o',
                        '--output',
                        help='Result file')
    parser.add_argument('-b',
                        '--bias',
                        type=float,
                        default=0.0,
                        help='Bias to add to the pooled score before sigmoid (e.g. from a learned kernel)')
    return parser.parse_args()


def read_sequence(path):
    """Read a single DNA sequence from a text file (any whitespace/case is normalized)."""
    with open(path) as f:
        seq = f.read().strip().upper()
    bad = sorted(set(seq) - set(BASES))
    if bad:
        raise ValueError(f'Sequence contains non-ACGT characters: {bad}')
    return seq


def one_hot(seq):
    """Encode a sequence as a 4xL matrix: one row per base (A,C,G,T), one column per position."""
    m = np.zeros((4, len(seq)))
    for col, base in enumerate(seq):
        m[BASES.index(base), col] = 1.0
    return m


def conv1d(onehot, kernel):
    """'Valid' cross-correlation along the sequence axis: same convention as
    conv_utils.conv2d (no padding, no kernel flip), just collapsed to one
    sliding axis since the kernel always spans all 4 base rows at once."""
    n_channels, seq_len = onehot.shape
    k_channels, k_len = kernel.shape
    if k_channels != n_channels:
        raise ValueError(f'Kernel has {k_channels} rows, expected {n_channels} (one per base: A,C,G,T)')
    out_len = seq_len - k_len + 1
    if out_len <= 0:
        raise ValueError('Kernel is longer than the sequence.')
    out = np.zeros(out_len)
    for i in range(out_len):
        out[i] = np.sum(onehot[:, i:i + k_len] * kernel)
    return out


def main():
    args = get_args()
    seq = read_sequence(args.seq)
    onehot = one_hot(seq)
    kernel = conv_utils.read_matrix(args.kernel, dtype=float)
    out = conv1d(onehot, kernel)

    if args.output:
        conv_utils.save_matrix(out, args.output)

     # Pooling
    max_score = np.max(out)
    mean_score = np.mean(out)
    # Sigmoid probabilities (bias applied here, not to the raw feature map --
    # for max/mean pooling the two are mathematically equivalent, and this
    # way the "Scores:" printout above stays the raw, bias-free convolution output)
    max_prob = conv_utils.sigmoid(max_score + args.bias)
    mean_prob = conv_utils.sigmoid(mean_score + args.bias)
    print("Scores:")
    print(f"  Max pooling  : {max_score:.6f}")
    print(f"  Mean pooling : {mean_score:.6f}")
    if args.bias != 0.0:
        print(f"  (bias = {args.bias:.6f}, added before sigmoid only)")
    print("\nAfter sigmoid (probability of motif):")
    print(f"  Max pooling  : {max_prob:.6f}")
    print(f"  Mean pooling : {mean_prob:.6f}")


if __name__ == '__main__':
    main()
