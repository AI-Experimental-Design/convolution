import argparse
import numpy as np
import matplotlib.pyplot as plt
import conv_utils

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i',
                        '--img',
                        required=True,
                        help='Text file with 0/1 image')

    parser.add_argument('-k',
                        '--kernel',
                        required=True,
                        help='Text file with kernel matrix')

    parser.add_argument('-o',
                        '--output',
                        required=True,
                        help='Result file')

    return parser.parse_args()

def main():
    args = get_args()

    img = conv_utils.read_matrix(args.img, dtype=float)
    kernel = conv_utils.read_matrix(args.kernel, dtype=float)

    out = conv_utils.conv2d(img, kernel)

    conv_utils.save_matrix(out, args.output)

     # Pooling
    max_score = np.max(out)
    mean_score = np.mean(out)

    # Sigmoid probabilities
    max_prob = conv_utils.sigmoid(max_score)
    mean_prob = conv_utils.sigmoid(mean_score)

    print("Scores:")
    print(f"  Max pooling  : {max_score:.6f}")
    print(f"  Mean pooling : {mean_score:.6f}")

    print("\nAfter sigmoid (probability of X):")
    print(f"  Max pooling  : {max_prob:.6f}")
    print(f"  Mean pooling : {mean_prob:.6f}")

if __name__ == '__main__':
    main()

