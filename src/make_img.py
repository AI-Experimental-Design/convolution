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
    parser.add_argument('-o',
                        '--output',
                        required=True,
                        help='Output image file path')
    parser.add_argument('--height',
                        type=int,
                        default=None,
                        help='Height of the generated image (default: inferred from the input matrix)')
    parser.add_argument('--width',
                        type=int,
                        default=None,
                        help='Width of the generated image (default: inferred from the input matrix)')
    return parser.parse_args()
def main():
    args = get_args()
    img = conv_utils.read_matrix(args.img, dtype=float)
    height = args.height if args.height is not None else img.shape[0]
    width = args.width if args.width is not None else img.shape[1]
    conv_utils.save_image(img,
                          args.output,
                          height=height,
                          width=width)
if __name__ == '__main__':
    main()
