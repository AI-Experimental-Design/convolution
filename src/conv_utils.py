import argparse
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def read_matrix(path, dtype=float):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            rows.append([dtype(x) for x in line.split()])
    return np.array(rows, dtype=dtype)


def pad_image(img, kernel):
    kH, kW = kernel.shape
    pad_h = kH // 2
    pad_w = kW // 2

    padded = np.zeros(
        (img.shape[0] + 2 * pad_h,
         img.shape[1] + 2 * pad_w),
        dtype=float
    )

    padded[pad_h:pad_h + img.shape[0],
           pad_w:pad_w + img.shape[1]] = img

    return padded


def conv2d(img, kernel):
    H, W = img.shape
    kH, kW = kernel.shape

    outH = H - kH + 1
    outW = W - kW + 1

    out = np.zeros((outH, outW), dtype=float)

    for i in range(outH):
        for j in range(outW):
            val = 0.0
            for ki in range(kH):
                for kj in range(kW):
                    val += img[i + ki, j + kj] * kernel[ki, kj]
            out[i, j] = val

    return out


def save_image(arr,
               path,
               height=4,
               width=4,
               vmin=-3,
               vmax=3):
    fig, ax = plt.subplots(figsize=(width, height))

    ax.imshow(arr,
              cmap="RdBu",
              interpolation='nearest',
              vmin=vmin,
              vmax=vmax)

    ax.set_xticks(np.arange(-0.5, arr.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, arr.shape[0], 1), minor=True)
    ax.grid(which='minor')

    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.tick_params(which='both', length=0)




    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)

def save_matrix(arr, path):
    with open(path, "w") as f:
        for row in arr:
            f.write(" ".join(f"{x:.6g}" for x in row))
            f.write("\n")

def main():
    args = get_args()

    img = read_matrix(args.img, dtype=float)
    kernel = read_matrix(args.kernel, dtype=float)

    out = conv2d(img, kernel)

    save_image(img, args.output_prefix + '.input.png')
    save_image(kernel, args.output_prefix + '.kernel.png')
    save_image(out, args.output_prefix + '.output.png')

    save_matrix(out, args.output_prefix \
            + f'.output.{args.mode}.txt')

if __name__ == '__main__':
    main()

