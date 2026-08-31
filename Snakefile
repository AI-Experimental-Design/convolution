SHAPES = ["x", "o"]
KERNELS = ["fwd_diagonal",
           "back_diagonal",
           "topl_curve",
           "botr_curve",
           "set_corner"]

rule all:
    input:
        expand("img/xo/kernel_{kernel}.png", kernel=KERNELS),
        expand("img/xo/{shape}.kernel_{kernel}.png", shape=SHAPES, kernel=KERNELS),
        expand("out/xo/{shape}.kernel_{kernel}.scores.txt", shape=SHAPES, kernel=KERNELS)

rule render_kernel:
    input: "data/xo/kernels/kernel_{kernel}.txt"
    output: "img/xo/kernel_{kernel}.png"
    shell: "python src/make_img.py -i {input} -o {output}"

rule convolve:
    input:
        img="data/xo/inputs/{shape}.txt",
        kernel="data/xo/kernels/kernel_{kernel}.txt"
    output:
        matrix="out/xo/{shape}.kernel_{kernel}.txt",
        scores="out/xo/{shape}.kernel_{kernel}.scores.txt"
    shell:
        "python src/img_conv.py -i {input.img} -k {input.kernel} -o {output.matrix} > {output.scores}"

rule render_result:
    input: "out/xo/{shape}.kernel_{kernel}.txt"
    output: "img/xo/{shape}.kernel_{kernel}.png"
    shell: "python src/make_img.py -i {input} -o {output}"
