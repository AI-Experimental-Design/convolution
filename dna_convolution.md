# DNA Sequence Convolution
## Data representation
### DNA Sequence One-hot Encoding
A DNA sequence is a string of nucleotide bases represented by the letters A, C,
G, and T, but convolution requires numerical input. In machine learning,
one-hot encoding is the standard method for dealing with categorical data,
where:
- the input is represented by a bit string
- the bit string length equals the number of categories
- each bit corresponds to one category
- the set bit identifies the selected category
- only one bit can be set

For a DNA sequence, the one-hot encoding is:

| Base | One-hot vector |
|-|-|
| `A` | `1 0 0 0` |
| `C` | `0 1 0 0` |
| `G` | `0 0 1 0` |
| `T` | `0 0 0 1` |

With this one-hot encoding, a DNA sequence of length L becomes a 4×L matrix:

| Sequence | One-hot representation |
|-|-|
| `GCATATAAATCG` | `0 0 1 0 1 0 1 1 1 0 0 0` <br> `0 1 0 0 0 0 0 0 0 0 1 0` <br> `1 0 0 0 0 0 0 0 0 0 0 1` <br> `0 0 0 1 0 1 0 0 0 1 0 0` |

### TATA box kernel
The TATA box is a sequence found in the promoter region of many eukaryotic genes.
The classical consensus sequence is `TATA(A/T)A(A/T)`, which can be represented
by the following kernel:
```
0 1 0 1 1 1 1
0 0 0 0 0 0 0
0 0 0 0 0 0 0
1 0 1 0 1 0 1
```

Note that this kernel is not a one-hot encoding. The variable base positions at
5 and 7, which can be either A or T, have both of those bits set in the kernel.
While our inputs will be one-hot encodings, the kernel is a set of weights and
is not required to be a one-hot encoding.

| Label | Sequence | Feature Map | Max Pooling | Mean Pooling | P(TATA) max | P(TATA) mean |
|-|-|-|-|-|-|-|
| TATA box    | `GCTATAAATCG` | `5  3  7  2  4` | 7 | 4.2 | 0.999 | 0.985 |
| TATA box    | `ACTATATAATG` | `5  2  7  3  5` | 7 | 4.4 | 0.999 | 0.988 |
| No TATA box | `CGGCATCGGAC` | `1  2  0  3  1` | 3 | 1.4 | 0.953 | 0.802 |
| No TATA box | `ATCGGCTAGCA` | `1  2  2  1  3` | 3 | 1.8 | 0.953 | 0.858 |

<details>

```
python src/dna_conv.py \
    -i data/dna/tata_ex/inputs/tata_box_0.txt \
    -k data/dna/tata_ex/kernels/kernel_tata_box.txt 
Scores:
    Max pooling : 7.000000
    Mean pooling : 4.200000
After sigmoid (probability of motif):
    Max pooling : 0.999089
    Mean pooling : 0.985226

python src/dna_conv.py
    -i data/dna/tata_ex/inputs/tata_box_1.txt \
    -k data/dna/tata_ex/kernels/kernel_tata_box.txt 
Scores:
    Max pooling : 7.000000
    Mean pooling : 4.400000
After sigmoid (probability of motif):
    Max pooling : 0.999089
    Mean pooling : 0.987872

python src/dna_conv.py \
    -i data/dna/tata_ex/inputs/no_tata_box_0.txt \
    -k data/dna/tata_ex/kernels/kernel_tata_box.txt
Scores:
    Max pooling : 3.000000
    Mean pooling : 1.400000
After sigmoid (probability of motif):
    Max pooling : 0.952574
    Mean pooling : 0.802184

python src/dna_conv.py \
    -i data/dna/tata_ex/inputs/no_tata_box_1.txt \
    -k data/dna/tata_ex/kernels/kernel_tata_box.txt 
Scores:
    Max pooling : 3.000000
    Mean pooling : 1.800000
After sigmoid (probability of motif):
    Max pooling : 0.952574
    Mean pooling : 0.858149
```
</details>

This TATA kernel successfully differentiated between sequences that contain a
TATA box and those that don't, correctly matching two different valid instances
of the degenerate consensus.  Max pooling asks whether the motif appears
anywhere in the sequence, and the TATA-box sequences had max pool scores of 7,
while neither non-TATA sequence had a max score over 3. Mean pooling asks how
motif-like the sequence is on average. This also separated the two, with scores
around 4 and 2.

The differentiation is less clear among the post-sigmoid probabilities, but
this is expected with uncalibrated scores. In training, the model will learn a
bias that recenters the sigmoid so match and non-match cases separate more
clearly.

## Training

There is a lot of genomic data, so there is no need here for data augmentation.
We just need to pick positive and negative examples. Since TATA boxes are in
the gene promoter region, we can use regions that are upstream of a gene as
positives. So all we need is a reference genome and a set of gene annotations.
NCBI has thousands of these. Here we will use yeast since it is small (12 Mb),
well characterized, and there are many strains to choose from. For training we
will use the most common lab strain, Saccharomyces cerevisiae. Yeast promoter
biology is also some of the best studied, so we can compare our results to the
literature. For example, only about 20% of the yeast genes have a TATA box,
while the rest use other structures.
([Basehoar et.  al](https://linkinghub.elsevier.com/retrieve/pii/S0092867404002053)).





### Generate training set

- Get the reference sequences and gene annotations from NCBI
  - https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000146045.2/
  - Get the RefSeq FASTA and GFF
  - move `ncbi_dataset.zip` into `data/dna/yeast/` and unzip
- Get positive set
  - Upstream-of-TSS regions
  
    </details>
    ```
    dir="data/dna/yeast/ncbi_dataset/data/GCF_000146045.2"
    python src/extract_upstream_tss.py \
      -f $dir/GCF_000146045.2_R64_genomic.fna \
      -g $dir/genomic.gff \
      -o data/dna/yeast/training/tss/upstream \
      --upstream 150 \
      --downstream 50
    genes in GFF (after biotype filter): 6021
      written             : 6020
      dropped (bad chrom) : 0
      dropped (edge/OOB)  : 1
    manifest: data/dna/yeast/training/tss/upstream/coords.tsv
    ```
    </details>

- Get negative sets
  - Random sequence
    </details>
    ```
    dir="data/dna/yeast/ncbi_dataset/data/GCF_000146045.2"
    python src/make_negatives_random_sequence.py \
        -f $dir/GCF_000146045.2_R64_genomic.fna \
        -m data/dna/yeast/training/tss/upstream/coords.tsv \
        -o data/dna/yeast/training/tss/negatives/random_sequence \
        --seed 0
    base probabilities (from genome): A=0.310, C=0.191, G=0.191, T=0.309
    wrote 6020 random_sequence negatives (length 200) -> data/dna/yeast/training/tss/negatives/random_sequence
    ```
    </details>
  - Random intervals
    </details>
    ```
    dir="data/dna/yeast/ncbi_dataset/data/GCF_000146045.2"
    python src/make_negatives_random_interval.py \
        -f $dir/GCF_000146045.2_R64_genomic.fna \
        -m data/dna/yeast/training/tss/upstream/coords.tsv \
        -o data/dna/yeast/training/tss/negatives/random_interval \
        --seed 0
    wrote 6020 random_interval negatives (length 200) -> data/dna/yeast/training/tss/negatives/random_interval
    ```
    </details>
  - Gene body
    </details>
    ```
    dir="data/dna/yeast/ncbi_dataset/data/GCF_000146045.2"
    python src/make_negatives_gene_body.py \
        -f $dir/GCF_000146045.2_R64_genomic.fna \
        -g $dir/genomic.gff \
        -m data/dna/yeast/training/tss/upstream/coords.tsv \
        -o data/dna/yeast/training/tss/negatives/gene_body \
        --seed 0
    window length: 200, buffer: 200 bp past TSS, candidate genes: 6021
    wrote 6020 gene_body negatives (length 200) -> data/dna/yeast/training/tss/negatives/gene_body
    ```
    </details>
