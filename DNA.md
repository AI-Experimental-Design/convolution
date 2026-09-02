# DNA Sequence Convolution

## Data representation, One-hot-encoding

A DNA sequence is a string of nucleotide bases represented by the letters A, C,
G, and T, but convolution requires numerical input. In machine learning,
one-hot encoding is the standard method for dealing with categorical data,
where:
- the input is represented by a bit string
- the bit string length equals the number of categories
- each bit corresponds to one category
- the set bit identifies the selected category
- only one bit can be set

For DNA seqeunce a one-hot-encoding is
| Base| One-hot vector |
|-|-|
| `A`	| `1 0 0 0` |
| `C`	| `0 1 0 0` |
| `G`	| `0 0 1 0` |
| `T`	| `0 0 0 1` |


With this one-hot encoding, a DNA sequnce of lenght L become a 4xL matrix

| Sequence | One-hot representation |
|-|-|
| `GCATATAAATCG` | `0 0 1 0 1 0 1 1 1 0 0 0` <br> `0 1 0 0 0 0 0 0 0 0 1 0` <br> `1 0 0 0 0 0 0 0 0 0 0 1` <br> `0 0 0 1 0 1 0 0 0 1 0 0` |

