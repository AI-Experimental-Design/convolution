#!/usr/bin/env python3
"""Negative-set strategy 1: random_sequence.

Fully synthetic sequences -- each base drawn i.i.d. from the genome's
overall base composition (not uniform 25/25/25/25; see note below).
Contains no real genomic structure at all -- the "easiest" of the
three negative strategies.

Uses pysam.FastaFile to measure base composition one contig at a
time (fasta.fetch(ref) with no start/end pulls a whole contig), so
the genome is never held fully in memory the way a hand-rolled
"read it all into a dict" parser would -- and with --uniform, the
FASTA isn't opened at all, since no composition is needed.

Requires: pysam (pip install pysam --break-system-packages)

Usage:
    python make_negatives_random_sequence.py \
        -f GCF_000146045.2_R64_genomic.fna \
        -m data/dna/tss/upstream/coords.tsv \
        -o data/dna/tss/negatives/random_sequence \
        --seed 0
"""
import argparse
import os
import random

import pysam

BASES = ['A', 'C', 'G', 'T']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--fasta', required=True,
        help='Genome FASTA file (used only to measure base '
             'composition; skipped entirely with --uniform)')
    parser.add_argument(
        '-m', '--manifest', default=None,
        help='coords.tsv from extract_upstream_tss.py, to infer '
             'window length and default -n (number of positives). '
             'Optional if you pass --length and -n directly instead.')
    parser.add_argument('-o', '--out_dir', required=True)
    parser.add_argument(
        '-n', type=int, default=None,
        help='Number of negatives to generate (default: number of '
             'positives in --manifest, for balanced classes)')
    parser.add_argument(
        '--length', type=int, default=None,
        help='Sequence length (default: inferred from --manifest)')
    parser.add_argument(
        '--uniform', action='store_true',
        help='Use uniform 25/25/25/25 base probabilities instead of '
             "the genome's measured composition (makes this an "
             'easier, composition-only baseline -- see script '
             'docstring)')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def read_manifest_length(path):
    """Return (window_len, n_positives) from a coords.tsv manifest."""
    lengths = []
    n = 0
    with open(path) as f:
        f.readline()  # header
        for line in f:
            if not line.strip():
                continue
            cols = line.rstrip('\n').split('\t')
            lengths.append(int(cols[-2]))  # length is second-to-last
            n += 1
    if not lengths:
        raise ValueError(f'No rows found in manifest {path}')
    window_len = max(set(lengths), key=lengths.count)
    return window_len, n


def genome_base_composition(fasta_path):
    counts = {b: 0 for b in BASES}
    total = 0
    with pysam.FastaFile(fasta_path) as fasta:
        for ref in fasta.references:
            seq = fasta.fetch(ref).upper()
            for b in BASES:
                c = seq.count(b)
                counts[b] += c
                total += c
    if total == 0:
        return {b: 0.25 for b in BASES}
    return {b: counts[b] / total for b in BASES}


def make_random_sequence(n, length, base_probs, rng):
    weights = [base_probs[b] for b in BASES]
    return [''.join(rng.choices(BASES, weights=weights, k=length))
            for _ in range(n)]


def main():
    args = get_args()
    rng = random.Random(args.seed)

    length = args.length
    n = args.n
    if args.manifest:
        m_length, m_n = read_manifest_length(args.manifest)
        length = length or m_length
        n = n or m_n
    if length is None:
        raise SystemExit(
            'Need --length or --manifest to determine sequence '
            'length.')
    if n is None:
        raise SystemExit(
            'Need -n or --manifest to determine how many to '
            'generate.')

    if args.uniform:
        base_probs = {b: 0.25 for b in BASES}
        print('base probabilities: uniform (0.25 each)')
    else:
        base_probs = genome_base_composition(args.fasta)
        print('base probabilities (from genome): ' +
              ', '.join(f'{b}={p:.3f}' for b, p in base_probs.items()))

    seqs = make_random_sequence(n, length, base_probs, rng)

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, 'coords.tsv')
    with open(manifest_path, 'w') as manifest:
        manifest.write('#id\tfile\tlength\n')
        for i, seq in enumerate(seqs):
            fname = f'randseq_{i:05d}.txt'
            with open(os.path.join(args.out_dir, fname), 'w') as out_f:
                out_f.write(seq + '\n')
            manifest.write(f'randseq_{i:05d}\t{fname}\t{len(seq)}\n')

    print(f'wrote {len(seqs)} random_sequence negatives '
          f'(length {length}) -> {args.out_dir}')


if __name__ == '__main__':
    main()
