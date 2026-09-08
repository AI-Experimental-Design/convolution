#!/usr/bin/env python3
"""Negative-set strategy 2: random_interval.

Real genomic windows, sampled from anywhere in the genome (weighted
by chromosome length), excluding any window that overlaps a true
positive TSS window. Real sequence statistics, but not targeted at
any particular biological context.

Uses pysam.FastaFile for indexed random-access fetching, so the
genome is never loaded fully into memory the way a hand-rolled
"read it all into a dict" parser would -- each sampled window is
pulled straight off disk via an .fai index.

Requires: pysam (pip install pysam --break-system-packages)

Usage:
    python make_negatives_random_interval.py \
        -f GCF_000146045.2_R64_genomic.fna \
        -m data/dna/tss/upstream/coords.tsv \
        -o data/dna/tss/negatives/random_interval \
        --seed 0
"""
import argparse
import os
import random

import pysam


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fasta', required=True,
                        help='Genome FASTA file')
    parser.add_argument(
        '-m', '--manifest', required=True,
        help='coords.tsv from extract_upstream_tss.py -- gives the '
             'window length and the true-positive regions to '
             'exclude')
    parser.add_argument('-o', '--out_dir', required=True)
    parser.add_argument(
        '-n', type=int, default=None,
        help='Number of negatives to generate (default: number of '
             'positives in --manifest, for balanced classes)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_tries_per_sample', type=int, default=200)
    return parser.parse_args()


def read_manifest(path):
    """Return (window_len, {chrom: [(start0, end0), ...]}) from
    coords.tsv."""
    excluded = {}
    lengths = []
    with open(path) as f:
        f.readline()  # header
        for line in f:
            if not line.strip():
                continue
            cols = line.rstrip('\n').split('\t')
            gene_id, chrom, strand, tss, w_start, w_end, length, \
                fname = cols
            excluded.setdefault(chrom, []).append(
                (int(w_start), int(w_end)))
            lengths.append(int(length))
    if not lengths:
        raise ValueError(f'No rows found in manifest {path}')
    window_len = max(set(lengths), key=lengths.count)
    for chrom in excluded:
        excluded[chrom].sort()
    return window_len, excluded


def overlaps_excluded(chrom, start0, end0, excluded):
    for ex_start, ex_end in excluded.get(chrom, []):
        if start0 < ex_end and end0 > ex_start:
            return True
    return False


def make_random_intervals(fasta, n, length, excluded, rng, max_tries=200):
    chroms = list(fasta.references)
    chrom_lens = [fasta.get_reference_length(c) for c in chroms]
    results = []
    tries = 0
    while len(results) < n and tries < n * max_tries:
        tries += 1
        chrom = rng.choices(chroms, weights=chrom_lens, k=1)[0]
        clen = fasta.get_reference_length(chrom)
        if clen < length:
            continue
        start0 = rng.randint(0, clen - length)
        end0 = start0 + length
        if overlaps_excluded(chrom, start0, end0, excluded):
            continue
        seq = fasta.fetch(chrom, start0, end0).upper()
        if 'N' in seq:
            continue
        results.append((seq, chrom, start0, end0))
    return results


def main():
    args = get_args()
    rng = random.Random(args.seed)

    window_len, excluded = read_manifest(args.manifest)
    n = args.n if args.n is not None else sum(
        len(v) for v in excluded.values())

    with pysam.FastaFile(args.fasta) as fasta:
        results = make_random_intervals(
            fasta, n, window_len, excluded, rng,
            max_tries=args.max_tries_per_sample)

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, 'coords.tsv')
    with open(manifest_path, 'w') as manifest:
        manifest.write('#id\tchrom\tstart0\tend0\tfile\tlength\n')
        for i, (seq, chrom, start0, end0) in enumerate(results):
            fname = f'randint_{i:05d}.txt'
            with open(os.path.join(args.out_dir, fname), 'w') as out_f:
                out_f.write(seq + '\n')
            manifest.write(
                f'randint_{i:05d}\t{chrom}\t{start0}\t{end0}\t'
                f'{fname}\t{len(seq)}\n')

    if len(results) < n:
        print(f'WARNING: only got {len(results)}/{n} (ran out of '
              f'retries avoiding true-positive overlaps / '
              f'N-containing regions)')
    print(f'wrote {len(results)} random_interval negatives '
          f'(length {window_len}) -> {args.out_dir}')


if __name__ == '__main__':
    main()
