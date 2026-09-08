#!/usr/bin/env python3
"""Negative-set strategy 3: gene_body.

Real genomic windows sampled from inside annotated gene bodies,
offset far enough past each gene's TSS to avoid overlapping the true
positive window. These come from genuinely "promoter-adjacent" DNA
(protein-coding genes) but the wrong location relative to the TSS --
the hardest of the three negative strategies to separate on
composition alone.

Uses pysam.FastaFile for indexed random-access fetching, so the
genome is never loaded fully into memory the way a hand-rolled
"read it all into a dict" parser would -- each sampled window is
pulled straight off disk via an .fai index. GFF parsing is still
plain text (see extract_upstream_tss.py for a pysam.tabix_iterator
version of that side).

Requires: pysam (pip install pysam --break-system-packages)

Usage:
    python make_negatives_gene_body.py \
        -f GCF_000146045.2_R64_genomic.fna \
        -g genomic.gff \
        -m data/dna/tss/upstream/coords.tsv \
        -o data/dna/tss/negatives/gene_body \
        --seed 0
"""
import argparse
import os
import random

import pysam

COMPLEMENT = str.maketrans('ACGTNacgtn', 'TGCANtgcan')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fasta', required=True,
                        help='Genome FASTA file')
    parser.add_argument('-g', '--gff', required=True,
                        help='GFF3 gene annotation file')
    parser.add_argument(
        '-m', '--manifest', required=True,
        help='coords.tsv from extract_upstream_tss.py -- gives the '
             'window length')
    parser.add_argument('-o', '--out_dir', required=True)
    parser.add_argument(
        '-n', type=int, default=None,
        help='Number of negatives to generate (default: number of '
             'positives in --manifest, for balanced classes)')
    parser.add_argument(
        '--buffer', type=int, default=None,
        help='Min bp into a gene body before sampling a window '
             '(default: same as the window length, so these can '
             'never overlap a true positive)')
    parser.add_argument(
        '--biotype', default='protein_coding',
        help='Restrict to this gene_biotype (default '
             'protein_coding); pass "" for no filter')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_tries_per_sample', type=int, default=200)
    return parser.parse_args()


def parse_attributes(field):
    attrs = {}
    for part in field.strip().split(';'):
        if not part or '=' not in part:
            continue
        k, v = part.split('=', 1)
        attrs[k] = v
    return attrs


def read_genes(path, biotype_filter):
    genes = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            cols = line.rstrip('\n').split('\t')
            if len(cols) != 9:
                continue
            (seqid, source, ftype, start, end, score, strand, phase,
             attr_field) = cols
            if ftype != 'gene':
                continue
            attrs = parse_attributes(attr_field)
            if biotype_filter and attrs.get('gene_biotype') != biotype_filter:
                continue
            gene_id = attrs.get('ID') or attrs.get('Name') or attrs.get('gene')
            if gene_id is None:
                continue
            genes.append((gene_id, seqid, int(start), int(end), strand))
    return genes


def read_manifest_length(path):
    lengths = []
    with open(path) as f:
        f.readline()  # header
        for line in f:
            if not line.strip():
                continue
            cols = line.rstrip('\n').split('\t')
            lengths.append(int(cols[-2]))
    if not lengths:
        raise ValueError(f'No rows found in manifest {path}')
    return max(set(lengths), key=lengths.count), len(lengths)


def revcomp(seq):
    return seq.translate(COMPLEMENT)[::-1]


def make_gene_body_windows(genes, fasta, n, length, buffer, rng,
                           max_tries=200):
    fasta_refs = set(fasta.references)
    eligible = []
    for gene_id, chrom, start, end, strand in genes:
        if chrom not in fasta_refs:
            continue
        gene_len = end - start + 1
        max_offset = gene_len - length
        if max_offset < buffer:
            continue  # gene too short to hold a buffered window
        eligible.append((gene_id, chrom, start, end, strand, buffer,
                         max_offset))

    results = []
    used = set()
    tries = 0
    if not eligible:
        return results
    while len(results) < n and tries < n * max_tries:
        tries += 1
        gene_id, chrom, start, end, strand, lo, hi = rng.choice(eligible)
        offset = rng.randint(lo, hi)
        if (gene_id, offset) in used:
            continue
        if strand == '+':
            start0 = (start - 1) + offset
            end0 = start0 + length
            seq = fasta.fetch(chrom, start0, end0).upper()
        else:
            tss0 = end - 1
            start0 = tss0 - offset - length + 1
            end0 = tss0 - offset + 1
            seq = revcomp(fasta.fetch(chrom, start0, end0).upper())
        if 'N' in seq:
            continue
        used.add((gene_id, offset))
        results.append((seq, gene_id, chrom, start0, end0, strand, offset))
    return results


def main():
    args = get_args()
    rng = random.Random(args.seed)

    genes = read_genes(args.gff, args.biotype)
    window_len, n_positives = read_manifest_length(args.manifest)
    n = args.n if args.n is not None else n_positives
    buffer = args.buffer if args.buffer is not None else window_len

    print(f'window length: {window_len}, buffer: {buffer} bp past '
          f'TSS, candidate genes: {len(genes)}')

    with pysam.FastaFile(args.fasta) as fasta:
        results = make_gene_body_windows(
            genes, fasta, n, window_len, buffer, rng,
            max_tries=args.max_tries_per_sample)

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, 'coords.tsv')
    with open(manifest_path, 'w') as manifest:
        manifest.write(
            '#id\tgene_id\tchrom\tstart0\tend0\tstrand\toffset\t'
            'file\tlength\n')
        for i, (seq, gene_id, chrom, start0, end0, strand,
                offset) in enumerate(results):
            fname = f'genebody_{i:05d}.txt'
            with open(os.path.join(args.out_dir, fname), 'w') as out_f:
                out_f.write(seq + '\n')
            manifest.write(
                f'genebody_{i:05d}\t{gene_id}\t{chrom}\t{start0}\t'
                f'{end0}\t{strand}\t{offset}\t{fname}\t{len(seq)}\n')

    if len(results) < n:
        print(f'WARNING: only got {len(results)}/{n} (buffer='
              f'{buffer} may be excluding too many short genes -- '
              f'try lowering it)')
    print(f'wrote {len(results)} gene_body negatives '
          f'(length {window_len}) -> {args.out_dir}')


if __name__ == '__main__':
    main()
