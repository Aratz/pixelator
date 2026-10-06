"""Generate the synthetic PNA fastq files used by the nf-core/pixelator tests.

Every cell gets one of ``N_CELL_TYPES`` cell types, cycled within each hashing
index, so every sample carries every cell type. The cell types boost the same
marker blocks in every pool and sample, which lets clustering recover them
across samples. ``N_SHARED_MARKERS`` markers are boosted in every cell, so the
Experiment Summary finds proximity scores for them in most cells of a sample.

- proxiome-v2: ``pool1`` and ``pool2`` each carry two samples, told apart by
  hashing indices 1 and 5. ``pool1_reseq`` resequences the ``pool1`` edge list.
- proxiome-v1: ``sample1`` and ``sample2`` are separate libraries without
  hashing. ``sample1_reseq`` resequences the ``sample1`` edge list.

Run from the repository root with:

    uv run python -m tests.common.data_generator.nf_core_test_data --output-dir <dir>

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from pixelator.pna.config import pna_config
from tests.common.data_generator.molecules import (
    _hashing_mask,
    _marker_probabilities,
    generate_edgelist,
)
from tests.common.data_generator.reads import write_pna_fastq

N_NODES = 200
N_EDGES = 600
MIN_NEIGHBORS = 20
N_CELL_TYPES = 3
CELL_TYPE_EFFECT = 10.0
N_SHARED_MARKERS = 3
HASHING_FRACTION = 0.2
HASHING_INDICES = [1, 5]
READS_PER_EDGE = 2
QUALITY_STD = 0.0

# name -> (assay, panel, n_cells, n_crossing_edges, edge list seed, fastq seeds)
LIBRARIES = {
    "pool1": (
        "proxiome-v2",
        "proxiome-v2-immuno-155-prerelease",
        48,
        200,
        1,
        {"pool1": 11, "pool1_reseq": 12},
    ),
    "pool2": (
        "proxiome-v2",
        "proxiome-v2-immuno-155-prerelease",
        48,
        200,
        5,
        {"pool2": 21},
    ),
    "sample1": (
        "proxiome-v1",
        "proxiome-v1-immuno-155-v1.1",
        18,
        100,
        3,
        {"sample1": 31, "sample1_reseq": 32},
    ),
    "sample2": (
        "proxiome-v1",
        "proxiome-v1-immuno-155-v1.1",
        18,
        100,
        4,
        {"sample2": 41},
    ),
}


def cell_type_blocks(panel) -> list[set[str]]:
    """Markers boosted by each cell type only, excluding the shared markers."""
    markers, base, _ = _marker_probabilities(panel)
    boosted = []
    for cell_type in range(N_CELL_TYPES):
        _, probs, _ = _marker_probabilities(
            panel, cell_type, N_CELL_TYPES, 2.0, N_SHARED_MARKERS
        )
        boosted.append(set(markers[probs / base > (probs / base).min() * 1.5]))
    shared = set.intersection(*boosted)
    return [block - shared for block in boosted]


def planted_table(edgelist: pl.DataFrame, panel) -> pl.DataFrame:
    """Count cells per (hashing index, cell type), recovered from the markers.

    Each cell's hashing index is its most frequent hashing marker suffix and its
    cell type is the marker block it carries most umis of.
    """
    markers, _, is_hashing = _marker_probabilities(panel)
    block_of = {
        m: cell_type
        for cell_type, block in enumerate(cell_type_blocks(panel))
        for m in block
    }
    hashing = set(markers[is_hashing])

    umis = pl.concat(
        [
            edgelist.select("component", umi="umi1", marker="marker_1"),
            edgelist.select("component", umi="umi2", marker="marker_2"),
        ]
    ).filter(pl.col("component").is_not_null())
    umis = umis.unique(["component", "umi"])

    hashing_index = (
        umis.filter(pl.col("marker").is_in(hashing))
        .group_by("component")
        .agg(hashing_index=pl.col("marker").str.split("-").list.last().mode().first())
    )
    cell_type = (
        umis.with_columns(
            cell_type=pl.col("marker").replace_strict(block_of, default=None)
        )
        .drop_nulls("cell_type")
        .group_by("component")
        .agg(pl.col("cell_type").mode().first())
    )
    cells = cell_type.join(hashing_index, on="component", how="left")
    return (
        cells.group_by("hashing_index", "cell_type")
        .len()
        .sort("hashing_index", "cell_type")
    )


def main() -> None:
    """Generate every library and write its fastq files and metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    output_dir = parser.parse_args().output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for library, (
        assay_name,
        panel_name,
        n_cells,
        n_crossing_edges,
        seed,
        fastqs,
    ) in LIBRARIES.items():
        panel = pna_config.get_panel(panel_name)
        assay = pna_config.get_assay(assay_name)
        has_hashing = _hashing_mask(panel.to_polars()).any()
        params = dict(
            n_cells=n_cells,
            n_nodes=N_NODES,
            n_edges=N_EDGES,
            min_neighbors=MIN_NEIGHBORS,
            n_crossing_edges=n_crossing_edges,
            hashing_fraction=HASHING_FRACTION,
            hashing_indices=HASHING_INDICES if has_hashing else None,
            n_cell_types=N_CELL_TYPES,
            cell_type_effect=CELL_TYPE_EFFECT,
            n_shared_markers=N_SHARED_MARKERS,
        )
        edgelist = generate_edgelist(panel=panel, rng=seed, **params)
        table = planted_table(edgelist, panel)
        print(f"{library}: {edgelist.height} edges, planted cells per type:")
        print(table)
        assert table["cell_type"].n_unique() == N_CELL_TYPES
        assert table.height == N_CELL_TYPES * max(table["hashing_index"].n_unique(), 1)

        n_reads = READS_PER_EDGE * edgelist.height
        for sample_name, fastq_seed in fastqs.items():
            write_pna_fastq(
                sample_name=sample_name,
                n_reads=n_reads,
                edgelist=edgelist,
                panel=panel,
                assay=assay,
                output_dir=output_dir,
                rng=fastq_seed,
                quality_std=QUALITY_STD,
            )
            metadata = dict(
                sample_name=sample_name,
                edgelist=library,
                panel=panel_name,
                assay=assay_name,
                edgelist_seed=seed,
                fastq_seed=fastq_seed,
                n_reads=n_reads,
                quality_std=QUALITY_STD,
                n_edges_total=edgelist.height,
                **params,
            )
            (output_dir / f"{sample_name}.metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
            print(f"  wrote {sample_name}_R1/R2.fastq.gz ({n_reads} reads)")


if __name__ == "__main__":
    main()
