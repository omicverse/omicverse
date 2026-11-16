"""``HeteroGraph``: typed entities + typed relations + a single int64 edge
table, built from (in-memory or disk-backed) AnnData matrices.

This is the data layer behind ``ov.ge`` (a from-scratch re-implementation of
the graph PyTorch-BigGraph / SIMBA train on): every entity type (cells,
genes, peaks, motifs, k-mers, ...) gets an ordered id table and a *local*
integer index (0..n-1); every relation references a (lhs type, rhs type,
weight, operator) tuple; every edge is ``(src_local, dst_local, rel_id)``.

Builders accept plain ``scipy.sparse``/``numpy`` matrices as well as
disk-backed matrices (AnnDataOOM's own ``_read_rows`` reader, or a dask
array) via :func:`_iter_row_blocks`, so the resulting graph is identical
whether built from an in-memory ``AnnData`` or an ``AnnDataOOM`` / Dask
``AnnData`` pointing at the same data -- the data is streamed in row blocks
rather than ever densified or loaded whole into host memory at once.

No ``torch`` import here: the graph is pure numpy/scipy/pandas, so it can be
built (and even persisted) without ever touching a GPU/CPU tensor library.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp


def _iter_row_blocks(X, block_size: int = 4096):
    """Yield ``(row_offset, csr_block)`` pairs covering ``X`` top to bottom.

    Supports, without ever materialising the whole matrix in host memory:

    * plain ``numpy``/``scipy.sparse`` matrices (in-memory ``AnnData``);
    * anndataoom-fast readers exposing ``_read_rows(start, end)`` (the same
      primitive ``ov.vi._data.OOMSource`` uses for disk-backed ``AnnData``);
    * dask arrays (row chunks via ``.blocks``).
    """
    n = X.shape[0]
    if block_size <= 0:
        block_size = n or 1
    if hasattr(X, "_read_rows"):
        for s in range(0, n, block_size):
            e = min(s + block_size, n)
            yield s, sp.csr_matrix(X._read_rows(s, e))
        return
    if hasattr(X, "blocks") and hasattr(X, "compute") and hasattr(X, "numblocks"):
        s = 0
        for blk in X.blocks:
            arr = blk.compute()
            arr = arr if sp.issparse(arr) else sp.csr_matrix(np.asarray(arr))
            yield s, arr.tocsr()
            s += arr.shape[0]
        return
    Xc = X.tocsr() if sp.issparse(X) else sp.csr_matrix(np.asarray(X))
    for s in range(0, n, block_size):
        e = min(s + block_size, n)
        yield s, Xc[s:e]


def _block_rowcoldata(block):
    """``(row, col, data)`` of a CSR block's explicit non-zeros, aligned
    1:1 (unlike ``.nonzero()`` + separate indexing into ``.data``)."""
    block = block.tocsr()
    block.eliminate_zeros()
    coo = block.tocoo()
    return coo.row, coo.col, coo.data


class HeteroGraph:
    """A typed multigraph for knowledge-graph embedding.

    Attributes
    ----------
    entity_ids : dict[str, pandas.Index]
        Ordered, de-duplicated ids per entity type; position in the index is
        the *local* integer used everywhere else (embedding row, edge
        endpoint).
    relations : list[dict]
        ``{"name", "lhs", "rhs", "weight", "operator"}`` per relation, in
        the order relations were added (``rel_id`` = position in this list).
    edges_src, edges_dst, edges_rel : numpy.ndarray[int64]
        Parallel arrays, one row per edge, available after :meth:`build`.
    """

    def __init__(self):
        self.entity_ids: dict = {}
        self.relations: list = []
        self._edges_src: list = []
        self._edges_dst: list = []
        self._edges_rel: list = []
        self._built = False
        self.edges_src = self.edges_dst = self.edges_rel = None

    # ------------------------------------------------------------------
    # entities
    # ------------------------------------------------------------------
    def add_entity_type(self, name: str, ids) -> np.ndarray:
        """Register (or extend) entity type ``name``; return the local
        index of every element of ``ids`` (order preserved, duplicates
        across calls resolve to the same index)."""
        ids = pd.Index(ids)
        if name not in self.entity_ids:
            uniq = ids.unique()
            self.entity_ids[name] = uniq
            return uniq.get_indexer(ids)
        existing = self.entity_ids[name]
        new_ids = ids.unique()
        extra = new_ids[~new_ids.isin(existing)]
        if len(extra):
            self.entity_ids[name] = existing.append(extra)
        return self.entity_ids[name].get_indexer(ids)

    def n_entities(self, name: str) -> int:
        return len(self.entity_ids.get(name, []))

    def entity_names(self) -> list:
        return list(self.entity_ids.keys())

    # ------------------------------------------------------------------
    # relations / raw edges
    # ------------------------------------------------------------------
    def add_relation(self, name, lhs, rhs, weight: float = 1.0, operator: str = "none") -> int:
        rel_id = len(self.relations)
        self.relations.append(dict(name=name, lhs=lhs, rhs=rhs,
                                    weight=float(weight), operator=operator))
        return rel_id

    def add_edges(self, lhs_type, rhs_type, src_local, dst_local, *, name=None,
                  weight: float = 1.0, operator: str = "none", rel_id=None) -> int:
        """Append ``len(src_local)`` edges of a (possibly new) relation.
        ``src_local``/``dst_local`` are already *local* indices into the
        ``lhs_type``/``rhs_type`` entity tables."""
        src_local = np.asarray(src_local, dtype=np.int64)
        dst_local = np.asarray(dst_local, dtype=np.int64)
        if len(src_local) != len(dst_local):
            raise ValueError("src_local and dst_local must have the same length")
        if rel_id is None:
            rel_id = self.add_relation(name or f"r{len(self.relations)}",
                                        lhs_type, rhs_type, weight, operator)
        self._edges_src.append(src_local)
        self._edges_dst.append(dst_local)
        self._edges_rel.append(np.full(len(src_local), rel_id, dtype=np.int64))
        self._built = False
        return rel_id

    # ------------------------------------------------------------------
    # builders from (possibly disk-backed) matrices
    # ------------------------------------------------------------------
    def add_edges_from_sparse(self, lhs_type, rhs_type, X, row_ids, col_ids, *,
                              name=None, weight: float = 1.0, operator: str = "none",
                              rel_id=None, block_size: int = 4096) -> int:
        """One relation's edges from the explicit non-zeros of ``X`` (rows ->
        ``lhs_type``, cols -> ``rhs_type``) -- the vectorised equivalent of
        SIMBA's ``gen_graph`` (``adata.X.nonzero()`` -> source/destination
        aliases). ``X`` may be a plain matrix or any disk-backed matrix
        supported by :func:`_iter_row_blocks`."""
        row_ids = pd.Index(row_ids)
        col_ids = pd.Index(col_ids)
        self.add_entity_type(lhs_type, row_ids)
        self.add_entity_type(rhs_type, col_ids)
        rhs_full_idx = self.entity_ids[rhs_type].get_indexer(col_ids)
        lhs_table = self.entity_ids[lhs_type]
        src_parts, dst_parts = [], []
        for row0, block in _iter_row_blocks(X, block_size=block_size):
            r, c, _ = _block_rowcoldata(block)
            if len(r) == 0:
                continue
            src_parts.append(lhs_table.get_indexer(row_ids[row0 + r]))
            dst_parts.append(rhs_full_idx[c])
        src = np.concatenate(src_parts) if src_parts else np.zeros(0, dtype=np.int64)
        dst = np.concatenate(dst_parts) if dst_parts else np.zeros(0, dtype=np.int64)
        return self.add_edges(lhs_type, rhs_type, src, dst, name=name, weight=weight,
                              operator=operator, rel_id=rel_id)

    def add_discretized_relations(self, lhs_type, rhs_type, disc, row_ids, col_ids, *,
                                  weight_min: float = 1.0, weight_max: float = 5.0,
                                  name_prefix: str = "r", block_size: int = 4096) -> list:
        """One relation *per discretization level* of ``disc`` (a sparse
        matrix whose non-zero values are small positive integer bin labels,
        e.g. from :func:`discretize`), with weights linearly spaced from
        ``weight_min`` to ``weight_max`` across levels -- SIMBA's cell/gene
        expression-level relation recipe (``gen_graph``'s ``list_CG``
        branch): each expression bin becomes its own relation so stronger
        expression pulls the cell and gene embeddings closer together."""
        row_ids = pd.Index(row_ids)
        col_ids = pd.Index(col_ids)
        self.add_entity_type(lhs_type, row_ids)
        self.add_entity_type(rhs_type, col_ids)
        rhs_full_idx = self.entity_ids[rhs_type].get_indexer(col_ids)
        lhs_table = self.entity_ids[lhs_type]

        level_src: dict = {}
        level_dst: dict = {}
        for row0, block in _iter_row_blocks(disc, block_size=block_size):
            r, c, d = _block_rowcoldata(block)
            if len(d) == 0:
                continue
            local_src_all = lhs_table.get_indexer(row_ids[row0 + r])
            local_dst_all = rhs_full_idx[c]
            for lvl in np.unique(d):
                m = d == lvl
                level_src.setdefault(float(lvl), []).append(local_src_all[m])
                level_dst.setdefault(float(lvl), []).append(local_dst_all[m])

        levels = np.array(sorted(level_src.keys()))
        if len(levels) == 0:
            return []
        weights = np.linspace(weight_min, weight_max, num=len(levels))
        rel_ids = []
        for lvl, w in zip(levels, weights):
            src = np.concatenate(level_src[lvl])
            dst = np.concatenate(level_dst[lvl])
            rel_id = self.add_edges(lhs_type, rhs_type, src, dst,
                                    name=f"{name_prefix}{len(self.relations)}",
                                    weight=round(float(w), 2), operator="none")
            rel_ids.append(rel_id)
        return rel_ids

    # ------------------------------------------------------------------
    # finalise / stats
    # ------------------------------------------------------------------
    def build(self) -> "HeteroGraph":
        if self._built:
            return self
        if self._edges_src:
            self.edges_src = np.concatenate(self._edges_src)
            self.edges_dst = np.concatenate(self._edges_dst)
            self.edges_rel = np.concatenate(self._edges_rel)
        else:
            self.edges_src = np.zeros(0, dtype=np.int64)
            self.edges_dst = np.zeros(0, dtype=np.int64)
            self.edges_rel = np.zeros(0, dtype=np.int64)
        self._built = True
        return self

    @property
    def n_edges(self) -> int:
        self.build()
        return len(self.edges_src)

    def relation_weights(self) -> np.ndarray:
        return np.array([r["weight"] for r in self.relations], dtype=np.float64)

    def edges_per_relation(self) -> dict:
        self.build()
        rel_names = [r["name"] for r in self.relations]
        counts = np.bincount(self.edges_rel, minlength=len(self.relations))
        return {rel_names[i]: int(counts[i]) for i in range(len(self.relations))}

    def auto_wd(self) -> float:
        """SIMBA's empirical weight-decay rule (``pbg_train(auto_wd=True)``):
        a sample-size-scaled weight decay anchored at two reference points
        from SIMBA's own simulation experiments."""
        n = self.n_edges
        if n <= 0:
            return 0.0
        if n < 5e7:
            wd = 0.013 * 2725781 / n
        else:
            wd = 0.0004 * 59103481 / n
        return float(np.around(wd, decimals=6))

    def split(self, eval_fraction: float = 0.05, seed: int = 0, method: str = "random"):
        """Train / held-out-eval edge indices.

        ``method="random"`` (default): a disjoint permutation split, reseeded
        per ``seed`` -- simpler and seed-independent-of-each-other, a
        deliberate determinism improvement over PBG's own scheme.

        ``method="pbg"``: PyTorch-BigGraph's *exact* holdout derivation
        (``torchbiggraph/train_cpu.py``) -- a single, **training-seed-
        independent** sample fixed by ``hash(bucket)`` (default
        ``(edge_path_idx, edge_chunk_idx, bucket.lhs, bucket.rhs) = (0,0,0,0)``,
        the only case for this package's always-single-partition entities),
        drawn **with replacement** over edge *positions* in construction
        order and excluded from training for the whole run (not resampled
        per epoch). ``seed`` is ignored for this method -- use it to
        reproduce a PBG run's held-out set exactly (see
        ``benchmarks/ge/simba/verify_eval_formula.py``), or to compare
        multiple training seeds against one fixed, shared holdout the way
        PBG's own 3-seed runs do.
        """
        self.build()
        n = self.n_edges
        n_eval = int(round(n * eval_fraction))
        if method == "random":
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n)
            return perm[n_eval:], perm[:n_eval]
        if method == "pbg":
            import torch

            g = torch.Generator()
            g.manual_seed(hash((0, 0, 0, 0)))
            eval_idx = torch.randint(n, (n_eval,), dtype=torch.long, generator=g)
            n_train = n - n_eval
            train_idx = torch.arange(n)
            train_idx[eval_idx] = torch.arange(n_train, n)
            train_idx = train_idx[:n_train]
            return train_idx.numpy(), eval_idx.numpy()
        raise ValueError(f"unknown split method {method!r}; choose 'random' or 'pbg'")

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.build()
        np.savez(os.path.join(path, "edges.npz"),
                 src=self.edges_src, dst=self.edges_dst, rel=self.edges_rel)
        meta = {
            "relations": self.relations,
            "entity_types": {k: [str(x) for x in v] for k, v in self.entity_ids.items()},
        }
        with open(os.path.join(path, "graph.json"), "w") as f:
            json.dump(meta, f)
        return path

    @classmethod
    def load(cls, path: str) -> "HeteroGraph":
        g = cls()
        with open(os.path.join(path, "graph.json")) as f:
            meta = json.load(f)
        g.relations = meta["relations"]
        for k, v in meta["entity_types"].items():
            g.entity_ids[k] = pd.Index(v)
        arr = np.load(os.path.join(path, "edges.npz"))
        g._edges_src = [arr["src"]]
        g._edges_dst = [arr["dst"]]
        g._edges_rel = [arr["rel"]]
        g.build()
        return g

    def __repr__(self):
        self.build()
        ents = ", ".join(f"{k}={v}" for k, v in
                         sorted(((k, self.n_entities(k)) for k in self.entity_ids)))
        return (f"HeteroGraph(entities=[{ents}], relations={len(self.relations)}, "
                f"edges={self.n_edges})")


__all__ = ["HeteroGraph"]
