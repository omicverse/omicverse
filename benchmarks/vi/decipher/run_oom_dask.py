"""Confirm ov.vi.Decipher trains identically on in-memory AnnData, AnnDataOOM
(disk) and Dask-backed AnnData, on the real erythroid-lineage validation set
(not just the synthetic tests/vi/test_decipher.py fixture)."""
import numpy as np
import omicverse as ov

DATA = "/scratch/users/steorra/tmp/vi_p2/data/decipher/erythroid_neurips2021.h5ad"
KW = dict(max_epochs=15, batch_size=128, device="cpu", progress=False, seed=0,
          shuffle="block", block_size=256, buffer_blocks=4)


def _train(adata):
    m = ov.vi.Decipher(adata, layer=None, dim_v=2, dim_z=10, beta=0.1, seed=0)
    m.train(**KW)
    return m


mem = ov.read(DATA)
z_mem = _train(mem).get_latent_representation()
print("in-memory ok:", z_mem.shape, np.isfinite(z_mem).all())

ov.settings.cpu_init()
oom = ov.read(DATA, storage="disk")
m_oom = _train(oom)
z_oom = m_oom.get_latent_representation()
print("oom kind:", m_oom._source().kind,
     "matches in-memory:", np.allclose(z_mem, z_oom, atol=1e-5))

from omicverse.pp._dask import read_h5ad
da_ad = read_h5ad(DATA, chunk_size=256, device="cpu")
m_da = _train(da_ad)
z_da = m_da.get_latent_representation()
print("dask kind:", m_da._source().kind,
     "matches in-memory:", np.allclose(z_mem, z_da, atol=1e-5))
