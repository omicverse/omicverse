"""Decisive test: record scvi.external.Decipher's exact per-epoch minibatch
cell-index order (train RandomSampler) and train/val split for one seed, then
drive ov.vi.Decipher's module manually (same init, bypassing fit()) on that
exact same per-epoch order, and compare the resulting trajectory recovery.

If replaying reference's data order fixes a "bad" ov.vi seed, the shuffle
stream (not a numerical bug -- already ruled out by replay_step0.py-style
single-step checks) explains the gap.
"""
import sys
import numpy as np
import torch
import torch.utils.data as tud
import anndata as ad
from scipy.stats import spearmanr

sys.path.insert(0, __file__.rsplit("/", 1)[0])

DATA = "/scratch/users/steorra/tmp/vi_p2/data/decipher/erythroid_neurips2021.h5ad"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MAX_EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 150

a = ad.read_h5ad(DATA)
pt = a.obs["GEX_pseudotime_order"].values.astype(np.float64)
ptm = ~np.isnan(pt)
Xs = a.layers["counts"]
X = Xs.toarray() if hasattr(Xs, "toarray") else np.asarray(Xs)

# ---------------------------------------------------------------------------
# 1. Train the reference model, recording every epoch's RandomSampler order
#    (local indices into its train Subset) and the final train/val split.
# ---------------------------------------------------------------------------
import scvi  # noqa: E402

captured = []
orig_iter = tud.RandomSampler.__iter__


def patched_iter(self):
    order = list(orig_iter(self))
    captured.append(order)
    return iter(order)


tud.RandomSampler.__iter__ = patched_iter
try:
    scvi.settings.seed = SEED
    torch.manual_seed(SEED)
    aa = a.copy()
    scvi.external.Decipher.setup_anndata(aa, layer="counts")
    ref = scvi.external.Decipher(aa, dim_v=2, dim_z=10, beta=0.1)
    ref.train(max_epochs=MAX_EPOCHS, batch_size=128, accelerator="cpu",
              enable_progress_bar=False, early_stopping=True, check_val_every_n_epoch=5,
              early_stopping_patience=10)
finally:
    tud.RandomSampler.__iter__ = orig_iter

ref.module.eval()
v_ref = ref.get_latent_representation()
rho_ref = [spearmanr(v_ref[ptm, k], pt[ptm]).statistic for k in range(2)]
train_idx = np.asarray(ref.train_indices)
val_idx = np.asarray(ref.validation_indices)
n_epochs_done = len(captured)
print(f"ref: epochs_done={n_epochs_done} best_rho={max(abs(rho_ref[0]), abs(rho_ref[1])):.3f} "
     f"(ax0={rho_ref[0]:.3f} ax1={rho_ref[1]:.3f})")

# drop_last=True, batch_size=128: chunk each captured epoch's local-index
# permutation into full batches only, then map local -> global cell index.
BATCH = 128
epoch_batches = []
for order in captured:
    order = np.asarray(order)
    n_full = (len(order) // BATCH) * BATCH
    order = order[:n_full]
    glob = train_idx[order]
    epoch_batches.append([glob[i:i + BATCH] for i in range(0, len(glob), BATCH)])
print(f"recorded {len(epoch_batches)} epochs, "
     f"{sum(len(e) for e in epoch_batches)} total minibatches")

# ---------------------------------------------------------------------------
# 2. Drive ov.vi.Decipher's module manually through the exact same per-epoch
#    order (same init: same torch.manual_seed as the reference run above).
# ---------------------------------------------------------------------------
import omicverse as ov  # noqa: E402
from omicverse.vi._train import ClippedAdam  # noqa: E402
from omicverse.vi._decipher import DecipherModule  # noqa: E402

torch.manual_seed(SEED)
mod = DecipherModule(n_input=X.shape[1], dim_v=2, dim_z=10, beta=0.1)
opt = ClippedAdam(mod.parameters(), lr=5e-3, eps=1e-8, weight_decay=1e-4, clip_norm=10.0)
mod.freeze_bn_from = 5  # matches Decipher.train()'s freeze_batchnorm='auto' for
mod._bn_frozen = False  # check_val_every_n_epoch=5 (what the reference run used)

torch.manual_seed(SEED + 1)  # matches BaseVIModel.train()'s own convention
for epoch, batches in enumerate(epoch_batches):
    mod.train()
    mod.on_train_epoch_start(epoch)
    for glob_idx in batches:
        xb = torch.as_tensor(X[glob_idx], dtype=torch.float32)
        out = mod({"x": xb})
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        opt.step()

mod.eval()
with torch.no_grad():
    xl, z_loc, _ = mod.encode(torch.as_tensor(X, dtype=torch.float32))
    v_replay = mod.encode_v(z_loc, xl)[0].numpy()
rho_replay = [spearmanr(v_replay[ptm, k], pt[ptm]).statistic for k in range(2)]
print(f"ours (replayed reference order+init, {len(epoch_batches)} epochs): "
     f"best_rho={max(abs(rho_replay[0]), abs(rho_replay[1])):.3f} "
     f"(ax0={rho_replay[0]:.3f} ax1={rho_replay[1]:.3f})")

# ---------------------------------------------------------------------------
# 3. For comparison: ours with its OWN default shuffle, same seed/epochs.
# ---------------------------------------------------------------------------
torch.manual_seed(SEED)
own = ov.vi.Decipher(a, layer="counts", seed=SEED, dim_v=2, dim_z=10, beta=0.1)
own.train(max_epochs=n_epochs_done, batch_size=128, device="cpu", progress=False, seed=SEED,
         early_stopping=False, check_val_every_n_epoch=None, freeze_batchnorm=5)
v_own = own.get_latent_representation()
rho_own = [spearmanr(v_own[ptm, k], pt[ptm]).statistic for k in range(2)]
print(f"ours (own default shuffle, same {n_epochs_done} epochs): "
     f"best_rho={max(abs(rho_own[0]), abs(rho_own[1])):.3f} "
     f"(ax0={rho_own[0]:.3f} ax1={rho_own[1]:.3f})")
