"""Shared hyperparameters / helpers for the MrVI ours-vs-scvi-tools validation."""
DATA = "/scratch/users/steorra/tmp/vi_p2/data/mrvi/wilk2020_prepped.h5ad"  # prep_data.py
SAMPLE_KEY = "donor_id"
# cell_type is used for downstream grouping (get_local_sample_distances groupby=...) and
# the latent scib-style bio-conservation metrics, but NOT fed to MRVI as labels_key: with
# labels_key set and n_labels>1, MrVI overrides u_prior_mixture_k to n_labels and adds a
# +10-logit per-label offset to the mixture categorical, turning the "mixture" prior into
# 13 independent per-label Gaussian-matching problems whose KL term is then very sensitive
# to each implementation's (different) random initialisation / optimisation trajectory --
# seen directly here (reconstruction_loss converges to ~203-204 either way, but kl_local
# converged to ~23 for ours vs ~6500 for the reference on this seed). Leaving labels_key
# unset keeps MODEL_KW['u_prior_mixture_k'] as a genuine free/unsupervised mixture, which
# is also scvi-tools' own common usage when no cell-type column is registered.
LABELS_KEY = "cell_type"
COV_KEY = "disease"

MODEL_KW = dict(n_latent=20, n_latent_u=8, encoder_n_hidden=128, encoder_n_layers=2,
               u_prior_mixture_k=12)
# cuda_graph=False (ours only, see common.OURS_TRAIN_KW): CUDA graph capture of MRVI's
# training step currently fails ("Offset increment outside graph capture") -- ov.vi's
# train() enables it by default for any CUDA, single-process run (not gated by a step
# count like torch.compile's own "auto"); not investigated further here, see report.
TRAIN_KW = dict(max_epochs=120, batch_size=256,
                plan_kwargs=dict(lr=2e-3, eps=1e-8, weight_decay=1e-8, n_epochs_kl_warmup=20))
OURS_TRAIN_KW = dict(TRAIN_KW, cuda_graph=False, check_val_every_n_epoch=1)
