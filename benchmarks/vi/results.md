# ov.vi benchmark results

## Parity with scvi-tools 1.5.2

### SEA-AD 5k cells (smoke), H100 (seeds per impl: {'ov': 3, 'scvi': 3}; epochs 400)

| impl | neg_elbo_train | neg_elbo_validation | iLISI | cLISI | kBET | ASW_batch | ASW_label | NMI | ARI | graph_conn | isolated_labels | train_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ov | 1114 ± 0.73 | 1252 ± 5 | 0.1275 ± 0.00068 | 1 ± 0 | 0.9263 ± 0.0016 | 0.7721 ± 0.0037 | 0.6652 ± 0.0033 | 0.9275 ± 0.0068 | 0.8908 ± 0.0058 | 0.9745 ± 0.0045 | 0.7133 ± 0.026 | 14.68 ± 0.92 |
| scvi | 1114 ± 0.43 | 1251 ± 6 | 0.1275 ± 0.001 | 1 ± 0 | 0.9314 ± 0.003 | 0.7705 ± 0.0038 | 0.6618 ± 0.0056 | 0.9237 ± 0.0019 | 0.8851 ± 0.0042 | 0.9569 ± 0.012 | 0.6831 ± 0.031 | 85.8 ± 0.45 |

### SEA-AD 5k cells (smoke), CPU 16 threads (seeds per impl: {'scvi': 1}; epochs 400)

| impl | neg_elbo_train | neg_elbo_validation | iLISI | cLISI | kBET | ASW_batch | ASW_label | NMI | ARI | graph_conn | isolated_labels | train_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scvi | 1114 ± 0 | 1258 ± 0 | 0.1272 ± 0 | 1 ± 0 | 0.9274 ± 0 | 0.7767 ± 0 | 0.6653 ± 0 | 0.9273 ± 0 | 0.8908 ± 0 | 0.953 ± 0 | 0.6028 ± 0 | 228.7 ± 0 |

### SEA-AD 50k cells, H100 (seeds per impl: {'ov': 3, 'scvi': 3}; epochs 160)

| impl | neg_elbo_train | neg_elbo_validation | iLISI | cLISI | kBET | ASW_batch | ASW_label | NMI | ARI | graph_conn | isolated_labels | train_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ov | 1103 ± 1.2 | 1117 ± 10 | 0.1047 ± 0.0002 | 1 ± 0 | 0.7507 ± 0.0058 | 0.7626 ± 0.0051 | 0.676 ± 0.0029 | 0.9264 ± 0.0047 | 0.8892 ± 0.021 | 0.98 ± 0.0015 | 0.7938 ± 0.0071 | 52.3 ± 1 |
| scvi | 1103 ± 0.99 | 1117 ± 11 | 0.1051 ± 0.00026 | 1 ± 0 | 0.7535 ± 0.0048 | 0.762 ± 0.0079 | 0.6785 ± 0.0055 | 0.9166 ± 0.016 | 0.8402 ± 0.088 | 0.9823 ± 0.0025 | 0.8025 ± 0.013 | 337.4 ± 2.6 |

## Block shuffle

### Block shuffle vs random (SEA-AD 50k, 160 epochs, 3 seeds)

| config | neg_elbo_train | neg_elbo_validation | iLISI | cLISI | kBET | ASW_batch | ASW_label | NMI | ARI | graph_conn | isolated_labels |
|---|---|---|---|---|---|---|---|---|---|---|---|
| block order=donor block=2048 x8 | 1109 ± 0.87 | 1119 ± 11 | 0.105 ± 0.00028 | 1 ± 0 | 0.7523 ± 0.0022 | 0.7641 ± 0.0058 | 0.6809 ± 0.0089 | 0.9307 ± 0.00079 | 0.907 ± 0.0092 | 0.9824 ± 0.0016 | 0.7969 ± 0.0098 |
| block order=donor block=256 x4 | 1108 ± 1.5 | 1117 ± 9.3 | 0.1041 ± 0.0003 | 1 ± 0 | 0.7486 ± 0.0024 | 0.764 ± 0.0038 | 0.678 ± 0.0036 | 0.9321 ± 0.0039 | 0.9022 ± 0.021 | 0.9813 ± 0.0051 | 0.7885 ± 0.0075 |
| block order=donor block=512 x8 | 1112 ± 8.5 | 1121 ± 11 | 0.1055 ± 0.00041 | 1 ± 0 | 0.7579 ± 0.0059 | 0.7704 ± 0.0093 | 0.6805 ± 0.0036 | 0.9326 ± 0.012 | 0.8993 ± 0.037 | 0.9816 ± 0.0036 | 0.7836 ± 0.017 |
| block order=file block=512 x8 | 1104 ± 1.3 | 1118 ± 10 | 0.1048 ± 7.1e-05 | 1 ± 0 | 0.7551 ± 0.0066 | 0.7693 ± 0.0013 | 0.6817 ± 0.0072 | 0.9277 ± 0.0047 | 0.8863 ± 0.022 | 0.983 ± 0.0015 | 0.7875 ± 0.015 |
| random order=any block=- x- | 1103 ± 1.2 | 1117 ± 10 | 0.1047 ± 0.0002 | 1 ± 0 | 0.7507 ± 0.0058 | 0.7626 ± 0.0051 | 0.676 ± 0.0029 | 0.9264 ± 0.0047 | 0.8892 ± 0.021 | 0.98 ± 0.0015 | 0.7938 ± 0.0071 |

## Training speed

### speed_cpu.csv

| data | device | impl | epochs | mean | min | count |
|---|---|---|---|---|---|---|
| seaad300k | cpu | ov | 3 | 54.49 | 54.49 | 1 |
| seaad300k | cpu | ov_compile | 3 | 49.58 | 49.58 | 1 |
| seaad300k | cpu | scvi | 3 | 113.16 | 113.16 | 1 |
| smoke | cpu | ov | 50 | 29.23 | 29.23 | 1 |
| smoke | cpu | ov_compile | 50 | 37.67 | 37.67 | 1 |
| smoke | cpu | scvi | 50 | 34.85 | 34.85 | 1 |

### speed_cuda.csv

| data | device | impl | epochs | mean | min | count |
|---|---|---|---|---|---|---|
| seaad300k | cuda | ov | 5 | 10.37 | 10.0 | 2 |
| seaad300k | cuda | ov_final | 5 | 10.5 | 10.17 | 2 |
| seaad300k | cuda | scvi | 5 | 64.37 | 63.15 | 2 |
| seaad300k | cuda | scvi_final | 5 | 63.06 | 62.47 | 2 |
| smoke | cuda | ov | 50 | 2.34 | 1.93 | 2 |
| smoke | cuda | ov_final | 50 | 2.36 | 1.97 | 2 |
| smoke | cuda | ov_graph_probe | 20 | 2.75 | 2.75 | 1 |
| smoke | cuda | ov_probe | 20 | 3.85 | 3.85 | 1 |
| smoke | cuda | scvi | 50 | 10.92 | 10.79 | 2 |
| smoke | cuda | scvi_final | 50 | 10.82 | 10.69 | 2 |
| smoke | cuda | scvi_probe | 20 | 5.05 | 5.05 | 1 |

## Disk training

### Full SEA-AD (1,378,211 cells, 2000 HVGs) trained from disk on the H100

| mode | tag | n_obs | epochs | epoch_s | train_s | latent_s | rss_before_train_gb | rss_peak_train_gb | rss_growth_train_gb | gpu_used_peak_train_gb | torch_peak_alloc_gb | iLISI | cLISI | kBET | ASW_batch | ASW_label | NMI | ARI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dask | nan | 1378211 | 6 | [22.6, 22.4, 22.4, 22.4, 22.4, 22.4] | 138.916 | 15.891 | 2.162 | 2.666 | 0.505 | 25.397 | 6.113 | 0.1 | 1.0 | 0.741 | 0.763 | 0.686 | 0.904 | 0.753 |
| oom | nan | 1378211 | 6 | [21.2, 20.3, 19.7, 19.5, 19.7, 19.5] | 121.146 | 19.166 | 2.092 | 2.923 | 0.831 | 27.961 | 1.229 | 0.099 | 1.0 | 0.715 | 0.756 | 0.684 | 0.937 | 0.917 |
| dask | _buf3 | 1378211 | 6 | [22.2, 22.7, 21.7, 21.6, 21.7, 21.6] | 132.943 | 15.937 | 2.107 | 2.491 | 0.383 | 17.848 | 0.895 | 0.099 | 1.0 | 0.729 | 0.753 | 0.684 | 0.931 | 0.888 |
| oom | _rerun | 1378211 | 6 | [20.4, 19.2, 18.4, 18.2, 17.7, 17.6] | 112.737 | 17.005 | 2.092 | 2.933 | 0.841 | 2.436 | 1.229 | 0.099 | 1.0 | 0.714 | 0.757 | 0.684 | 0.91 | 0.756 |
| dask | _final | 1378211 | 6 | [22.0, 21.5, 21.4, 21.5, 21.4, 21.7] | 130.935 | 15.683 | 2.109 | 2.494 | 0.385 | 16.44 | 0.895 | 0.099 | 1.0 | 0.727 | 0.753 | 0.684 | 0.911 | 0.762 |

## Concurrent reads (lock test)

| mode | procs | split | rank | elements | gb_read | seconds | gb_per_s | full | error | wall_s | hdf5_locking | file | writer_open |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pread | 4 | True | 0 | 1905517726 | 22.87 | 16.1 | 1.418 | True | nan | 29.2 | default | nan | nan |
| pread | 4 | True | 1 | 1905062067 | 22.86 | 17.0 | 1.345 | True | nan | 29.2 | default | nan | nan |
| pread | 4 | True | 2 | 1906913857 | 22.88 | 14.0 | 1.636 | True | nan | 29.2 | default | nan | nan |
| pread | 4 | True | 3 | 1904256692 | 22.85 | 14.9 | 1.529 | True | nan | 29.2 | default | nan | nan |
| h5py | 4 | True | 0 | 1905517726 | 22.87 | 12.6 | 1.813 | True | nan | 13.3 | default | nan | nan |
| h5py | 4 | True | 1 | 1905062067 | 22.86 | 13.1 | 1.748 | True | nan | 13.3 | default | nan | nan |
| h5py | 4 | True | 2 | 1906913857 | 22.88 | 11.9 | 1.918 | True | nan | 13.3 | default | nan | nan |
| h5py | 4 | True | 3 | 1904256692 | 22.85 | 12.3 | 1.854 | True | nan | 13.3 | default | nan | nan |
| pread | 4 | False | 0 | 27527215 | 0.22 | 0.3 | 0.715 | True | nan | 6.6 | default | lockcopy.h5ad | True |
| pread | 4 | False | 1 | 27527215 | 0.22 | 0.4 | 0.593 | True | nan | 6.6 | default | lockcopy.h5ad | True |
| pread | 4 | False | 2 | 27527215 | 0.22 | 0.4 | 0.499 | True | nan | 6.6 | default | lockcopy.h5ad | True |
| pread | 4 | False | 3 | 27527215 | 0.22 | 0.3 | 0.782 | True | nan | 6.6 | default | lockcopy.h5ad | True |
| h5py | 4 | False | 0 | 0 | 0.0 | 0.0 | 0.0 | False | BlockingIOError: [Errno 11] Unable to synchronously open file (unable to lock file, errno = 11, error message = 'Resource temporarily unavailable') | 0.3 | default | lockcopy.h5ad | True |
| h5py | 4 | False | 1 | 0 | 0.0 | 0.0 | 0.0 | False | BlockingIOError: [Errno 11] Unable to synchronously open file (unable to lock file, errno = 11, error message = 'Resource temporarily unavailable') | 0.3 | default | lockcopy.h5ad | True |
| h5py | 4 | False | 2 | 0 | 0.0 | 0.0 | 0.0 | False | BlockingIOError: [Errno 11] Unable to synchronously open file (unable to lock file, errno = 11, error message = 'Resource temporarily unavailable') | 0.3 | default | lockcopy.h5ad | True |
| h5py | 4 | False | 3 | 0 | 0.0 | 0.0 | 0.0 | False | BlockingIOError: [Errno 11] Unable to synchronously open file (unable to lock file, errno = 11, error message = 'Resource temporarily unavailable') | 0.3 | default | lockcopy.h5ad | True |
