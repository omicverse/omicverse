The model code for scFoundation

## Inference
### How to Use:
#### Overview

Our model is designed to process gene expression data, either from single cells or in bulk, and return the corresponding cell or gene context embeddings.
  
**Required Package**:
To run the provided Python script, ensure you have the following packages installed:
```
argparse
numpy
pandas
os
scipy
pytorch
einops
anndata
omicverse
local_attention
```

#### 1. Convert Gene Symbol: (If you use the demo data, skip this step)
- Convert the gene symbol in your data to match our list `OS_scRNA_gene_index.19264.tsv`.
- For Python users, you can use the `main_gene_selection` function in `get_embedding.py`:
  ```python
  # X_df represents your single cell data with cells in rows and genes in columns
  gene_list_df = pd.read_csv('../OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
  gene_list = list(gene_list_df['gene_name'])
  X_df, to_fill_columns, var = main_gene_selection(X_df, gene_list)
  ```
- Save your data `X_df` in either `npy` or `csv` format.

#### 2. Inference:
- Please download the model weight file via https://hopebio2020-my.sharepoint.com/:f:/g/personal/dongsheng_biomap_com/Eh22AX78_AVDv6k6v4TZDikBXt33gaWXaz27U9b1SldgbA and put it into the  `models` folder
- Please download the raw gene expression example data used for inference from Figshare: https://doi.org/10.6084/m9.figshare.24049200 , and then unzip it as a folder named `examples`
- In the `demo.sh` file, we provide several scripts to infer various types of embeddings, including single cell, bulk, and gene embeddings. To run these scripts, simply copy the corresponding Python command and paste it into your command line or terminal.
- Here's an example command for inferring cell embeddings:
  ```bash
  ### Cell embedding
  python get_embedding.py --task_name Baron --input_type singlecell --output_type cell --pool_type all --tgthighres a5 --data_path ./examples/enhancement/Baron_enhancement.csv --save_path ./examples/enhancement/ --pre_normalized F --version rde
  ```

After running these scripts, you will generate embeddings for use in our downstream task analyses. To verify the consistency of these generated embeddings with those provided in the downstream task folders, refer to the `check_consistency.ipynb` file. Please note that due to differences in CUDA driver versions and hardware configurations, minor numerical differences might occur beyond the thousandth decimal place.

#### 3. Adjusting Arguments:
For a quick start, simply modify the `--data_path` to point to your data.
Below are detailed descriptions for each argument:

- **input_type**: Specifies the type of input. 
  - Choices: `singlecell`, `bulk`
  - Default: `singlecell`

- **output_type**: Determines the type of output.
  - Choices: `cell`, `gene`, `gene_batch`
  - Default: `cell`
  - Note: In `cell` mode, The output shape is (N,h), where N is the number of cells, h is the hidden dimension. In `gene*` mode, The output shape is (N,19264,h),where N is the number of cells,19264 is the gene number, and h is the hidden dimension. In `gene` mode, gene embedding of each cell is processed individually. In `gene_batch` mode, all cells in your data are treated as a single batch and processed together. Ensure the number of input cells doesn't exceed 5 in this mode.

- **pool_type**: Defines the pooling types for cell embedding.
  - Choices: `all`, `max`
  - Default: `all`
  - The method of getting cell embeddings. Applicable only when `output_type` is set to `cell`.

- **tgthighres**: Sets the value of token T. 
  - Default: `t4`
  - Note: Can be set in three ways - targeted high resolution which means T=number (starting with 't'), fold change of high resolution which means T/S=number (starting with 'f'), or addition of high resolution which means T=S+number (starting with 'a'). Only valid when `input_type` is `singlecell`.

- **pre_normalized**: Controls the computation method for the S token.
  - Choices: `F`, `T`, `A`
  - Default: `F`
  - Note: When `input_type` is `singlecell`, `T` or `F` indicates if the input gene expression data is already normalized+log1p. `A` means data is normalized+log1p with the total count appended at the end, resulting in a data shape of N*19265. This mode is used for the GEARS task. For `bulk` input type, `F` means the T and S token values are log10(sum of gene expression), while `T` means they are the sum without log transformation. This is useful for bulk data with few sequenced genes.

- **version**: Model versions for generating cell embeddings.
  - Default: `ce`
  - Note: Use `rce` for read depth enhancement and `ce` otherwise. Only valid when `output_type` is `cell`.

- **model_path**: Path to the model.
  - Default: `None`

- **ckpt_name**: Checkpoint Name.
  - Default: `01B-resolution`

## Finetune/Integrate scFoundation with other models
To finetune or integrate the scFoundation model with additional layers or models, you can refer to the example model code provided in the `finetune_model.py` file. The essential steps involve loading the scFoundation model and appending it with other layers as needed. Here's a snippet to get you started:
```
import sys 
sys.path.append("../model/") # path to this folder
from load import *
pretrainmodel,pretrainconfig = load_model_frommmf(ckpt_path,key)

self.token_emb = model.token_emb
self.pos_emb = model.pos_emb
self.encoder = model.encoder
```
If you're facing GPU memory limitations, the following code allows you to finetune only a part of the scFoundation model.
```
for na, param in self.encoder.named_parameters():
    param.requires_grad = False
for na, param in self.encoder.transformer_encoder[-2].named_parameters():
    print('self.encoder.transformer_encoder ',na,' have grad')
    param.requires_grad = True
```
Once you've defined the finetuned-model class based on scFoundation, it can be incorporated into your existing training loop code. We have updated the GEARS directory, demonstrating how the scFoundation model can be seamlessly integrated and finetuned with the GEARS model.


## Copyright Notice

### Model Weight

Model Weights are licensed under CC BY-NC-SA 4.0 License (https://creativecommons.org/licenses/by-nc-sa/4.0/).

### Third-party Software License

Use of the third-party software, libraries or code referred to in the Acknowledgements section may be governed by separate terms and conditions or license provisions.

Your use of the third-party software, libraries or code is subject to any such terms and you should check that you can comply with any applicable restrictions or terms and conditions before use.

## Acknowledgements

scFoundation inference code uses and/or references the following separate libraries and packages (ordered alphabetically):

- [einops](https://github.com/arogozhnikov/einops)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Pytorch](https://pytorch.org/)
- [Scipy](https://scipy.org/)
- [Scanpy](https://scanpy.readthedocs.io/en/stable/)
- [Tqdm](https://github.com/tqdm/tqdm)

Thanks for all their contributors and maintainers!