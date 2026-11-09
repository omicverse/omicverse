from __future__ import annotations

from scanpy import read_10x_h5
from scanpy import logging as logg

import json
from pathlib import Path, PurePath
from typing import BinaryIO, Literal
import pandas as pd
from matplotlib.image import imread
from typing import Tuple, Optional, Literal, Dict
import numpy as np
from scipy import sparse



import math

#actual bin2cell dependencies start here
#the ones above are for read_visium()
#from stardist.plot import render_label
from copy import deepcopy


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import scipy.spatial
import scipy.sparse
import scipy.stats
import anndata as ad
import scanpy as sc
import numpy as np
import os

from PIL import Image
#setting needed so PIL can load the large TIFFs
#Image.MAX_IMAGE_PIXELS = None

#setting needed so cv2 can load the large TIFFs
#os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)


#NOTE ON DIMENSIONS WITHIN ANNDATA AND BEYOND
#.obs["array_row"] matches .obsm["spatial"][:,1] matches np.array image [:,0]
#.obs["array_col"] matches .obsm["spatial"][:,0] matches np.array image [:,1]
#array coords start relative to some magical point on the grid (seen bottom left/top right)
#and can require flipping the array row or column to match the image orientation
#row/col do seem to consistenly refer to the stated np.array image dimensions though
#spatial starts in top left corner of image, matching what np.array image is doing
#of note, cv2 treats [:,1] as dim 0 and [:,0] as dim 1, despite working on np.arrays
#also cv2 works with channels in a BGR order, while everything else is RGB



def render_label(labels, img=None):
    """替代 stardist.plot.render_label"""
    from skimage.color import label2rgb
    return label2rgb(labels, image=img, bg_label=0, alpha=0.3)


import numpy as np

def load_image(image_path, gray=False, dtype=np.uint8, backend="opencv"):
    """
    Efficiently load an external image and return it as an RGB numpy array.
    
    Parameters
    ----------
    image_path : str
        Path to image to be loaded.
    gray : bool, optional (default: False)
        Whether to return grayscale image.
    dtype : numpy.dtype, optional (default: np.uint8)
        Data type of the numpy array output.
    backend : {"opencv", "pil", "tifffile"}, optional (default: "opencv")
        Which library to use for reading. "opencv" is fastest, but may fail on huge images.
    
    Returns
    -------
    np.ndarray
        The loaded image.
    """
    img = None

    if backend == "opencv":
        import cv2
        #解除 OpenCV 的最大像素限制
        cv2.setConfigOption("OPENCV_IO_MAX_IMAGE_PIXELS", str(2**63 - 1))
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"OpenCV failed to load image: {image_path}")
        # 默认 BGR → RGB
        if len(img.shape) == 3 and img.shape[2] >= 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    elif backend == "pil":
        from PIL import Image
        img = Image.open(image_path)
        img = img.convert("L" if gray else "RGB")
        img = np.array(img, dtype=dtype)

    elif backend == "tifffile":
        import tifffile
        img = tifffile.imread(image_path)
        if gray and img.ndim == 3:
            img = img.mean(axis=-1).astype(dtype)

    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return img.astype(dtype, copy=False)


def normalize(img):
    '''
    Extremely naive reimplementation of default ``cbsdeep.utils.normalize()`` 
    percentile normalisation, with a lower RAM footprint than the original.
    
    Input
    -----
    img : ``numpy.array``
        Numpy array image to normalise
    '''
    eps = 1e-20
    mi = np.percentile(img,3)
    ma = np.percentile(img,99.8)
    return ((img - mi) / ( ma - mi + eps ))


def cellseg(
    image_path: str,
    labels_npz_path: str,
    stardist_model: str = "2D_versatile_he",
    block_size: int = 2048,
    min_overlap: int = 256,
    context: int = 64,
    gpu: bool = False,
    diameter: Optional[float] = None,
    mask_threshold: Optional[float] = None,   # maps from prob_thresh if provided
    flow_threshold: Optional[float] = None,
    iou_merge_threshold: float = 0.5,
    build_sparse_directly: bool = True,
    show_progress: bool = True,
    progress_desc: Optional[str] = None,
    backend: str = "cellpose",
    bbox_threshold: float = 0.4,
    **kwargs,
):
    """
    Tiled cell segmentation of a bin2cell image with any ``ov.space.seg`` backend.
    Writes a SciPy sparse CSR label matrix to labels_npz_path.

    Tiles are stitched by centroid ownership (see ``ov.space.seg.segment_tiled``),
    so a cell crossing a tile seam keeps one label.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    labels_npz_path : str
        Output path for the sparse label matrix (.npz).
    stardist_model : str
        Image hint: '2D_versatile_he' (H&E) or '2D_versatile_fluo' (expression
        or fluorescence image). It is the model for ``backend='stardist'``;
        for Cellpose < 4 it selects 'cyto' / 'nuclei'.
    block_size, min_overlap, context : int
        Tile size; the margin segmented around each tile is
        ``max(context, min_overlap // 2)``.
    gpu : bool
        Whether to use GPU.
    backend : str
        'cellpose' (default), 'stardist', 'watershed', 'sam', 'sam2',
        'micro_sam' or 'cellsam'.
    bbox_threshold : float
        CellSAM bounding box confidence threshold. Default 0.4.
    show_progress : bool
        Display progress bar.
    **kwargs
        Additional arguments forwarded to the backend
        (see ``ov.space.seg.segment``); ``iou_merge_threshold`` and
        ``build_sparse_directly`` are accepted for compatibility and unused.
    """
    if backend == "cellsam":
        return _cellseg_cellsam(
            image_path, labels_npz_path,
            block_size=block_size, min_overlap=min_overlap, context=context,
            gpu=gpu, iou_merge_threshold=iou_merge_threshold,
            show_progress=show_progress, progress_desc=progress_desc,
            bbox_threshold=bbox_threshold, **kwargs,
        )

    from ...space.seg import segment

    # an H&E hint on a single-channel image still segments, as before
    image_type = "fluorescence" if stardist_model == "2D_versatile_fluo" else "auto"
    method_kwargs = dict(kwargs)
    if backend == "cellpose":
        # StarDist-style threshold names used by the Visium HD wrappers
        prob = method_kwargs.pop("prob_thresh", None)
        method_kwargs.pop("nms_thresh", None)
        if mask_threshold is None:
            mask_threshold = prob
        if mask_threshold is not None:
            method_kwargs["cellprob_threshold"] = mask_threshold
        if flow_threshold is not None:
            method_kwargs["flow_threshold"] = flow_threshold
        if "model" not in method_kwargs:
            from importlib.metadata import version
            try:
                major = int(version("cellpose").split(".")[0])
            except Exception:
                major = 4
            if major < 4:  # keep the models this path always used before Cellpose-SAM
                method_kwargs["model"] = "nuclei" if image_type == "fluorescence" else "cyto"
    elif backend == "stardist":
        method_kwargs.setdefault("model", stardist_model)

    labels_csr = segment(
        image_path, method=backend, image_type=image_type, diameter=diameter, gpu=gpu,
        tile_size=block_size, context=max(context, min_overlap // 2), sparse=True,
        show_progress=show_progress, **method_kwargs,
    )
    sparse.save_npz(labels_npz_path, labels_csr)


def _cellseg_cellsam(
    image_path: str,
    labels_npz_path: str,
    block_size: int = 1024,
    min_overlap: int = 128,
    context: int = 64,
    gpu: bool = False,
    iou_merge_threshold: float = 0.5,
    show_progress: bool = True,
    progress_desc: Optional[str] = None,
    bbox_threshold: float = 0.4,
    cellsam_model: str = "cellsam_general",
    **kwargs,
):
    """CellSAM backend for tiled cell segmentation.

    Uses the CellSAM foundation model (vanvalenlab) with non-overlapping
    tiling to keep memory usage low. Each tile is segmented independently
    and labels are assigned sequentially.

    Requires ``pip install git+https://github.com/vanvalenlab/cellSAM.git``
    and the ``DEEPCELL_ACCESS_TOKEN`` environment variable for model download.
    """
    try:
        from cellSAM import get_model, segment_cellular_image
    except ImportError:
        raise ImportError(
            "CellSAM is not installed. Install with:\n"
            "  pip install git+https://github.com/vanvalenlab/cellSAM.git\n"
            "Also set DEEPCELL_ACCESS_TOKEN environment variable."
        )

    device = "cuda" if gpu else "cpu"

    # Read image dimensions without loading full array
    import tifffile
    with tifffile.TiffFile(image_path) as tif:
        page = tif.pages[0]
        H, W = page.shape[:2]
        n_channels = page.shape[2] if page.ndim > 2 else 1

    # Load CellSAM model
    print(f"Loading CellSAM model ({cellsam_model})...", flush=True)
    model = get_model(model=cellsam_model)
    print(f"CellSAM model loaded. Image: {H}x{W}", flush=True)

    # Non-overlapping tiling. Read tiles lazily to keep memory bounded.
    import tempfile, os as _os

    stride = block_size
    y_starts = list(range(0, H, stride))
    x_starts = list(range(0, W, stride))
    total_tiles = len(y_starts) * len(x_starts)

    # Write partial results to temp files every N tiles to bound memory
    flush_every = max(1, min(50, total_tiles // 10))
    tmp_dir = tempfile.mkdtemp(prefix="cellsam_")
    part_files = []
    buf_rows, buf_cols, buf_vals = [], [], []
    next_label = 1
    buf_pixels = 0

    pbar = None
    tiles_done = 0
    if show_progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=total_tiles, desc=progress_desc or "bin2cell.cellsam", unit="tile")
        except Exception:
            pass

    def _flush_buf():
        nonlocal buf_rows, buf_cols, buf_vals, buf_pixels
        if not buf_rows:
            return
        r = np.concatenate(buf_rows)
        c = np.concatenate(buf_cols)
        v = np.concatenate(buf_vals)
        part_path = _os.path.join(tmp_dir, f"part_{len(part_files)}.npz")
        np.savez_compressed(part_path, r=r, c=c, v=v)
        part_files.append(part_path)
        buf_rows.clear(); buf_cols.clear(); buf_vals.clear()
        buf_pixels = 0

    # Lazy tile reader via zarr (avoids loading full image into memory)
    _tif = tifffile.TiffFile(image_path)
    _store = _tif.pages[0].aszarr()
    import zarr as _zarr
    _img_z = _zarr.open(_store, mode='r')

    for ys in y_starts:
        for xs in x_starts:
            y1 = min(ys + block_size, H)
            x1 = min(xs + block_size, W)
            if _img_z.ndim == 2:
                tile_gray = np.array(_img_z[ys:y1, xs:x1])
                tile = np.stack([tile_gray] * 3, axis=-1)
            elif _img_z.shape[-1] >= 3:
                tile = np.array(_img_z[ys:y1, xs:x1, :3])
            else:
                tile = np.array(_img_z[ys:y1, xs:x1])

            try:
                result = segment_cellular_image(
                    tile, model, device=device,
                    bbox_threshold=bbox_threshold,
                    **{k: v for k, v in kwargs.items()
                       if k in ('normalize', 'postprocess', 'remove_boundaries', 'fast')},
                )
                masks = result[0] if result[0] is not None else None
            except Exception:
                masks = None

            if masks is not None and masks.max() > 0:
                # Vectorised relabel: shift all labels at once
                n_cells = masks.max()
                nz = masks > 0
                ry, rx = np.where(nz)
                shifted = masks[nz] + (next_label - 1)
                buf_rows.append((ry + ys).astype(np.int32))
                buf_cols.append((rx + xs).astype(np.int32))
                buf_vals.append(shifted.astype(np.int32))
                buf_pixels += len(ry)
                next_label += n_cells

            tiles_done += 1
            if pbar is not None:
                pbar.update(1)
            elif show_progress and tiles_done % max(1, total_tiles // 20) == 0:
                print(f"[bin2cell.cellsam] {tiles_done}/{total_tiles} tiles", flush=True)

            # Flush to disk when buffer gets large (~50M pixels)
            if buf_pixels > 50_000_000:
                _flush_buf()

    if pbar is not None:
        pbar.close()

    # Final flush
    _flush_buf()

    # Merge all parts into a single CSR matrix
    all_r, all_c, all_v = [], [], []
    for pf in part_files:
        d = np.load(pf)
        all_r.append(d['r']); all_c.append(d['c']); all_v.append(d['v'])
        _os.remove(pf)
    _os.rmdir(tmp_dir)

    if all_r:
        rows = np.concatenate(all_r)
        cols = np.concatenate(all_c)
        vals = np.concatenate(all_v)
        labels_csr = sparse.csr_matrix((vals, (rows, cols)), shape=(H, W), dtype=np.int32)
    else:
        labels_csr = sparse.csr_matrix((H, W), dtype=np.int32)

    _tif.close()
    print(f"CellSAM segmentation complete: {next_label - 1} cells", flush=True)
    sparse.save_npz(labels_npz_path, labels_csr)


def view_cellseg_labels(image_path, labels_npz_path, crop, **kwargs):
    '''
    Use StarDist's label rendering to view segmentation results in a crop 
    of the input image.
    
    Obsoleted by ``b2c.view_labels()``.
    
    Input
    -----
    image_path : ``filepath``
        Path to image that was segmented.
    labels_npz_path : ``filepath``
        Path to sparse labels generated by ``b2c.stardist()``.
    crop : tuple of ``int``
        A PIL-formatted crop specification - a four integer tuple, 
        provided as (left, upper, right, lower) coordinates.
    kwargs
        Any additional arguments to pass to StarDist's ``render_labels()``. 
        Practically most likely to be ``normalize_img``.
    '''
    #PIL is better at handling crops memory efficiently than cv2
    import cv2
    # allow very large images without triggering decompression bomb errors
    try:
        from PIL import Image as _PILImage
        _PILImage.MAX_IMAGE_PIXELS = None
    except Exception:
        pass
    img = Image.open(image_path)
    img = img.crop(crop)
    #stardist likes images on a 0-1 scale
    img = np.array(img)/255
    #load labels and subset to area of interest
    #crop is (left, upper, right, lower)
    #https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.crop
    labels_sparse = scipy.sparse.load_npz(labels_npz_path)
    #upper:lower, left:right
    labels_sparse = labels_sparse[crop[1]:crop[3], crop[0]:crop[2]]
    #reset labels within crop to start from 1 and go up by 1
    #which makes the stardist view more colourful in objects
    #calling this on [5,7,7,9] yields [1,2,2,3] which is what we want
    labels_sparse.data = scipy.stats.rankdata(labels_sparse.data, method="dense")
    labels = np.array(labels_sparse.todense())
    return render_label(labels, img=img, **kwargs)

def view_labels(image_path, labels_npz_path, 
                crop=None, 
                stardist_normalize=False, 
                fill=False, 
                border=True, 
                fill_palette=None, 
                fill_label_weight=0.5, 
                border_color=[255,255,0]
               ):
    '''
    Render segmentation results in a lightweight manner on a full image 
    level. Can do fills or borders. Returns image as ``np.array``.
    
    Input
    -----
    image_path : ``filepath``
        Path to image that was segmented.
    labels_npz_path : ``filepath``
        Path to sparse labels generated by ``b2c.stardist()``.
    crop : ``None`` or tuple of ``int``, optional (default: ``None``)
        A PIL-formatted crop specification - a four integer tuple, 
        provided as (left, upper, right, lower) coordinates. If ``None``, 
        will render full segmentation results.
    stardist_normalize : ``bool``, optional (default: ``False``)
        If ``True``, percentile normalise the input image prior to rendering 
        the segmentation labels on it.
    fill : ``bool``, optional (default: ``False``)
        If ``True``, render the objects in full, with transparency 
        controlled by ``fill_label_weight``.
    border : ``bool``, optional (default, ``True``)
        If ``True``, render a fully opaque border of a specified colour 
        around the objects
    fill_palette : ``None`` or ``np.array`` of ``np.uint8``, optional (default: ``None``)
        Two-dimensional ``np.array``, with rows corresponding to colours of 
        the palette and columns corresponding to RGB, in ``np.uint8``. If 
        ``None``, will use Seaborn's bright palette, skipping the pink due 
        to it rendering poorly against H&E images. Objects will be coloured 
        based on the remainder of dividing its label ID by the number of 
        colours.
    fill_label_weight : ``float``, optional (default: 0.5)
        Weight to assign the object fill render when constructing output. 
        0 is completely transparent, 1 is completely opaque.
    border_color : ``list`` of ``np.uint8``, optional (default: ``[255, 255, 0]``)
        Border colour in RGB.
    '''
    #load the sparse labels
    import skimage.segmentation
    import skimage
    labels_sparse = scipy.sparse.load_npz(labels_npz_path)
    #determine memory efficient dtype to load the image as
    #if we'll be normalising, we want np.float16 for optimal RAM footprint
    #otherwise use np.uint8
    if stardist_normalize:
        dtype = np.float16
    else:
        dtype = np.uint8
    if crop is None:
        #this will load greyscale as 3 channel, which is what we want here
        img = load_image(image_path, dtype=dtype)
    else:
        #PIL is better at handling crops memory efficiently than cv2
        try:
            from PIL import Image as _PILImage
            _PILImage.MAX_IMAGE_PIXELS = None
        except Exception:
            pass
        img = Image.open(image_path)
        #ensure that it's in RGB (otherwise there's a single channel for greyscale)
        img = np.array(img.crop(crop).convert('RGB'), dtype=dtype)
        #subset labels to area of interest
        #crop is (left, upper, right, lower)
        #https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.crop
        #upper:lower, left:right
        labels_sparse = labels_sparse[crop[1]:crop[3], crop[0]:crop[2]]
    #optionally normalise image
    if stardist_normalize:
        img = normalize(img)
        #actually cap the values - currently there are sub 0 and above 1 entries
        img[img<0] = 0
        img[img>1] = 1
        #turn back to uint8 for internal consistency
        img = (255*img).astype(np.uint8)
    #turn labels to COO for ease of position retrieval
    labels_sparse = labels_sparse.tocoo()
    if fill:
        if fill_palette is None:
            #use the seaborn bright palette, but remove the pink from it
            #as it blends too much into the H&E background
            fill_palette = (np.array(sns.color_palette("bright"))*255).astype(np.uint8)
            fill_palette = np.delete(fill_palette, 6, 0)
        #now we have a master list of pixels with objects to show
        #.row is [:,0], .col is [:,1]
        #extract the existing values from the image
        #and simultaneously get a fill colour by doing % on number of fill colours
        #weight the two together to get the new pixel value
        img[labels_sparse.row, labels_sparse.col, :] = \
            (1-fill_label_weight) * img[labels_sparse.row, labels_sparse.col, :] + \
            fill_label_weight * fill_palette[labels_sparse.data % fill_palette.shape[0], :]
    if border:
        #unfortunately the boundary finder wants a dense matrix, so turn our labels to it for a sec
        #turn the output back into a sparse COO, both for memory efficiency and pure convenience
        border_sparse = scipy.sparse.coo_matrix(skimage.segmentation.find_boundaries(np.array(labels_sparse.todense())))
        #can now easily colour the borders similar to what was done for the fill
        img[border_sparse.row, border_sparse.col, :] = border_color
    return img

def overlay_onto_img(img, labels_sparse, cdata, key, common_objects, 
                      fill_label_weight=1, 
                      cat_cmap="tab20",
                      cont_cmap="viridis"
                     ):
    '''
    Helper function used by ``b2c.view_cell_labels()`` to actually overlay 
    the metadata/expression onto the morphology image
    
    Input
    -----
    image_path : ``numpy.array``
        Morphology image in Numpy array form.
    labels_sparse : ``scipy.sparse.coo_matrix``
        Processed to just the pixels to render, 
    cdata : ``AnnData``
        Cell-level VisiumHD object with pertinent gene/metadata. Must have 
        ``.obs_names`` unchanged from ``b2c.bin_to_cell()`` output.
    key : ``str``
        ``.obs`` column (float/integer or categorical) or gene name 
        (expression taken from ``.X``) to colour the ``labels_sparse`` 
        pixels by.
    common_objects : ``numpy.array`` of ``int``
        Object IDs to plot.
    fill_label_weight : ``float``, optional (default: 1)
        Weight to assign the object fill render when constructing output. 
        0 is completely transparent, 1 is completely opaque.
    cat_cmap : ``str``, optional (default: ``"tab20"``)
        Colormap name (must be understood by ``seaborn.color_palette()``) to 
        use for categoricals.
    cont_cmap : ``str``, optional (default: ``"viridis"``)
        Colormap name (must be understood by ``matplotlib.colormaps.get_cmap()``) 
        to use for integers/floats.
    '''
    #are we in obs or in var?
    if key in cdata.obs.columns:
        #continuous or categorical?
        if ("float" in cdata.obs[key].dtype.name) or ("int" in cdata.obs[key].dtype.name):
            #we've got a continous
            #subset on common_objects, so the order matches them
            #also need to turn to string to match cdata.obs_names
            vals = cdata.obs.loc[[str(i) for i in common_objects], key].values
            #we'll continue processing shortly outside the ifs
        elif "category" in cdata.obs[key].dtype.name:
            #we've got a categorical
            #pull out the category indices for each of the common objects
            #and store them in a numpy array that can be indexed on the object in int form
            cats = np.zeros(np.max(common_objects)+1, dtype=np.int32)
            #need to turn common_objects back to string so cdata can be subset on them
            cats[common_objects] = cdata.obs.loc[[str(i) for i in common_objects], key].cat.codes
            #store the original present category codes for legend purposes
            cats_unique_original = np.unique(cats[common_objects])
            #reset the cats[common_objects] to start from 0 and go up by 1
            #which may avoid some palette overlaps in some corner case scenarios
            #calling this on [5,7,7,9] yields [1,2,2,3] which is what we want
            #except then shift it back by 1 so it starts from 0
            cats[common_objects] = scipy.stats.rankdata(cats[common_objects], method="dense") - 1
            #make a cmap with the correct number of colours
            #convert to uint8 for internal consistency
            fill_palette = (np.array(sns.color_palette(cat_cmap, n_colors=len(cats_unique_original)))*255).astype(np.uint8)
            #now we have a master list of pixels with objects to show
            #.row is [:,0], .col is [:,1]
            #extract the existing values from the image
            #pull out the category index by subsetting on the actual object ID
            #no need to do % because we have one entry in the palette per category
            #weight the two together to get the new pixel value
            img[labels_sparse.row, labels_sparse.col, :] = \
                (1-fill_label_weight) * img[labels_sparse.row, labels_sparse.col, :] + \
                fill_label_weight * fill_palette[cats[labels_sparse.data], :]
            #set up legend
            #figsize is largely irrelevant because of bbox_inches='tight' when saving
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.axis("off")
            #alright, there's some stuff going on here. let's explain
            #the categories don't actually get subset when you pull out a chunk of the vector
            #which kinda makes sense but is annoying
            #meanwhile we want to minimise our ID footprint to try to use the colour map nicely
            #as such, earlier we made cats_unique_original, having all the codes from the objects
            #and then we reset the cats to start at 0 and go up by 1
            #now we can get a correspondence of those to cats_unique_original
            #by doing a subset and np.unique(), as this is monotonically preserved
            #once again, no need for % because we have an entry for each category
            legend_patches = [
                matplotlib.patches.Patch(color=fill_palette[i,:]/255.0, 
                                         label=cdata.obs[key].cat.categories[j]
                                        )
                for i, j in zip(np.unique(cats[common_objects]), cats_unique_original)
            ]
            ax.legend(handles=legend_patches, loc="center", title=key, frameon=False)
            #close the thing so it doesn't randomly show. still there though
            plt.close(fig)
            #okay we're happy. return to recycle continuous processing code
            return img, fig
        else:
            #we've got to raise an error
            raise ValueError("``cdata.obs['"+key+"']`` must be a float, int, or categorical")
    elif key in cdata.var_names:
        #gene, continuous
        #fast enough to just subset the cdata, as always turn to string
        #then regardless if it's sparse or dense data need to .toarray().flatten()
        #if it's sparse then this turns it dense
        #if it's dense then it gets out of ArrayView back into a normal array
        #and then flattened regardless
        vals = cdata[[str(i) for i in common_objects]][:,key].X.toarray().flatten()
    else:
        #we've got a whiff
        raise ValueError("'"+key+"' not found in ``cdata.obs`` or ``cdata.var``")
    #we're out here. so we're processing a continuous, be it obs or var
    #set up legend while vals are on their actual scale
    #figsize is largely irrelevant because of bbox_inches='tight' when saving
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis("off")
    colormap = matplotlib.colormaps.get_cmap(cont_cmap)
    norm = matplotlib.colors.Normalize(vmin=np.min(vals), vmax=np.max(vals))
    sm = matplotlib.cm.ScalarMappable(cmap=colormap, norm=norm)
    fig.colorbar(sm, ax=ax, orientation="horizontal", label=key)
    #close the thing so it doesn't randomly show. still there though
    plt.close(fig)
    #for continuous operations, we need a 0-1 scaled vector of values
    vals = (vals-np.min(vals))/(np.max(vals)-np.min(vals))
    #construct a fill palette by busting out a colormap
    #and then getting its values at vals, and sending them to a prepared numpy array
    #that can then be subset on the object ID to get its matching RGB of the continuous value
    fill_palette = np.zeros((np.max(common_objects)+1, 3))
    #send to common_objects, matching vals order. also convert to uint8 for consistency
    fill_palette[common_objects, :] = (colormap(vals)[:,:3]*255).astype(np.uint8)
    #now we have a master list of pixels with objects to show
    #.row is [:,0], .col is [:,1]
    #extract the existing values from the image
    #and simultaneously get a fill colour by subsetting the palette on the label ID
    #no fancy % needed here as each object has its own fill value prepared
    #weight the two together to get the new pixel value
    img[labels_sparse.row, labels_sparse.col, :] = \
        (1-fill_label_weight) * img[labels_sparse.row, labels_sparse.col, :] + \
        fill_label_weight * fill_palette[labels_sparse.data, :]
    return img, fig
    
    
def view_cell_labels(image_path, labels_npz_path, cdata, 
                     fill_key=None, 
                     border_key=None, 
                     crop=None, 
                     stardist_normalize=False, 
                     fill_label_weight=1, 
                     thicken_border=True,
                     cat_cmap="tab20",
                     cont_cmap="viridis"
                    ):
    '''
    Colour morphology segmentations by cell-level metadata or gene 
    expression. Can do fills or borders. Returns image as ``np.array``, 
    and a dictionary of legends (as matplotlib figures) named after the 
    keys it used.
    
    Input
    -----
    image_path : ``filepath``
        Path to morphology image that was segmented.
    labels_npz_path : ``filepath``
        Path to sparse morphology labels generated by ``b2c.stardist()``.
    cdata : ``AnnData``
        Cell-level VisiumHD object with pertinent gene/metadata. Must have 
        ``.obs_names`` unchanged from ``b2c.bin_to_cell()`` output.
    fill_key : ``str`` or ``None``, optional (default: ``None``)
        ``.obs`` column (float/integer or categorical) or gene name 
        (expression taken from ``.X``) to colour the fill by. 
        Skipped if ``None``.
    border_key : ``str`` or ``None``, optional (default: ``None``)
        ``.obs`` column (float/integer or categorical) or gene name 
        (expression taken from ``.X``) to colour the border by. 
        Skipped if ``None``.
    crop : ``None`` or tuple of ``int``, optional (default: ``None``)
        A PIL-formatted crop specification - a four integer tuple, 
        provided as (left, upper, right, lower) coordinates. If ``None``, 
        will render full segmentation results.
    stardist_normalize : ``bool``, optional (default: ``False``)
        If ``True``, percentile normalise the input image prior to rendering 
        the segmentation labels on it.
    fill_label_weight : ``float``, optional (default: 1)
        Weight to assign the object fill render when constructing output. 
        0 is completely transparent, 1 is completely opaque.
    thicken_border : ``bool``, optional (default: ``True``)
        The default identified border is thinner than that of 
        ``b2c.view_labels()``. Setting this to ``True``, thickens it for 
        ease of viewing.
    cat_cmap : ``str``, optional (default: ``"tab20"``)
        Colormap name (must be understood by ``seaborn.color_palette()``) to 
        use for categoricals.
    cont_cmap : ``str``, optional (default: ``"viridis"``)
        Colormap name (must be understood by ``matplotlib.colormaps.get_cmap()``) 
        to use for integers/floats.
    '''
    #load the sparse labels
    import skimage.segmentation
    import skimage
    labels_sparse = scipy.sparse.load_npz(labels_npz_path)
    #determine memory efficient dtype to load the image as
    #if we'll be normalising, we want np.float16 for optimal RAM footprint
    #otherwise use np.uint8
    if stardist_normalize:
        dtype = np.float16
    else:
        dtype = np.uint8
    if crop is None:
        #this will load greyscale as 3 channel, which is what we want here
        img = load_image(image_path, dtype=dtype)
    else:
        #PIL is better at handling crops memory efficiently than cv2
        img = Image.open(image_path)
        #ensure that it's in RGB (otherwise there's a single channel for greyscale)
        img = np.array(img.crop(crop).convert('RGB'), dtype=dtype)
        #subset labels to area of interest
        #crop is (left, upper, right, lower)
        #https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.crop
        #upper:lower, left:right
        labels_sparse = labels_sparse[crop[1]:crop[3], crop[0]:crop[2]]
    #optionally normalise image
    if stardist_normalize:
        img = normalize(img)
        #actually cap the values - currently there are sub 0 and above 1 entries
        img[img<0] = 0
        img[img>1] = 1
        #turn back to uint8 for internal consistency
        img = (255*img).astype(np.uint8)
    #turn labels to COO for ease of position retrieval
    labels_sparse = labels_sparse.tocoo()
    #identify overlap between label objects and cdata observations
    #which should be the same nomenclature for the morphology segmentation
    #just as strings, while the labels are ints
    common_objects = np.sort(list(set(np.unique(labels_sparse.data)).intersection(set([int(i) for i in cdata.obs_names]))))
    #kick out filtered out objects from segmentation results
    labels_sparse.data[~np.isin(labels_sparse.data, common_objects)] = 0
    labels_sparse.eliminate_zeros()
    #catch legends to dictionary for returning later
    legends = {}
    #do a fill if requested
    if fill_key is not None:
        #legend comes in the form of a figure
        img, fig = overlay_onto_img(img=img, 
                                    labels_sparse=labels_sparse, 
                                    cdata=cdata, 
                                    key=fill_key, 
                                    common_objects=common_objects, 
                                    fill_label_weight=fill_label_weight,
                                    cat_cmap=cat_cmap,
                                    cont_cmap=cont_cmap
                                   )
        legends[fill_key] = fig
    #do a border if requested
    if border_key is not None:
        #actually get the border
        #unfortunately the boundary finder wants a dense matrix, so turn our labels to it for a sec
        #go for inner borders as that's all we care about, and that's less pixels to worry about
        #keep the border dense because of implementation reasons. it spikes RAM anyway when it's made
        border = skimage.segmentation.find_boundaries(np.array(labels_sparse.todense()), mode="inner")
        #whether we thicken or not, we need nonzero coordinates
        coords = np.nonzero(border)
        if thicken_border:
            #we're thickening. skimage.segmentation.expand_labels() explodes when asked to do this
            #the following gets the job done quicker and with lower RAM footprint
            #take existing nonzero coordinates and move them to the left, right, up and down by 1
            border_rows = np.hstack([
                np.clip(coords[0]-1, a_min=0, a_max=None),
                np.clip(coords[0]+1, a_min=None, a_max=border.shape[0]-1),
                coords[0],
                coords[0]
            ])
            border_cols = np.hstack([
                coords[1],
                coords[1],
                np.clip(coords[1]-1, a_min=0, a_max=None),
                np.clip(coords[1]+1, a_min=None, a_max=border.shape[1]-1)
            ])
            #set the positions to True. this is BLAZING FAST compared to sparse attempts
            #or, surprisingly, trying to np.unique() to get non-duplicate coordinates
            #and the entire reason we kept the borders dense for this process
            border[border_rows, border_cols] = True
            #update our nonzero coordinates
            coords = np.nonzero(border)
        #to assign borders back to objects, subset the object labels to just the border pixels
        #technically need to construct a new COO matrix for it, pulling out values at the border coordinates
        #use (data, (row, col)) constructor
        #also need to turn labels to CSR to be able to pull out their values
        #which results in a 2d numpy matrix, so turn to 1D array or constructor errors
        labels_sparse = scipy.sparse.coo_matrix((np.array(labels_sparse.tocsr()[coords[0], coords[1]]).flatten(), coords), shape=labels_sparse.shape)
        #there will be zeroes from the thickener, border mode="inner" means no thickener no zeros
        labels_sparse.eliminate_zeros()
        #can now run the overlayer, set weights to 1 to have fully opaque borders
        #legend comes in the form of a figure
        img, fig = overlay_onto_img(img=img, 
                                    labels_sparse=labels_sparse, 
                                    cdata=cdata, 
                                    key=border_key, 
                                    common_objects=common_objects, 
                                    fill_label_weight=1,
                                    cat_cmap=cat_cmap,
                                    cont_cmap=cont_cmap
                                   )
        legends[border_key] = fig
    return img, legends

#as PR'd to scanpy: https://github.com/scverse/scanpy/pull/2992
def read_visium(
    path: Path | str,
    genome: str | None = None,
    *,
    count_file: str = "filtered_feature_bc_matrix.h5",
    library_id: str | None = None,
    load_images: bool | None = True,
    source_image_path: Path | str | None = None,
    spaceranger_image_path: Path | str | None = None,
) -> AnnData:
    """\
    Read 10x-Genomics-formatted visum dataset.

    In addition to reading regular 10x output,
    this looks for the `spatial` folder and loads images,
    coordinates and scale factors.
    Based on the `Space Ranger output docs`_.

    See :func:`~scanpy.pl.spatial` for a compatible plotting function.

    .. _Space Ranger output docs: https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/overview

    Parameters
    ----------
    path
        Path to directory for visium datafiles.
    genome
        Filter expression to genes within this genome.
    count_file
        Which file in the passed directory to use as the count file. Typically would be one of:
        'filtered_feature_bc_matrix.h5' or 'raw_feature_bc_matrix.h5'.
    library_id
        Identifier for the visium library. Can be modified when concatenating multiple adata objects.
    source_image_path
        Path to the high-resolution tissue image. Path will be included in
        `.uns["spatial"][library_id]["metadata"]["source_image_path"]`.
    spaceranger_image_path
        Path to the folder containing the spaceranger output hires/lowres tissue images. If `None`, 
        will go with the `spatial` folder of the provided `path`.

    Returns
    -------
    Annotated data matrix, where observations/cells are named by their
    barcode and variables/genes by gene name. Stores the following information:

    :attr:`~anndata.AnnData.X`
        The data matrix is stored
    :attr:`~anndata.AnnData.obs_names`
        Cell names
    :attr:`~anndata.AnnData.var_names`
        Gene names for a feature barcode matrix, probe names for a probe bc matrix
    :attr:`~anndata.AnnData.var`\\ `['gene_ids']`
        Gene IDs
    :attr:`~anndata.AnnData.var`\\ `['feature_types']`
        Feature types
    :attr:`~anndata.AnnData.obs`\\ `[filtered_barcodes]`
        filtered barcodes if present in the matrix
    :attr:`~anndata.AnnData.var`
        Any additional metadata present in /matrix/features is read in.
    :attr:`~anndata.AnnData.uns`\\ `['spatial']`
        Dict of spaceranger output files with 'library_id' as key
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['images']`
        Dict of images (`'hires'` and `'lowres'`)
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['scalefactors']`
        Scale factors for the spots
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['metadata']`
        Files metadata: 'chemistry_description', 'software_version', 'source_image_path'
    :attr:`~anndata.AnnData.obsm`\\ `['spatial']`
        Spatial spot coordinates, usable as `basis` by :func:`~scanpy.pl.embedding`.
    """
    
    path = Path(path)
    #if not provided, assume the hires/lowres images are in the same folder as everything
    #except in the spatial subdirectory
    if spaceranger_image_path is None:
        spaceranger_image_path = path / "spatial"
    else:
        spaceranger_image_path = Path(spaceranger_image_path)
    adata = read_10x_h5(path / count_file, genome=genome)

    adata.uns["spatial"] = dict()

    from h5py import File

    with File(path / count_file, mode="r") as f:
        attrs = dict(f.attrs)
    if library_id is None:
        library_id = str(attrs.pop("library_ids")[0], "utf-8")

    adata.uns["spatial"][library_id] = dict()

    if load_images:
        tissue_positions_file = (
            path / "spatial/tissue_positions.csv"
            if (path / "spatial/tissue_positions.csv").exists()
            else path / "spatial/tissue_positions.parquet" if (path / "spatial/tissue_positions.parquet").exists()
            else path / "spatial/tissue_positions_list.csv"
        )
        files = dict(
            tissue_positions_file=tissue_positions_file,
            scalefactors_json_file=path / "spatial/scalefactors_json.json",
            hires_image=spaceranger_image_path / "tissue_hires_image.png",
            lowres_image=spaceranger_image_path / "tissue_lowres_image.png",
        )

        # check if files exists, continue if images are missing
        for f in files.values():
            if not f.exists():
                if any(x in str(f) for x in ["hires_image", "lowres_image"]):
                    logg.warning(
                        f"You seem to be missing an image file.\n"
                        f"Could not find '{f}'."
                    )
                else:
                    raise OSError(f"Could not find '{f}'")

        adata.uns["spatial"][library_id]["images"] = dict()
        for res in ["hires", "lowres"]:
            try:
                adata.uns["spatial"][library_id]["images"][res] = imread(
                    str(files[f"{res}_image"])
                )
            except Exception:
                raise OSError(f"Could not find '{res}_image'")

        # read json scalefactors
        adata.uns["spatial"][library_id]["scalefactors"] = json.loads(
            files["scalefactors_json_file"].read_bytes()
        )

        adata.uns["spatial"][library_id]["metadata"] = {
            k: (str(attrs[k], "utf-8") if isinstance(attrs[k], bytes) else attrs[k])
            for k in ("chemistry_description", "software_version")
            if k in attrs
        }

        # read coordinates
        if files["tissue_positions_file"].name.endswith(".csv"):
            positions = pd.read_csv(
                files["tissue_positions_file"],
                header=0 if tissue_positions_file.name == "tissue_positions.csv" else None,
                index_col=0,
            )
        elif files["tissue_positions_file"].name.endswith(".parquet"):
            positions = pd.read_parquet(files["tissue_positions_file"])
            #need to set the barcode to be the index
            positions.set_index("barcode", inplace=True)
        positions.columns = [
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_col_in_fullres",
            "pxl_row_in_fullres",
        ]

        adata.obs = adata.obs.join(positions, how="left")

        adata.obsm["spatial"] = adata.obs[
            ["pxl_row_in_fullres", "pxl_col_in_fullres"]
        ].to_numpy()
        adata.obs.drop(
            columns=["pxl_row_in_fullres", "pxl_col_in_fullres"],
            inplace=True,
        )

        # put image path in uns
        if source_image_path is not None:
            # get an absolute path
            source_image_path = str(Path(source_image_path).resolve())
            adata.uns["spatial"][library_id]["metadata"]["source_image_path"] = str(
                source_image_path
            )

    return adata

def destripe_counts(adata, counts_key="n_counts", adjusted_counts_key="n_counts_adjusted"):
    '''
    Scale each row (bin) of ``adata.X`` to have ``adjusted_counts_key`` 
    rather than ``counts_key`` total counts.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Raw counts, needs to have ``counts_key`` 
        and ``adjusted_counts_key`` in ``.obs``.
    counts_key : ``str``, optional (default: ``"n_counts"``)
        Name of ``.obs`` column with raw counts per bin.
    adjusted_counts_key : ``str``, optional (default: ``"n_counts_adjusted"``)
        Name of ``.obs`` column storing the desired destriped counts per bin.
    '''
    #scanpy's utility function to make sure the anndata is not a view
    #if it is a view then weird stuff happens when you try to write to its .X
    sc._utils.view_to_actual(adata)
    #adjust the count matrix to have n_counts_adjusted sum per bin (row)
    #premultiplying by a diagonal matrix multiplies each row by a value: https://solitaryroad.com/c108.html
    bin_scaling = scipy.sparse.diags(adata.obs[adjusted_counts_key]/adata.obs[counts_key])
    adata.X = bin_scaling.dot(adata.X)

def destripe(adata, quantile=0.99, counts_key="n_counts", factor_key="destripe_factor", adjusted_counts_key="n_counts_adjusted", adjust_counts=True):
    '''
    Correct the raw counts of the input object for known variable width of 
    VisiumHD 2um bins. Scales the total UMIs per bin on a per-row and 
    per-column basis, dividing by the specified ``quantile``. The resulting 
    value is stored in ``.obs[factor_key]``, and is multiplied by the 
    corresponding total UMI ``quantile`` to get ``.obs[adjusted_counts_key]``.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Raw counts, needs to have ``counts_key`` in 
        ``.obs``.
    quantile : ``float``, optional (default: 0.99)
        Which row/column quantile to use for the computation.
    counts_key : ``str``, optional (default: ``"n_counts"``)
        Name of ``.obs`` column with raw counts per bin.
    factor_key : ``str``, optional (default: ``"destripe_factor"``)
        Name of ``.obs`` column to hold computed factor prior to reversing to 
        count space.
    adjusted_counts_key : ``str``, optional (default: ``"n_counts_adjusted"``)
        Name of ``.obs`` column for storing the destriped counts per bin.
    adjust_counts : ``bool``, optional (default: ``True``)
        Whether to use the computed adjusted count total to adjust the counts in 
        ``adata.X``.
    '''
    # Ensure counts_key exists in obs (compute from X if missing)
    if counts_key not in adata.obs.columns:
        adata.obs[counts_key] = np.asarray(adata.X.sum(axis=1)).flatten()
    #apply destriping via sequential quantile scaling
    #get specified quantile per row
    quant = adata.obs.groupby("array_row")[counts_key].quantile(quantile)
    #divide each row by its quantile (order of obs[counts_key] and obs[array_row] match)
    adata.obs[factor_key] = adata.obs[counts_key] / adata.obs["array_row"].map(quant)
    #repeat on columns
    quant = adata.obs.groupby("array_col")[factor_key].quantile(quantile)
    adata.obs[factor_key] /= adata.obs["array_col"].map(quant)
    #propose adjusted counts as the global quantile multipled by the destripe factor
    adata.obs[adjusted_counts_key] = adata.obs[factor_key] * np.quantile(adata.obs[counts_key], quantile)
    #correct the count space unless told not to
    if adjust_counts:
        destripe_counts(adata, counts_key=counts_key, adjusted_counts_key=adjusted_counts_key)

def check_array_coordinates(adata, row_max=3349, col_max=3349):
    '''
    Assess the relationship between ``.obs["array_row"]``/``.obs["array_col"]`` 
    and ``.obsm["spatial"]``, as the array coordinates tend to have their 
    origin in places that isn't the top left of the image. Array coordinates 
    are deemed to be flipped or not based on the Pearson correlation with the 
    corresponding spatial dimension. Creates ``.uns["bin2cell"]["array_check"]`` 
    that is used by ``b2c.grid_image()``, ``b2c.insert_labels()`` and 
    ``b2c.get_crop()``.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object.
    row_max : ``int``, optional (default: 3349)
        Maximum possible ``array_row`` value.
    col_max : ``int``, optional (default: 3349)
        Maximum possible ``array_col`` value.
    '''
    #store the calls here
    if not "bin2cell" in adata.uns:
        adata.uns["bin2cell"] = {}
    adata.uns["bin2cell"]["array_check"] = {}
    #we'll need to check both the rows and columns
    for axis in ["row", "col"]:
        #we may as well store the maximum immediately
        adata.uns["bin2cell"]["array_check"][axis] = {}
        if axis == "row":
            adata.uns["bin2cell"]["array_check"][axis]["max"] = row_max
        elif axis == "col":
            adata.uns["bin2cell"]["array_check"][axis]["max"] = col_max
        #are we going to be extracting values for a single col or row?
        #set up where we'll be looking to get values to correlate
        if axis == "col":
            single_axis = "row"
            #spatial[:,0] matches axis_col (note at start)
            spatial_axis = 0
        elif axis == "row":
            single_axis = "col"
            #spatial[:,1] matches axis_row (note at start)
            spatial_axis = 1
        #get the value of the other axis with the highest number of bins present
        val = adata.obs['array_'+single_axis].value_counts().index[0]
        #get a boolean mask of the bins of that value
        mask = (adata.obs['array_'+single_axis] == val)
        #use the mask to get the spatial and array coordinates to compare
        array_vals = adata.obs.loc[mask,'array_'+axis].values
        spatial_vals = adata.obsm['spatial'][mask, spatial_axis]
        #check whether they're positively or negatively correlated
        if scipy.stats.pearsonr(array_vals, spatial_vals)[0] < 0:
            adata.uns["bin2cell"]["array_check"][axis]["flipped"] = True
        else:
            adata.uns["bin2cell"]["array_check"][axis]["flipped"] = False

def grid_image(adata, val, log1p=False, mpp=2, sigma=None, save_path=None):
    '''
    Create an image of a specified ``val`` across the array coordinate grid. 
    Orientation matches the morphology image and spatial coordinates.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Should have array coordinates evaluated by 
        calling ``b2c.check_array_coordinates()``.
    val : ``str``
        ``.obs`` column or variable name to visualise.
    log1p : ``bool``, optional (default: ``False``)
        Whether to log1p-transform the values in the image.
    mpp : ``float``, optional (default: 2)
        Microns per pixel of the output image.
    sigma : ``float`` or ``None``, optional (default: ``None``)
        If not ``None``, will run the final image through 
        ``skimage.filters.gaussian()`` with the provided sigma value.
    save_path : ``filepath`` or ``None``, optional (default: ``None``)
        If specified, will save the generated image to this path (e.g. for 
        StarDist use). If not provided, will return image.
    '''
    #pull out the values for the image. start by checking .obs
    import skimage
    import cv2
    if val in adata.obs.columns:
        vals = adata.obs[val].values.copy()
    elif val in adata.var_names:
        #if not in obs, it's presumably in the feature space
        vals = adata[:, val].X
        #may be sparse
        if scipy.sparse.issparse(vals):
            vals = vals.todense()
        #turn it to a flattened numpy array so it plays nice
        vals = np.asarray(vals).flatten()
    else:
        #failed to find
        raise ValueError('"'+val+'" not located in ``.obs`` or ``.var_names``')
    #make the values span from 0 to 255
    vals = (255 * (vals-np.min(vals))/(np.max(vals)-np.min(vals))).astype(np.uint8)
    #optionally log1p
    if log1p:
        vals = np.log1p(vals)
        vals = (255 * (vals-np.min(vals))/(np.max(vals)-np.min(vals))).astype(np.uint8)
    #spatial coordinates match what's going on in the image, array coordinates may not
    #have we checked if the array row/col need flipping?
    if not "bin2cell" in adata.uns:
        check_array_coordinates(adata)
    elif not "array_check" in adata.uns["bin2cell"]:
        check_array_coordinates(adata)
    #can now create an empty image the shape of the grid and stick the values in based on the coordinates
    #need to nudge up the dimensions by 1 as python is zero-indexed
    img = np.zeros((adata.uns["bin2cell"]["array_check"]["row"]["max"]+1, 
                    adata.uns["bin2cell"]["array_check"]["col"]["max"]+1), 
                   dtype=np.uint8)
    img[adata.obs['array_row'], adata.obs['array_col']] = vals
    #check if the row or column need flipping
    if adata.uns["bin2cell"]["array_check"]["row"]["flipped"]:
        img = np.flip(img, axis=0)
    if adata.uns["bin2cell"]["array_check"]["col"]["flipped"]:
        img = np.flip(img, axis=1)
    #resize image to appropriate mpp. bins are 2um apart, so current mpp is 2
    #need to reverse dimensions relative to the array for cv2, and turn to int
    if mpp != 2:
        dim = np.round(np.array(img.shape) * 2/mpp).astype(int)[::-1]
        img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    #run through the gaussian filter if need be
    if sigma is not None:
        img = skimage.filters.gaussian(img, sigma=sigma)
        img = (255 * (img-np.min(img))/(np.max(img)-np.min(img))).astype(np.uint8)
    #save or return image
    if save_path is not None:
        cv2.imwrite(save_path, img)
    else:
        return img

def check_bin_image_overlap(adata, img, overlap_threshold=0.9):
    '''
    Assess the number of bins that fall within the source image coordinate 
    space. If an insufficient proportion are captured then throw an informative 
    error.
    
    Obsoleted by ``b2c.actual_vs_inferred_image_shape()``.
    
    Input
    -----
    adata : ``AnnData``
        2um bin Visium object.
    img : ``np.array``
        Loaded full resolution morphology image, prior to any cropping/scaling.
    overlap_threshold : ``float``, optional (default: 0.9)
        Throw the error if fewer than this fraction of bin spatial coordinates 
        fall within the dimensions of the image.
    '''
    #spatial[:,1] matches img[:,0] and spatial[:,0] matches img[:,1]
    #check how many fall within the dimensions, and get a fraction of total bin count
    overlap = np.sum((adata.obsm["spatial"][:,1] < img.shape[0]) & (adata.obsm["spatial"][:,0] < img.shape[1])) / adata.shape[0]
    if overlap < overlap_threshold:
        #something is amiss. print a bunch of diagnostics
        print("Source image dimensions: "+ str(img.shape))
        #the end user does not need to know about the messiness of the representations
        #pre-format the spatial maxima to match the dimensions of the image
        print("Corresponding ``.obsm['spatial']`` maxima: "+str(np.max(adata.obsm["spatial"], axis=0)[::-1]))
        raise ValueError("Only "+str(100*overlap)+"% of bins fall within image. Are you running with ``source_image_path`` set to the full resolution morphology image, as used for ``--image`` in Spaceranger?")

def actual_vs_inferred_image_shape(adata, img, ratio_threshold=0.99):
    '''
    Compare the shape of the actual morphology image versus what the shape of 
    the morphology image that was used for Spaceranger appears to be from 
    information stored for the hires. If there's a mismatch throw an 
    informative error with both sets of dimensions.
    
    Input
    -----
    adata : ``AnnData``
        2um bin Visium object.
    img : ``np.array``
        Loaded full resolution morphology image, prior to any cropping/scaling.
    ratio_threshold : ``float``, optional (default: 0.99)
        Throw the error if any ratio of corresponding actual and inferred 
        dimensions falls below this value.
    '''
    #identify name of spatial key for subsequent access of fields
    library = list(adata.uns['spatial'].keys())[0]
    #infer the dimensions as the shape of the hires tissue image
    #divided by the hires scale factor
    inferred_dim = np.array(adata.uns['spatial'][library]['images']['hires'].shape)[:2] / adata.uns['spatial'][library]['scalefactors']['tissue_hires_scalef']
    #retrieve actual dimension as we have the full morphology image loaded
    actual_dim = np.array(img.shape)[:2]
    #do the two match, within some tolerance of rounding etc?
    #divide both ways just in case
    if np.min(np.hstack((actual_dim/inferred_dim, inferred_dim/actual_dim))) < ratio_threshold:
        raise ValueError("Morphology image dimension mismatch. Dimensions inferred from Spaceranger output: "+str(inferred_dim)+", actual image dimensions: "+str(actual_dim)+". Are you running with ``source_image_path`` set to the full resolution morphology image, as used for ``--image`` in Spaceranger?")

def mpp_to_scalef(adata, mpp):
    '''
    Compute a scale factor for a specified mpp value.
    
    Input
    -----
    adata : ``AnnData``
        2um bin Visium object.
    mpp : ``float``
        Microns per pixel to report scale factor for.
    '''
    #identify name of spatial key for subsequent access of fields
    library = list(adata.uns['spatial'].keys())[0]
    #get original image mpp value
    mpp_source = adata.uns['spatial'][library]['scalefactors']['microns_per_pixel']
    #our scale factor is the original mpp divided by the new mpp
    return mpp_source/mpp

def get_mpp_coords(adata, basis="spatial", spatial_key="spatial", mpp=None):
    '''
    Get an mpp-adjusted representation of spatial or array coordinates of the 
    provided object. Origin in top left, dimensions correspond to ``np.array()`` 
    representation of image (``[:,0]`` is up-down, ``[:,1]`` is left-right). 
    The resulting coordinates are integers for ease of retrieval of labels from 
    arrays or defining crops.
    
    adata : ``AnnData``
        2um bin VisiumHD object.
    basis : ``str``, optional (default: ``"spatial"``)
        Whether to get ``"spatial"`` or ``"array"`` coordinates. The former is 
        the source morphology image, the latter is a GEX-based grid representation.
    spatial_key : ``str``, optional (default: ``"spatial"``)
        Only used with ``basis="spatial"``. Needs to be present in ``.obsm``. 
        Rounded coordinates will be used to represent each bin when retrieving 
        labels.
    mpp : ``float`` or ``None``, optional (default: ``None``)
        The mpp value. Mandatory for GEX (``basis="array"``), if not provided 
        with morphology (``basis="spatial"``) will assume full scale image.
    '''
    #if we're using array coordinates, is there an mpp provided?
    if basis == "array" and mpp is None:
        raise ValueError("Need to specify mpp if working with array coordinates.")
    if basis == "spatial":
        if mpp is not None:
            #get necessary scale factor
            scalef = mpp_to_scalef(adata, mpp=mpp)
        else:
            #no mpp implies full blown morphology image, so scalef is 1
            scalef = 1
        #get the matching coordinates, rounding to integers makes this agree
        #need to reverse them here to make the coordinates match the image, as per note at start
        #multiply by the scale factor to account for possible custom mpp morphology image
        coords = (adata.obsm[spatial_key]*scalef).astype(int)[:,::-1]
    elif basis == "array":
        #generate the pixels in the GEX image at the specified mpp
        #which actually correspond to the locations of the bins
        #easy to define scale factor as starting array mpp is 2
        scalef = 2/mpp
        coords = np.round(adata.obs[['array_row','array_col']].values*scalef).astype(int)
        #need to flip axes maybe
        #need to scale up maximum appropriately
        if adata.uns["bin2cell"]["array_check"]["row"]["flipped"]:
            coords[:,0] = np.round(adata.uns["bin2cell"]["array_check"]["row"]["max"]*scalef).astype(int) - coords[:,0]
        if adata.uns["bin2cell"]["array_check"]["col"]["flipped"]:
            coords[:,1] = np.round(adata.uns["bin2cell"]["array_check"]["col"]["max"]*scalef).astype(int) - coords[:,1]
    return coords

def get_crop(adata, basis="spatial", spatial_key="spatial", mpp=None, buffer=0):
    '''
    Get a PIL-formatted crop tuple from a provided object and coordinate 
    representation.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object.
    basis : ``str``, optional (default: ``"spatial"``)
        Whether to use ``"spatial"`` or ``"array"`` coordinates. The former is 
        the source morphology image, the latter is a GEX-based grid representation.
    spatial_key : ``str``, optional (default: ``"spatial"``)
        Only used with ``basis="spatial"``. Needs to be present in ``.obsm``. 
        Rounded coordinates will be used to represent each bin when retrieving 
        labels.
    mpp : ``float`` or ``None``, optional (default: ``None``)
        The micron per pixel value to use. Mandatory for GEX (``basis="array"``), 
        if not provided with morphology (``basis="spatial"``) will assume full scale 
        image.
    buffer : ``int``, optional (default: 0)
        How many extra pixels to include to each side the cropped grid for 
        extra visualisation.
    '''
    #get the appropriate coordinates, be they spatial or array, at appropriate mpp
    coords = get_mpp_coords(adata, basis=basis, spatial_key=spatial_key, mpp=mpp)
    #PIL crop is defined as a tuple of (left, upper, right, lower) coordinates
    #coords[:,0] is up-down, coords[:,1] is left-right
    #don't forget to add/remove buffer, and to not go past 0
    return (np.max([np.min(coords[:,1])-buffer, 0]), 
            np.max([np.min(coords[:,0])-buffer, 0]), 
            np.max(coords[:,1])+buffer, 
            np.max(coords[:,0])+buffer
           )

def scaled_he_image(adata, mpp=1, crop=True, buffer=150, 
                    spatial_cropped_key=None, store=True, 
                    img_key=None, save_path=None,
                    backend='tifffile'):
    '''
    Create a custom microns per pixel render of the full scale H&E image for 
    visualisation and downstream application. Store resulting image and its 
    corresponding size factor in the object. If cropping to just the spatial 
    grid, also store the cropped spatial coordinates. Optionally save to file.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Path to high resolution H&E image provided via 
        ``source_image_path`` to ``b2c.read_visium()``.
    mpp : ``float``, optional (default: 1)
        Microns per pixel of the desired H&E image to create.
    crop : ``bool``, optional (default: ``True``)
        If ``True``, will limit the image to the actual spatial coordinate area, 
        with ``buffer`` added to each dimension.
    buffer : ``int``, optional (default: 150)
        Only used with ``crop=True``. How many extra pixels (in original 
        resolution) to include on each side of the captured spatial grid.
    spatial_cropped_key : ``str`` or ``None``, optional (default: ``None``)
        Only used with ``crop=True``. ``.obsm`` key to store the adjusted 
        spatial coordinates in. If ``None``, defaults to 
        ``"spatial_cropped_X_buffer"``, where ``X`` is the value of ``buffer``.
    store : ``bool``, optional (default: ``True``)
        Whether to store the generated image within the object.
    img_key : ``str`` or ``None``, optional (default: ``None``)
        Only used with ``store=True``. The image key to store the image 
        under in the object. If ``None``, defaults to ``"X_mpp_Y_buffer"``, 
        where ``X`` is the value of ``mpp`` and ``Y`` is the value of 
        ``buffer`` in the instance of ``crop=True``. If no cropping is to 
        be done, defaults to ``"X_mpp"``.
    save_path : ``filepath`` or ``None``, optional (default: ``None``)
        If specified, will save the generated image to this path (e.g. for 
        StarDist use).
    '''
    #identify name of spatial key for subsequent access of fields
    import cv2
    library = list(adata.uns['spatial'].keys())[0]
    #retrieve specified source image path and load it
    img = load_image(adata.uns['spatial'][library]['metadata']['source_image_path'],
                     backend=backend)
    #assess that the image dimensions match what they're supposed to be
    #if not, inform the user what image they should retrieve and use
    actual_vs_inferred_image_shape(adata, img)
    #crop image if necessary
    if crop:
        crop_coords = get_crop(adata, basis="spatial", spatial_key="spatial", mpp=None, buffer=buffer)
        #this is already capped at a minimum of 0, so can just subset freely
        #left, upper, right, lower; image is up-down, left-right
        img = img[crop_coords[1]:crop_coords[3], crop_coords[0]:crop_coords[2], :]
        #set up the spatial cropped key if one is not passed
        if spatial_cropped_key is None:
            spatial_cropped_key = "spatial_cropped_"+str(buffer)+"_buffer"
        #need to move spatial so it starts at the new crop top left point
        #spatial[:,1] is up-down, spatial[:,0] is left-right
        adata.obsm[spatial_cropped_key] = adata.obsm["spatial"].copy()
        adata.obsm[spatial_cropped_key][:,0] -= crop_coords[0]
        adata.obsm[spatial_cropped_key][:,1] -= crop_coords[1]
        #print off the spatial cropped key just in case
        print("Cropped spatial coordinates key: "+spatial_cropped_key)
    #reshape image to desired microns per pixel
    #get necessary scale factor for the custom mpp
    #multiply dimensions by this to get the shrunken image size
    #multiply .obsm['spatial'] by this to get coordinates matching the image
    scalef = mpp_to_scalef(adata, mpp=mpp)
    #need to reverse dimension order and turn to int for cv2
    dim = (np.array(img.shape[:2])*scalef).astype(int)[::-1]
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    #we have everything we need. store in object
    if store:
        if img_key is None:
            img_key = str(mpp)+"_mpp"
            if crop:
                img_key = img_key+"_"+str(buffer)+"_buffer"
        adata.uns['spatial'][library]['images'][img_key] = img
        #the scale factor needs to be prefaced with "tissue_"
        adata.uns['spatial'][library]['scalefactors']['tissue_'+img_key+"_scalef"] = scalef
        #print off the image key just in case
        print("Image key: "+img_key)
    if save_path is not None:
        #cv2 expects BGR channel order, we're working with RGB
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def scaled_if_image(adata, channel, mpp=1, crop=True, buffer=150, spatial_cropped_key=None, store=True, img_key=None, save_path=None):
    '''
    Create a custom microns per pixel render of the full scale IF image for 
    visualisation and downstream application. Store resulting image and its 
    corresponding size factor in the object. If cropping to just the spatial 
    grid, also store the cropped spatial coordinates. Optionally save to file.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Path to high resolution IF image provided via 
        ``source_image_path`` to ``b2c.read_visium()``.
    channel : ``int``
        The channel of the IF image holding the DAPI capture.
    mpp : ``float``, optional (default: 1)
        Microns per pixel of the desired IF image to create.
    crop : ``bool``, optional (default: ``True``)
        If ``True``, will limit the image to the actual spatial coordinate area, 
        with ``buffer`` added to each dimension.
    buffer : ``int``, optional (default: 150)
        Only used with ``crop=True``. How many extra pixels (in original 
        resolution) to include on each side of the captured spatial grid.
    spatial_cropped_key : ``str`` or ``None``, optional (default: ``None``)
        Only used with ``crop=True``. ``.obsm`` key to store the adjusted 
        spatial coordinates in. If ``None``, defaults to 
        ``"spatial_cropped_X_buffer"``, where ``X`` is the value of ``buffer``.
    store : ``bool``, optional (default: ``True``)
        Whether to store the generated image within the object.
    img_key : ``str`` or ``None``, optional (default: ``None``)
        Only used with ``store=True``. The image key to store the image 
        under in the object. If ``None``, defaults to ``"X_mpp_Y_buffer"``, 
        where ``X`` is the value of ``mpp`` and ``Y`` is the value of 
        ``buffer`` in the instance of ``crop=True``. If no cropping is to 
        be done, defaults to ``"X_mpp"``.
    save_path : ``filepath`` or ``None``, optional (default: ``None``)
        If specified, will save the generated image to this path (e.g. for 
        StarDist use).
    '''
    #identify name of spatial key for subsequent access of fields
    import cv2
    import tifffile as tf
    library = list(adata.uns['spatial'].keys())[0]
    #pull out specified channel from IF tiff via tifffile
    #pretype to float32 for space while working with plots (float16 does not)
    img = tf.imread(adata.uns['spatial'][library]['metadata']['source_image_path'], key=channel).astype(np.float32)
    #assess that the image dimensions match what they're supposed to be
    #if not, inform the user what image they should retrieve and use
    actual_vs_inferred_image_shape(adata, img)
    #this can be dark, apply stardist normalisation to fix
    img = normalize(img)
    #actually cap the values - currently there are sub 0 and above 1 entries
    img[img<0] = 0
    img[img>1] = 1
    #crop image if necessary
    if crop:
        crop_coords = get_crop(adata, basis="spatial", spatial_key="spatial", mpp=None, buffer=buffer)
        #this is already capped at a minimum of 0, so can just subset freely
        #left, upper, right, lower; image is up-down, left-right
        img = img[crop_coords[1]:crop_coords[3], crop_coords[0]:crop_coords[2]]
        #set up the spatial cropped key if one is not passed
        if spatial_cropped_key is None:
            spatial_cropped_key = "spatial_cropped_"+str(buffer)+"_buffer"
        #need to move spatial so it starts at the new crop top left point
        #spatial[:,1] is up-down, spatial[:,0] is left-right
        adata.obsm[spatial_cropped_key] = adata.obsm["spatial"].copy()
        adata.obsm[spatial_cropped_key][:,0] -= crop_coords[0]
        adata.obsm[spatial_cropped_key][:,1] -= crop_coords[1]
        #print off the spatial cropped key just in case
        print("Cropped spatial coordinates key: "+spatial_cropped_key)
    #reshape image to desired microns per pixel
    #get necessary scale factor for the custom mpp
    #multiply dimensions by this to get the shrunken image size
    #multiply .obsm['spatial'] by this to get coordinates matching the image
    scalef = mpp_to_scalef(adata, mpp=mpp)
    #need to reverse dimension order and turn to int for cv2
    dim = (np.array(img.shape[:2])*scalef).astype(int)[::-1]
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    #we have everything we need. store in object
    if store:
        if img_key is None:
            img_key = str(mpp)+"_mpp"
            if crop:
                img_key = img_key+"_"+str(buffer)+"_buffer"
        adata.uns['spatial'][library]['images'][img_key] = img
        #the scale factor needs to be prefaced with "tissue_"
        adata.uns['spatial'][library]['scalefactors']['tissue_'+img_key+"_scalef"] = scalef
        #print off the image key just in case
        print("Image key: "+img_key)
    if save_path is not None:
        #cv2 expects BGR channel order, we have a greyscale image
        #oh also we should make it a uint8 as otherwise stuff won't work
        cv2.imwrite(save_path, cv2.cvtColor((255*img).astype(np.uint8), cv2.COLOR_GRAY2BGR))

def insert_labels(adata, labels_npz_path, basis="spatial", spatial_key="spatial", mpp=None, labels_key="labels"):
    '''
    Load StarDist segmentation results and store them in the object. Labels 
    will be stored as integers, with 0 being unassigned to an object.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object.
    labels_npz_path : ``filepath``
        Path to sparse labels generated by ``b2c.stardist()``.
    basis : ``str``, optional (default: ``"spatial"``)
        Whether the image represents ``"spatial"`` or ``"array"`` coordinates. 
        The former is the source morphology image, the latter is a GEX-based grid 
        representation.
    spatial_key : ``str``, optional (default: ``"spatial"``)
        Only used with ``basis="spatial"``. Needs to be present in ``.obsm``. 
        Rounded coordinates will be used to represent each bin when retrieving 
        labels.
    mpp : ``float`` or ``None``, optional (default: ``None``)
        The mpp value that was used to generate the segmented image. Mandatory 
        for GEX (``basis="array"``), if not provided with morphology 
        (``basis="spatial"``) will assume full scale image.
    labels_key : ``str``, optional (default: ``"labels"``)
        ``.obs`` key to store the labels under.
    '''
    #load sparse segmentation results
    labels_sparse = scipy.sparse.load_npz(labels_npz_path)
    #may as well stash that path in .uns['bin2cell'] since we have it
    if "bin2cell" not in adata.uns:
        adata.uns["bin2cell"] = {}
    if "labels_npz_paths" not in adata.uns["bin2cell"]:
        adata.uns["bin2cell"]["labels_npz_paths"] = {}
    #store as absolute path if it's relative
    if labels_npz_path[0] != "/":
        npz_prefix = os.getcwd()+"/"
    else:
        npz_prefix = ""
    adata.uns["bin2cell"]["labels_npz_paths"][labels_key] = npz_prefix + labels_npz_path
    #get the appropriate coordinates, be they spatial or array, at appropriate mpp
    coords = get_mpp_coords(adata, basis=basis, spatial_key=spatial_key, mpp=mpp)
    #there is a possibility that some coordinates will fall outside labels_sparse
    #start by pregenerating an obs column of all zeroes so all bins are covered
    adata.obs[labels_key] = 0
    #can now construct a mask defining which coordinates fall within range
    #apply the mask to the coords and the obs to just go for the relevant bins
    mask = ((coords[:,0] >= 0) & 
            (coords[:,0] < labels_sparse.shape[0]) & 
            (coords[:,1] >= 0) & 
            (coords[:,1] < labels_sparse.shape[1])
           )
    #pull out the cell labels for the coordinates, can just index the sparse matrix with them
    #insert into bin object, need to turn it into a 1d numpy array from a 1d numpy matrix first
    label_vals = labels_sparse[coords[mask,0], coords[mask,1]]
    if scipy.sparse.issparse(label_vals):
        label_vals = np.asarray(label_vals.todense()).flatten()
    else:
        label_vals = np.asarray(label_vals).flatten()
    adata.obs.loc[mask, labels_key] = label_vals

def expand_labels(adata, labels_key="labels", expanded_labels_key="labels_expanded", algorithm="max_bin_distance", max_bin_distance=2, volume_ratio=4, k=4, subset_pca=True):
    '''
    Expand StarDist segmentation results to bins a maximum distance away in 
    the array coordinates. In the event of multiple equidistant bins with 
    different labels, ties are broken by choosing the closest bin in a PCA 
    representation of gene expression. The resulting labels will be integers, 
    with 0 being unassigned to an object.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Raw or destriped counts.
    labels_key : ``str``, optional (default: ``"labels"``)
        ``.obs`` key holding the labels to be expanded. Integers, with 0 being 
        unassigned to an object.
    expanded_labels_key : ``str``, optional (default: ``"labels_expanded"``)
        ``.obs`` key to store the expanded labels under.
    algorithm : ``str``, optional (default: ``"max_bin_distance"``)
        Toggle between ``max_bin_distance`` or ``volume_ratio`` based label 
        expansion.
    max_bin_distance : ``int`` or ``None``, optional (default: 2)
        Maximum number of bins to expand the nuclear labels by.
    volume_ratio : ``float``, optional (default: 4)
        A per-label expansion distance will be proposed as 
        ``ceil((volume_ratio**(1/3)-1) * sqrt(n_bins/pi))``, where 
        ``n_bins`` is the number of bins for the corresponding pre-expansion 
        label. Default based on cell line 
        `data <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8893647/>`_
    k : ``int``, optional (default: 4)
        Number of assigned spatial coordinate bins to find as potential nearest 
        neighbours for each unassigned bin.
    subset_pca : ``bool``, optional (default: ``True``)
        If ``True``, will obtain the PCA representation of just the bins 
        involved in the tie breaks rather than the full bin space. Results in 
        a slightly different embedding at a lower resource footprint.
    '''
    #this is where the labels will go
    adata.obs[expanded_labels_key] = adata.obs[labels_key].values.copy()
    #get out our array grid, and preexisting labels
    coords = adata.obs[["array_row","array_col"]].values
    labels = adata.obs[labels_key].values
    #we'll be splitting the space in two - the bins with labels, and those without
    object_mask = (labels != 0)
    #get their indices in cell space
    full_reference_inds = np.arange(adata.shape[0])[object_mask]
    full_query_inds = np.arange(adata.shape[0])[~object_mask]
    #for each unassigned bin, we'll find its k nearest neighbours in the assigned space
    #build a reference using the assigned bins' coordinates
    ckd = scipy.spatial.cKDTree(coords[object_mask, :])
    #query it using the unassigned bins' coordinates
    dists, hits = ckd.query(x=coords[~object_mask,:], k=k, workers=-1)
    #convert the identified indices back to the full cell space
    hits = full_reference_inds[hits]
    #get the label calls for each of the hits
    calls = labels[hits]
    #get the area (bin count) of each object
    label_values, label_counts = np.unique(labels, return_counts=True)
    #this is how the algorithm was toggled early on
    #switched to an argument to avoid potential future spaghetti
    if max_bin_distance is None:
        raise ValueError("Use ``algorithm`` to toggle between algorithms")
    if algorithm == "volume_ratio":
        #compute the object's sphere's radius as sqrt(nbin/pi)
        #scale to radius of cell by multiplying by volume_ratio^(1/3)
        #and subtract away the original radius to account for presence of nucleus
        #do a ceiling to compensate for possible reduction of area in slice
        label_distances = np.ceil((volume_ratio**(1/3)-1) * np.sqrt(label_counts/np.pi))
        #get an array where you can index on object and get the distance
        #needs +1 as the max value of label_values is actually present in the data
        label_distance_array = np.zeros((np.max(label_values)+1,))
        label_distance_array[label_values] = label_distances
    elif algorithm == "max_bin_distance":
        #just use the provided value
        label_distance_array = np.ones((np.max(label_values)+1,)) * max_bin_distance
    else:
        raise ValueError("``algorithm`` must be ``'max_bin_distance'`` or ``'volume_ratio'``")
    #construct a matching dimensionality array of max distance allowed per call
    max_call_distance = label_distance_array[calls]
    #mask bins too far away from call with arbitrary high value
    dist_mask = 1000
    dists[dists > max_call_distance] = dist_mask
    #evaluate the minima in each row. start by getting said minima
    min_per_bin = np.min(dists, axis=1)[:,None]
    #now get positions in each row that have the minimum (and aren't the mask)
    is_hit = (dists == min_per_bin) & (min_per_bin < dist_mask)
    #case one - we have a solitary hit of the minimum
    clear_mask = (np.sum(is_hit, axis=1) == 1)
    #get out the indices of the bins
    clear_query_inds = full_query_inds[clear_mask]
    #np.argmin(axis=1) finds the column of the minimum per row
    #subsequently retrieve the matching hit from calls
    clear_query_labels = calls[clear_mask, np.argmin(dists[clear_mask, :], axis=1)]
    #insert calls into object
    adata.obs.loc[adata.obs_names[clear_query_inds], expanded_labels_key] = clear_query_labels
    #case two - 2+ assigned bins are equidistant
    ambiguous_mask = (np.sum(is_hit, axis=1) > 1)
    if np.sum(ambiguous_mask) > 0:
        #get their indices in the original cell space
        ambiguous_query_inds = full_query_inds[ambiguous_mask]
        if subset_pca:
            #in preparation of PCA, get a master list of all the bins to PCA
            #we've got two sets - the query bins, and their k hits
            #the hits needs to be .flatten()ed after masking to become 1d again
            #np.unique sorts in an ascending fashion, which is convenient
            smol = np.unique(np.concatenate([hits[ambiguous_mask,:].flatten(), ambiguous_query_inds]))
            #prepare a PCA as a representation of the GEX space for solving ties
            #can just run straight on an array to get a PCA matrix back. convenient!
            #keep the object's X raw for subsequent cell creation
            pca_smol = sc.pp.pca(np.log1p(adata.X[smol, :]))
            #mock up a "full-scale" PCA matrix to not have to worry about different indices
            pca = np.zeros((adata.shape[0], pca_smol.shape[1]))
            pca[smol, :] = pca_smol
        else:
            #just run a full space PCA
            pca = sc.pp.pca(np.log1p(adata.X))
        #compute the distances between the expression profiles of the undecided bin and the neighbours
        #np.linalg.norm is the fastest way to get euclidean, subtract two point sets beforehand
        #pca[hits[ambiguous_mask, :]] is bins by k by num_pcs
        #pca[ambiguous_query_inds, :] is bins by num_pcs
        #add the [:, None, :] and it's bins by 1 by num_pcs, and subtracts as you'd hope
        eucl_input = pca[hits[ambiguous_mask, :]] - pca[ambiguous_query_inds, :][:, None, :]
        #can just do this along axis=2 and get all the distances at once
        eucl_dists = np.linalg.norm(eucl_input, axis=2)
        #mask ineligible bins with arbitrary high value
        eucl_mask = 1000
        eucl_dists[~is_hit[ambiguous_mask, :]] = eucl_mask
        #define calls based on euclidean minimum
        #same argmin/mask logic as with clear before
        ambiguous_query_labels = calls[ambiguous_mask, np.argmin(eucl_dists, axis=1)]
        #insert calls into object
        adata.obs.loc[adata.obs_names[ambiguous_query_inds], expanded_labels_key] = ambiguous_query_labels

def salvage_secondary_labels(adata, primary_label="labels_he_expanded", secondary_label="labels_gex", labels_key="labels_joint"):
    '''
    Create a joint ``labels_key`` that takes the ``primary_label`` and fills in 
    unassigned bins based on calls from ``secondary_label``. Only objects that do not 
    overlap with any bins called as part of ``primary_label`` are transferred over.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Needs ``primary_key`` and ``secodary_key`` in ``.obs``.
    primary_label : ``str``, optional (default: ``"labels_he_expanded"``)
        ``.obs`` key holding the main labels. Integers, with 0 being unassigned to an 
        object.
    secondary_label : ``str``, optional (default: ``"labels_gex"``)
        ``.obs`` key holding the labels to be inserted into unassigned bins. Integers, 
        with 0 being unassigned to an object.
    labels_key : ``str``, optional (default: ``"labels_joint"``)
        ``.obs`` key to store the combined label information into. Will also add a 
        second column with ``"_source"`` appended to differentiate whether the bin was 
        tagged from the primary or secondary label.
    '''
    #these are the bins that have the primary label assigned
    primary = adata.obs.loc[adata.obs[primary_label] > 0, :]
    #these are the bins that lack the primary label, but have the secondary label
    secondary = adata.obs.loc[adata.obs[primary_label] == 0, :]
    secondary = secondary.loc[secondary[secondary_label] > 0, :]
    #kick out any secondary labels that appear in primary-labelled bins
    #we are just interested in ones that are unique to bins without primary labelling
    secondary_to_take = np.array(list(set(secondary[secondary_label]).difference(set(primary[secondary_label]))))
    #both of these labels are integers, starting from 1
    #offset the new secondary labels by however much the maximum primary label is
    offset = np.max(adata.obs[primary_label])
    #use the primary labels as a basis
    adata.obs[labels_key] = adata.obs[primary_label].copy()
    #flag any bins that are assigned to our secondary labels of interest
    mask = np.isin(adata.obs[secondary_label], secondary_to_take)
    adata.obs.loc[mask, labels_key] = adata.obs.loc[mask, secondary_label] + offset
    #store information on origin of call
    adata.obs[labels_key+"_source"] = "none"
    adata.obs.loc[adata.obs[primary_label]>0, labels_key+"_source"] = "primary"
    adata.obs.loc[mask, labels_key+"_source"] = "secondary"
    #stash secondary label offset as that seems potentially useful
    if "bin2cell" not in adata.uns:
        adata.uns["bin2cell"] = {}
    if "secondary_label_offset" not in adata.uns["bin2cell"]:
        adata.uns["bin2cell"]["secondary_label_offset"] = {}
    adata.uns["bin2cell"]["secondary_label_offset"][labels_key] = offset
    #notify of how much was salvaged
    print("Salvaged "+str(len(secondary_to_take))+" secondary labels")

def _copy_spatial_uns_light(spatial_uns):
    """
    Copy the spatial container without deep-copying large image arrays.

    We only mutate ``scalefactors['spot_diameter_fullres']`` in ``bin_to_cell``,
    so a shallow library-level copy plus a dedicated scalefactor dict copy is enough.
    """
    if not isinstance(spatial_uns, dict):
        return deepcopy(spatial_uns)

    copied = {}
    for lib_key, lib_value in spatial_uns.items():
        if isinstance(lib_value, dict):
            lib_copy = dict(lib_value)
            scalefactors = lib_value.get("scalefactors")
            if isinstance(scalefactors, dict):
                lib_copy["scalefactors"] = dict(scalefactors)
            copied[lib_key] = lib_copy
        else:
            copied[lib_key] = lib_value
    return copied


def bin_to_cell(adata, labels_key="labels_expanded", 
                spatial_keys=["spatial"], diameter_scale_factor=None, show_progress=False):
    '''
    Collapse all bins for a given nonzero ``labels_key`` into a single cell. 
    Gene expression added up, array coordinates and ``spatial_keys`` averaged out. 
    ``"spot_diameter_fullres"`` in the scale factors multiplied by 
    ``diameter_scale_factor`` to reflect increased unit size. Returns cell level AnnData, 
    including ``.obs["bin_count"]`` reporting how many bins went into creating the cell.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Raw or destriped counts. Needs ``labels_key`` in ``.obs`` 
        and ``spatial_keys`` in ``.obsm``.
    labels_key : ``str``, optional (default: ``"labels_expanded"``)
        Which ``.obs`` key to use for grouping 2um bins into cells. Integers, with 0 being 
        unassigned to an object. If an extra ``"_source"`` column is detected as a result 
        of ``b2c.salvage_secondary_labels()`` calling, its info will be propagated per 
        label.
    spatial_keys : list of ``str``, optional (default: ``["spatial"]``)
        Which ``.obsm`` keys to average out across all bins falling into a cell to get a 
        cell's respective spatial coordinates.
    diameter_scale_factor : ``float`` or ``None``, optional (default: ``None``)
        The object's ``"spot_diameter_fullres"`` will be multiplied by this much to reflect 
        the change in unit per observation. If ``None``, will default to the square root of 
        the mean of the per-cell bin counts.
    show_progress : ``bool``, optional (default: ``False``)
        Whether to display a lightweight progress bar for major aggregation stages.
    '''
    pbar = None
    if show_progress:
        try:
            from tqdm.auto import tqdm

            pbar = tqdm(total=7, desc="bin_to_cell", leave=False)
        except Exception:
            pbar = None

    def _tick():
        if pbar is not None:
            pbar.update(1)

    try:
        # Fast path: avoid pandas.get_dummies for large tables.
        label_series = adata.obs[labels_key]
        if pd.api.types.is_numeric_dtype(label_series.dtype):
            labels_all = np.asarray(label_series.to_numpy(), dtype=np.float64)
            labels_all = np.nan_to_num(labels_all, nan=0.0).astype(np.int64, copy=False)
        else:
            labels_all = pd.to_numeric(label_series, errors="coerce").fillna(0).astype(np.int64).to_numpy()
        valid_idx = np.flatnonzero(labels_all != 0)
        if valid_idx.size == 0:
            raise ValueError(f"No non-zero labels found in `adata.obs['{labels_key}']`.")
        _tick()

        labels = labels_all[valid_idx]
        unique_labels, inverse = np.unique(labels, return_inverse=True)  # sorted ascending, stable naming
        n_cells = unique_labels.shape[0]
        n_bins = labels.shape[0]
        rows = inverse.astype(np.int64, copy=False)
        _tick()

        # Build sparse cell_to_bin directly: rows=cells, cols=valid bins
        cols = np.arange(n_bins, dtype=np.int64)
        data = np.ones(n_bins, dtype=np.int8)
        cell_to_bin = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n_cells, n_bins))
        _tick()

        # Aggregate expression: cell-bin * bin-gene -> cell-gene
        X_bins = adata.X[valid_idx]
        if scipy.sparse.issparse(X_bins):
            X_bins = X_bins.tocsr()
        X = cell_to_bin.dot(X_bins)
        if scipy.sparse.issparse(X):
            X = X.tocsr()
        _tick()

        # Format cell IDs to match SpaceRanger v4 convention: cellid_XXXXXXXXX-1
        cell_names = [f"cellid_{int(lbl):09d}-1" for lbl in unique_labels]
        cell_adata = ad.AnnData(X, var=adata.var)
        cell_adata.obs_names = cell_names
        cell_adata.obs["object_id"] = unique_labels.astype(np.int64)

        # Avoid deep-copying large image arrays in uns['spatial'].
        cell_adata.uns["spatial"] = _copy_spatial_uns_light(adata.uns["spatial"])

        # Per-cell bin counts and weighted coordinate means
        bin_count = np.bincount(rows, minlength=n_cells).astype(np.int64, copy=False)
        cell_adata.obs["bin_count"] = bin_count
        inv_bin_count = np.divide(
            1.0,
            bin_count.astype(np.float64),
            out=np.zeros_like(bin_count, dtype=np.float64),
            where=bin_count > 0,
        )
        _tick()

        arr_row = np.asarray(adata.obs["array_row"].to_numpy())[valid_idx]
        arr_col = np.asarray(adata.obs["array_col"].to_numpy())[valid_idx]
        cell_adata.obs["array_row"] = np.bincount(rows, weights=arr_row, minlength=n_cells) * inv_bin_count
        cell_adata.obs["array_col"] = np.bincount(rows, weights=arr_col, minlength=n_cells) * inv_bin_count

        # Generate averaged spatial coordinate systems
        if type(spatial_keys) is not list:
            spatial_keys = [spatial_keys]
        for spatial_key in spatial_keys:
            coords = np.asarray(adata.obsm[spatial_key])[valid_idx]
            summed = np.zeros((n_cells, coords.shape[1]), dtype=np.float64)
            np.add.at(summed, rows, coords)
            cell_adata.obsm[spatial_key] = summed * inv_bin_count[:, None]
        _tick()

        # Rescale rendered diameter
        if diameter_scale_factor is None:
            diameter_scale_factor = np.sqrt(np.mean(bin_count))
        library = list(adata.uns["spatial"].keys())[0]
        cell_adata.uns["spatial"][library]["scalefactors"]["spot_diameter_fullres"] *= diameter_scale_factor

        # Propagate *_source mapping per object label
        source_key = labels_key + "_source"
        if source_key in adata.obs.columns:
            source_vals = adata.obs[source_key].astype(str).to_numpy()[valid_idx]
            _, first_indices = np.unique(rows, return_index=True)
            cell_adata.obs[source_key] = source_vals[first_indices]
        _tick()
    finally:
        if pbar is not None:
            pbar.close()
    
    return cell_adata


def _add_segmentation_to_adata(cell_adata, original_adata, labels_key, segmentation_key):
    """
    Generate segmentation mask from cell labels and add to cell_adata for boundary visualization.
    
    Parameters
    ----------
    cell_adata : AnnData
        The cell-level AnnData object to add segmentation data to.
    original_adata : AnnData 
        The original bin-level AnnData object containing spatial coordinates and labels.
    labels_key : str
        The key in obs containing cell labels.
    segmentation_key : str
        Key name for storing the segmentation mask.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    
    # Get library key for accessing spatial data structures
    library = list(cell_adata.uns['spatial'].keys())[0]
    
    # Get spatial dimensions from the original data
    # We need to reconstruct a segmentation mask from the bin labels and spatial coordinates
    spatial_coords = original_adata.obsm['spatial']
    labels = original_adata.obs[labels_key].values
    
    # Determine the image dimensions based on spatial coordinates
    # Add some padding to ensure all cells are captured
    min_coords = np.floor(spatial_coords.min(axis=0)).astype(int)
    max_coords = np.ceil(spatial_coords.max(axis=0)).astype(int) + 1
    
    # Create segmentation mask
    height = max_coords[1] - min_coords[1]
    width = max_coords[0] - min_coords[0]
    segmentation_mask = np.zeros((height, width), dtype=np.int32)
    
    # Map each spatial location to its label
    # Adjust coordinates to start from (0,0)
    adjusted_coords = (spatial_coords - min_coords).astype(int)
    
    # Fill the segmentation mask
    for i, (x, y) in enumerate(adjusted_coords):
        if 0 <= y < height and 0 <= x < width:
            segmentation_mask[y, x] = labels[i]
    
    # Store the segmentation mask in the spatial data structure
    cell_adata.uns['spatial'][library]['images'][segmentation_key] = segmentation_mask
    
    # Add cell_id column to obs for compatibility with spatial plotting functions
    cell_adata.obs['cell_id'] = cell_adata.obs['object_id']
    
    # Store metadata about the segmentation
    if 'segmentation_metadata' not in cell_adata.uns['spatial'][library]:
        cell_adata.uns['spatial'][library]['segmentation_metadata'] = {}
    
    cell_adata.uns['spatial'][library]['segmentation_metadata'][segmentation_key] = {
        'original_coords_min': min_coords,
        'original_coords_max': max_coords,
        'mask_shape': (height, width),
        'labels_key': labels_key
    }


def plot_cell_boundaries(cell_adata, 
                        color=None, 
                        segmentation_key="segmentation",
                        seg_cell_id="cell_id",
                        seg_outline=True,
                        seg_contourpx=None,
                        library_id=None,
                        auto_detect=True,
                        **kwargs):
    """
    Plot cell boundaries using the stored segmentation mask.
    
    This function provides a convenient interface to visualize cell boundaries
    similar to omicverse.pl.spatial() with segmentation support.
    
    Parameters
    ----------
    cell_adata : AnnData
        Cell-level AnnData object with segmentation data stored by bin_to_cell().
    color : str or None
        Gene or observation to color the cells by. If None, shows segmentation only.
    segmentation_key : str, optional (default: "segmentation")
        Key name for the segmentation mask in spatial images.
    seg_cell_id : str, optional (default: "cell_id") 
        Column name in cell_adata.obs containing cell IDs for segmentation.
    seg_outline : bool, optional (default: True)
        Whether to show cell boundaries as outlines.
    seg_contourpx : int or None, optional
        Width of boundary contours in pixels. If specified, boundaries will be eroded.
    library_id : str or None, optional
        Library ID to use. If None, uses the first available library.
    auto_detect : bool, optional (default: True)
        Whether to automatically detect and fix parameter mismatches.
    **kwargs
        Additional arguments passed to plotting function.
        
    Returns
    -------
    matplotlib axis or None
        The plotting axis if successful.
    """
    try:
        # Import the spatial plotting function
        from omicverse.pl._spatial import spatial
        
        # Get library_id if not specified
        if library_id is None:
            library_id = list(cell_adata.uns['spatial'].keys())[0]
        
        # Check if segmentation data exists
        if (segmentation_key not in cell_adata.uns['spatial'][library_id]['images']):
            raise ValueError(f"Segmentation key '{segmentation_key}' not found. "
                           "Make sure to call bin_to_cell() with store_segmentation=True")
        
        # Check if cell_id column exists
        if seg_cell_id not in cell_adata.obs:
            raise ValueError(f"Cell ID column '{seg_cell_id}' not found in cell_adata.obs")
        
        # Build plot parameters with auto-detection
        plot_params = {
            'color': color,
            'seg': True,
            'seg_key': segmentation_key,
            'seg_cell_id': seg_cell_id,
            'seg_outline': seg_outline,
            'seg_contourpx': seg_contourpx
        }
        
        # Auto-detect library parameters if enabled
        if auto_detect:
            # Only specify library_id if there are multiple libraries
            if len(list(cell_adata.uns['spatial'].keys())) == 1:
                # Single library case - let spatial function auto-detect
                pass
            else:
                plot_params['library_id'] = library_id
                
                # Try to find appropriate library_key
                potential_keys = ['fov', 'library', 'library_id', 'sample', 'batch']
                for key in potential_keys:
                    if (key in cell_adata.obs and 
                        library_id in cell_adata.obs[key].astype(str).values):
                        plot_params['library_key'] = key
                        break
        else:
            plot_params['library_id'] = library_id
        
        # Add user-specified parameters
        plot_params.update(kwargs)
        
        # Use omicverse spatial plotting with segmentation
        ax = spatial(cell_adata, **plot_params)
        return ax
        
    except ImportError:
        print("Warning: omicverse.pl.spatial not available. Using basic matplotlib visualization.")
        return _plot_boundaries_matplotlib(cell_adata, color, segmentation_key, library_id, **kwargs)


def _plot_boundaries_matplotlib(cell_adata, color, segmentation_key, library_id, **kwargs):
    """
    Basic matplotlib-based boundary visualization as fallback.
    """
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    import numpy as np
    
    # Get segmentation mask
    seg_mask = cell_adata.uns['spatial'][library_id]['images'][segmentation_key]
    
    # Create figure
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 10)))
    
    # Find boundaries
    boundaries = find_boundaries(seg_mask)
    
    # Create boundary visualization
    if color is None:
        # Just show boundaries
        ax.imshow(boundaries, cmap='gray')
        ax.set_title('Cell Boundaries')
    else:
        # Show colored cells with boundaries
        # This is a simplified version - full implementation would need proper color mapping
        colored_mask = np.zeros_like(seg_mask, dtype=float)
        if color in cell_adata.obs:
            for i, cell_id in enumerate(cell_adata.obs['cell_id']):
                mask = seg_mask == cell_id
                colored_mask[mask] = cell_adata.obs[color].iloc[i]
        
        im = ax.imshow(colored_mask, cmap=kwargs.get('cmap', 'viridis'))
        
        # Add boundaries
        boundary_mask = boundaries[..., np.newaxis] * np.array([1, 1, 1])  # White boundaries
        ax.imshow(boundary_mask, alpha=0.5)
        
        plt.colorbar(im, ax=ax, label=color)
        ax.set_title(f'Cells colored by {color}')
    
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    
    return ax

# Backward-compatible aliases
stardist = cellseg
view_stardist_labels = view_cellseg_labels

