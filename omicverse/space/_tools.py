import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import scanpy as sc
from .._registry import register_function
from .._optional import build_optional_dependency_error

__all__ = [
    "subset_window",
    "crop_space_visium",
    "rotate_space_visium",
    "find_image_offset_phase_correlation_array_input",
    "find_image_offset_phase_correlation_torch",
    "map_spatial_auto",
    "map_spatial_manual",
    "read_visium_10x",
    "visium_10x_hd_cellpose_he",
    "visium_10x_hd_cellpose_expand",
    "visium_10x_hd_cellpose_gex",
    "salvage_secondary_labels",
    "bin2cell",
    "sync_visium_hd_seg_geometries",
    "write_visium_hd_cellseg",
]
"""Public surface of this module.

``ov.space`` does ``from ._tools import *``. Without this list that star
import also republished ``np``, ``os``, ``plt``, ``sc``, ``mpl`` and four
matplotlib symbols as if they were part of the spatial API -- nine of the
sixty-four public names in ``ov.space`` were other libraries' imports.
"""


def _resolve_spatial_library(adata, library_id=None, library_key=None):
    """Resolve one image library and the observations that belong to it."""
    spatial = adata.uns.get("spatial")
    if not isinstance(spatial, dict) or not spatial:
        raise KeyError("adata.uns['spatial'] does not contain any spatial libraries.")
    available = list(spatial)
    if library_id is None:
        if len(available) != 1:
            raise ValueError(
                "`library_id` must be specified when multiple spatial libraries are present."
            )
        library_id = available[0]
    if library_id not in spatial:
        raise KeyError(f"library_id {library_id!r} was not found in adata.uns['spatial'].")

    if library_key is None:
        if len(available) > 1:
            raise ValueError(
                "A multi-library object also requires `library_key` so image "
                "operations cannot transform observations from other libraries."
            )
        mask = np.ones(adata.n_obs, dtype=bool)
    else:
        if library_key not in adata.obs:
            raise KeyError(f"library_key {library_key!r} was not found in adata.obs.")
        labels = adata.obs[library_key]
        if labels.isna().any():
            raise ValueError(f"adata.obs[{library_key!r}] contains missing library labels.")
        mask = labels.astype(str).to_numpy() == str(library_id)
        if not mask.any():
            raise ValueError(
                f"No observations in adata.obs[{library_key!r}] match library_id "
                f"{library_id!r}."
            )
    return library_id, mask


def _record_removed_stale_images(spatial_info, operation, image_keys):
    """Record image keys removed because their coordinate transform was unknown."""
    if not image_keys:
        return
    metadata = spatial_info.setdefault("metadata", {})
    records = metadata.setdefault("omicverse_removed_stale_images", {})
    record_key = f"{len(records):04d}_{operation}"
    records[record_key] = {
        "operation": str(operation),
        "image_keys": np.asarray([str(key) for key in image_keys], dtype=str),
        "reason": "missing matching tissue_<image_key>_scalef",
    }


def _crop_scaled_image(image, x0, y0, x1, y1):
    """Crop an image at possibly fractional pixel bounds without shifting its origin."""
    image = np.asarray(image)
    width = int(np.ceil(x1 - x0))
    height = int(np.ceil(y1 - y0))
    if width <= 0 or height <= 0:
        return image[:0, :0].copy()

    rounded = np.rint([x0, y0, x1, y1]).astype(int)
    if np.allclose([x0, y0, x1, y1], rounded, rtol=0, atol=1e-10):
        ix0, iy0, ix1, iy1 = rounded
        return image[iy0:iy1, ix0:ix1].copy()

    # A plain floor/ceil slice changes the image origin by up to one pixel when
    # the requested full-resolution crop maps to fractional pixels at another
    # resolution. Sample that resolution at the exact origin instead. This
    # keeps ``new_spatial * tissue_<resolution>_scalef`` aligned to the image.
    from scipy.ndimage import affine_transform

    trailing_shape = image.shape[2:]
    flat = image.reshape(image.shape[0], image.shape[1], -1)
    sampled = np.empty((height, width, flat.shape[2]), dtype=image.dtype)
    order = 0 if np.issubdtype(image.dtype, np.bool_) else 1
    for channel in range(flat.shape[2]):
        channel_image = flat[:, :, channel]
        if channel_image.dtype == np.float16:
            # scipy.ndimage does not accept float16 input; calculate in float32
            # and cast back through the preallocated output.
            channel_image = channel_image.astype(np.float32)
        sampled[:, :, channel] = affine_transform(
            channel_image,
            matrix=np.eye(2),
            offset=(float(y0), float(x0)),
            output_shape=(height, width),
            order=order,
            mode="nearest",
            prefilter=False,
        )
    return sampled.reshape((height, width) + trailing_shape)

@register_function(
    aliases=["subset_window", "spatial_window", "spatial_subset",
             "crop spatial subset", "空间窗口子集"],
    category="space",
    description=(
        "Subset an AnnData to cells whose `obsm[basis]` coordinates fall inside "
        "a rectangular (xlim, ylim) window. Coordinate-system agnostic: works on "
        "microns (Xenium / Atera), full-res pixels (Visium HD), or any other 2-D "
        "embedding. Preserves `uns`."
    ),
    examples=[
        "bdata = ov.space.subset_window(adata, xlim=(1000, 2000), ylim=(500, 1500))",
        "# Window centred on the sample median:",
        "import numpy as np",
        "x0, y0 = np.median(adata.obsm['spatial'], axis=0)",
        "bdata = ov.space.subset_window(adata,",
        "    xlim=(x0 - 500, x0 + 500),",
        "    ylim=(y0 - 500, y0 + 500))",
    ],
    related=["space.crop_space_visium"],
)

def subset_window(adata, xlim, ylim, basis='spatial'):
    """Subset an AnnData by a rectangular spatial window.

    Cells whose ``adata.obsm[basis]`` coordinates fall inside ``(xlim, ylim)``
    are kept; the rest are dropped. The interval is half-open
    (``xlim[0] <= x < xlim[1]``), matching numpy slicing semantics.

    Coordinates are interpreted in whatever unit the basis stores — microns
    for Xenium / Atera, full-res pixels for Visium HD. Use the same window
    you intend to pass to ``ov.pl.spatial(crop_coord=...)`` /
    ``ov.pl.spatialseg(crop_coord=...)`` so the polygon set and the rendered
    background stay aligned.

    Parameters
    ----------
    adata : AnnData
    xlim : tuple of float
        ``(x_min, x_max)`` window, half-open on the right.
    ylim : tuple of float
        ``(y_min, y_max)`` window, half-open on the right.
    basis : str
        Key in ``adata.obsm`` providing the 2-D coordinates. Default ``'spatial'``.

    Returns
    -------
    AnnData
        Copy of ``adata`` restricted to the window. ``uns`` is preserved.
    """
    coords = adata.obsm[basis]
    x, y = coords[:, 0], coords[:, 1]
    mask = (x >= xlim[0]) & (x < xlim[1]) & (y >= ylim[0]) & (y < ylim[1])
    return adata[mask].copy()


@register_function(
    aliases=["裁剪空间数据", "crop_space_visium", "crop_visium", "空间数据裁剪", "Visium裁剪"],
    category="space",
    description="Crop Visium spatial transcriptomics data to focus on a region of interest",
    examples=[
        "adata_cropped = ov.space.crop_space_visium(",
        "    adata, crop_loc=(500, 500), crop_area=(1000, 1000),",
        "    library_id='V1_Human_Brain', scale=1.0, coordinate_order='xy')",
    ],
    related=["space.rotate_space_visium", "space.map_spatial_auto"],
)
def crop_space_visium(adata, crop_loc, crop_area,
                     library_id, scale, spatial_key='spatial', res='hires',
                     coordinate_order=None, library_key=None):
    """
    Crop Visium spatial data to a specific region of interest.
    
    This function allows cropping of Visium spatial transcriptomics data to focus on
    a specific region while maintaining proper scaling and coordinate systems.

    Arguments:
        adata: AnnData
            Annotated data matrix containing Visium spatial data.
        crop_loc: tuple
            (x, y) coordinates for the top-left corner of the crop region.
        crop_area: tuple
            (width, height) of the cropping area in spatial coordinates.
        library_id: str
            Library ID for the spatial data in adata.uns['spatial'].
        scale: float
            Scale factor for the cropping operation.
        spatial_key: str, optional (default='spatial')
            Key in adata.obsm containing spatial coordinates.
        res: str, optional (default='hires')
            Image resolution to use ('hires' or 'lowres').
        coordinate_order: {'xy', 'yx'} or None, default=None
            Interpretation of ``crop_loc`` and ``crop_area``. ``'xy'`` means
            ``(x, y)`` and ``(width, height)``. ``'yx'`` preserves the legacy
            Squidpy-style ``(y, x)`` / ``(height, width)`` convention. ``None``
            currently selects legacy ``'yx'`` with a deprecation warning;
            specify the order explicitly because a future major release will
            default to ``'xy'``.
        library_key: str or None
            Observation column identifying library membership. Required for a
            multi-library AnnData.

    Returns:
        AnnData
            Cropped AnnData object containing only spots within the specified region.
            The spatial coordinates and image are adjusted accordingly.

    Notes:
        - The function preserves the original coordinate system scaling
        - The cropped image is stored in adata.uns['spatial'][library_id]['images'][res]
        - Coordinates are automatically adjusted to the new cropped region

    Examples:
        >>> import scanpy as sc
        >>> import omicverse as ov
        >>> # Load Visium data
        >>> adata = sc.read_visium(...)
        >>> # Crop a 1000x1000 region starting at (500, 500)
        >>> adata_cropped = ov.space.crop_space_visium(
        ...     adata,
        ...     crop_loc=(500, 500),
        ...     crop_area=(1000, 1000),
        ...     library_id='V1_Human_Brain',
        ...     scale=1.0,
        ...     coordinate_order='xy'
        ... )
    """
    if scale != 1:
        raise NotImplementedError(
            "scale != 1 is not yet supported in the squidpy-free implementation. "
            "Use scale=1."
        )

    library_id, library_mask = _resolve_spatial_library(
        adata,
        library_id=library_id,
        library_key=library_key,
    )
    adata1 = adata.copy()
    spatial_info = adata1.uns['spatial'][library_id]
    images = spatial_info.get("images", {})
    scalefactors = spatial_info.get("scalefactors", {})
    if res not in images:
        raise KeyError(f"Image resolution {res!r} was not found for library {library_id!r}.")
    scale_key = f'tissue_{res}_scalef'
    if scale_key not in scalefactors:
        raise KeyError(f"Scalefactor {scale_key!r} was not found for library {library_id!r}.")
    scalef = float(scalefactors[scale_key])
    full_img = np.asarray(images[res])

    if coordinate_order is None:
        import warnings

        warnings.warn(
            "The implicit crop coordinate order is deprecated and currently "
            "preserves legacy (y, x)/(height, width) behavior. Pass "
            "coordinate_order='xy' or 'yx' explicitly.",
            FutureWarning,
            stacklevel=2,
        )
        coordinate_order = 'yx'
    coordinate_order = str(coordinate_order).lower()
    if coordinate_order == 'xy':
        x0, y0 = map(int, crop_loc)
        w, h = map(int, crop_area)
    elif coordinate_order == 'yx':
        y0, x0 = map(int, crop_loc)
        h, w = map(int, crop_area)
    else:
        raise ValueError("coordinate_order must be 'xy' or 'yx'.")
    if w <= 0 or h <= 0:
        raise ValueError("crop_area width and height must be positive.")

    # Clamp to image bounds
    img_h, img_w = full_img.shape[:2]
    y0 = max(0, min(y0, img_h))
    x0 = max(0, min(x0, img_w))
    y1 = max(0, min(y0 + h, img_h))
    x1 = max(0, min(x0 + w, img_w))

    # Convert the selected-resolution crop to the shared full-resolution
    # coordinate frame, then crop every image resolution with its own scale.
    x0_spatial = x0 / scalef
    y0_spatial = y0 / scalef
    x1_spatial = x1 / scalef
    y1_spatial = y1 / scalef
    cropped_images = {}
    stale_image_keys = []
    for image_key, image in list(images.items()):
        image_scale_key = f"tissue_{image_key}_scalef"
        if image_scale_key not in scalefactors or image is None:
            stale_image_keys.append(image_key)
            continue
        image_scale = float(scalefactors[image_scale_key])
        image_array = np.asarray(image)
        if image_array.ndim < 2:
            stale_image_keys.append(image_key)
            continue
        image_h, image_w = image_array.shape[:2]
        image_x0 = max(0.0, min(x0_spatial * image_scale, float(image_w)))
        image_y0 = max(0.0, min(y0_spatial * image_scale, float(image_h)))
        image_x1 = max(0.0, min(x1_spatial * image_scale, float(image_w)))
        image_y1 = max(0.0, min(y1_spatial * image_scale, float(image_h)))
        cropped_images[image_key] = _crop_scaled_image(
            image_array,
            image_x0,
            image_y0,
            image_x1,
            image_y1,
        )

    # Scale spatial coords to hires pixel coords
    # Visium obsm['spatial'] columns are (x=col, y=row)
    pixel_coords = adata1.obsm[spatial_key] * scalef

    # Filter spots within the crop region [x0, x1) x [y0, y1)
    mask = library_mask & (
        (pixel_coords[:, 0] >= x0) & (pixel_coords[:, 0] < x1) &
        (pixel_coords[:, 1] >= y0) & (pixel_coords[:, 1] < y1)
    )
    adata_crop = adata1[mask].copy()

    # Update image
    adata_crop.uns["spatial"][library_id]["images"] = cropped_images
    _record_removed_stale_images(
        adata_crop.uns["spatial"][library_id],
        "crop",
        stale_image_keys,
    )
    adata_crop.uns["spatial"] = {
        library_id: adata_crop.uns["spatial"][library_id]
    }

    # Adjust spatial coordinates relative to the cropped region
    adata_crop.obsm[spatial_key] = np.column_stack([
        adata_crop.obsm[spatial_key][:, 0] - x0_spatial,
        adata_crop.obsm[spatial_key][:, 1] - y0_spatial,
    ])

    return adata_crop

@register_function(
    aliases=["旋转空间数据", "rotate_space_visium", "rotate_visium", "空间数据旋转", "Visium旋转"],
    category="space",
    description="Rotate Visium spatial data image and coordinates by specified angle",
    examples=[
        "# Basic rotation",
        "adata_rotated = ov.space.rotate_space_visium(",
        "    adata, angle=45, library_id='V1_Human_Brain')",
        "# Custom center rotation",
        "adata_rotated = ov.space.rotate_space_visium(",
        "    adata, angle=90, center=(100, 100),",
        "    library_id='sample_1', interpolation_order=3)",
        "# Precise rotation with high-quality interpolation",
        "adata_rotated = ov.space.rotate_space_visium(",
        "    adata, angle=30, res='hires',",
        "    library_id=library_id, interpolation_order=1)",
        "# Apply spatial mapping after rotation",
        "ov.space.map_spatial_auto(adata_rotated, method='phase')"
    ],
    related=["space.crop_space_visium", "space.map_spatial_auto", "space.map_spatial_manual"]
)
def rotate_space_visium(
    adata,
    angle,
    center=None,
    spatial_key='spatial',
    res='hires',
    library_id=None,
    interpolation_order=1,
    library_key=None,
):
    """
    Rotate Visium spatial data image and coordinates by a specified angle.

    This function performs rotation of both the tissue image and spot coordinates
    while maintaining proper alignment and scaling.

    Arguments:
        adata: AnnData
            Annotated data matrix containing Visium spatial data.
        angle: float
            Rotation angle in degrees (positive for counterclockwise).
        center: tuple, optional (default=None)
            (x, y) coordinates of rotation center. If None, uses center of spatial coordinates.
        spatial_key: str, optional (default='spatial')
            Key in adata.obsm containing spatial coordinates.
        res: str, optional (default='hires')
            Image resolution to use ('hires' or 'lowres').
        library_id: str
            Library ID for the spatial data in adata.uns['spatial'].
        interpolation_order: int, optional (default=1)
            Order of interpolation for image rotation:
            0: nearest neighbor, 1: bilinear, 3: cubic spline.
        library_key: str or None
            Observation column identifying library membership. Required for a
            multi-library AnnData; only matching coordinates are transformed.

    Returns:
        AnnData
            Rotated AnnData object with transformed coordinates and image.

    Notes:
        - The function preserves the original coordinate system scaling
        - Both image and spot coordinates are rotated around the same center
        - The rotation is performed counterclockwise for positive angles
        - Image interpolation can be adjusted for quality vs speed tradeoff

    Examples:
        >>> import scanpy as sc
        >>> import omicverse as ov
        >>> # Load Visium data
        >>> adata = sc.read_visium(...)
        >>> # Rotate 45 degrees counterclockwise
        >>> adata_rotated = ov.space.rotate_space_visium(
        ...     adata,
        ...     angle=45,
        ...     library_id='V1_Human_Brain'
        ... )
    """
    from skimage.transform import rotate as rotate_image

    library_id, library_mask = _resolve_spatial_library(
        adata,
        library_id=library_id,
        library_key=library_key,
    )
    adata_rotate = adata.copy()
    if spatial_key not in adata_rotate.obsm:
        raise KeyError(f"{spatial_key!r} was not found in adata.obsm.")
    if library_id not in adata_rotate.uns.get("spatial", {}):
        raise KeyError(f"library_id {library_id!r} was not found in adata.uns['spatial'].")

    spatial_info = adata_rotate.uns["spatial"][library_id]
    if res not in spatial_info.get("images", {}):
        raise KeyError(f"Image resolution {res!r} was not found for library {library_id!r}.")
    scale_key = f"tissue_{res}_scalef"
    if scale_key not in spatial_info.get("scalefactors", {}):
        raise KeyError(f"Scalefactor {scale_key!r} was not found for library {library_id!r}.")

    images = spatial_info.get("images", {})
    scalefactors = spatial_info.get("scalefactors", {})
    original_spatial = np.asarray(adata_rotate.obsm[spatial_key], dtype=np.float64).copy()
    if original_spatial.ndim != 2 or original_spatial.shape[1] < 2:
        raise ValueError(
            f"adata.obsm[{spatial_key!r}] must have at least two coordinate columns."
        )
    if not np.isfinite(original_spatial[:, :2]).all():
        raise ValueError(f"adata.obsm[{spatial_key!r}] contains non-finite coordinates.")

    selected_spatial = original_spatial[library_mask, :2]
    rotation_center_spatial = np.asarray(
        selected_spatial.mean(axis=0) if center is None else center,
        dtype=np.float64,
    )
    if rotation_center_spatial.shape != (2,) or not np.isfinite(rotation_center_spatial).all():
        raise ValueError("center must contain two finite spatial coordinates (x, y).")

    rotated_images = {}
    stale_image_keys = []
    for image_key, image in list(images.items()):
        image_scale_key = f"tissue_{image_key}_scalef"
        if image_scale_key not in scalefactors or image is None:
            stale_image_keys.append(image_key)
            continue
        image_array = np.asarray(image)
        if image_array.ndim < 2:
            stale_image_keys.append(image_key)
            continue
        image_scale = float(scalefactors[image_scale_key])
        rotation_center_image = tuple(rotation_center_spatial * image_scale)
        rotated_image = rotate_image(
            image_array,
            angle=float(angle),
            resize=False,
            center=rotation_center_image,
            order=int(interpolation_order),
            mode="reflect",
            preserve_range=True,
        )
        if np.issubdtype(image_array.dtype, np.integer):
            limits = np.iinfo(image_array.dtype)
            rotated_image = np.clip(np.rint(rotated_image), limits.min, limits.max)
        rotated_images[image_key] = rotated_image.astype(image_array.dtype, copy=False)
    spatial_info["images"] = rotated_images
    _record_removed_stale_images(spatial_info, "rotate", stale_image_keys)

    theta_rad = np.radians(float(angle))
    rotation = np.array(
        [
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad), np.cos(theta_rad)],
        ]
    )
    rotated_spatial = original_spatial.copy()
    rotated_spatial[library_mask, :2] = (
        (selected_spatial - rotation_center_spatial) @ rotation
        + rotation_center_spatial
    )
    adata_rotate.obsm[spatial_key] = rotated_spatial
    return adata_rotate


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift
import os
import scanpy as sc  # 假设您使用了 scanpy

def find_image_offset_phase_correlation_array_input(image1_array, image2_array):
    """
    Calculate image offset using phase correlation method.

    This function computes the relative displacement between two images using
    Fourier-based phase correlation, which is robust to intensity variations
    and noise.

    Arguments:
        image1_array: numpy.ndarray
            First image array (image to be aligned).
            Can be grayscale or RGB.
        image2_array: numpy.ndarray
            Second image array (reference image).
            Can be grayscale or RGB.

    Returns:
        tuple
            (offset, aligned_image) where:
            - offset: tuple (dx, dy) representing the displacement in pixels
            - aligned_image: numpy.ndarray of the aligned first image

    Notes:
        - Images are automatically converted to grayscale if RGB
        - The method is based on frequency-domain phase correlation
        - Works best with images of similar size and content
        - Handles sub-pixel accuracy for precise alignment

    Examples:
        >>> import omicverse as ov
        >>> import numpy as np
        >>> # Create sample images
        >>> img1 = np.random.rand(100, 100)
        >>> img2 = np.roll(img1, shift=(5, 10), axis=(0, 1))
        >>> # Find offset
        >>> offset, aligned = ov.space.find_image_offset_phase_correlation_array_input(
        ...     img1, img2
        ... )
        >>> print(f"Detected offset: {offset}")
    """
    # **不再需要读取图像文件，直接使用输入的 NumPy 数组**
    import cv2
    img1 = image1_array
    img2 = image2_array

    # 确保是灰度图 (如果您的 img 已经是灰度图，则可以跳过)
    if len(img1.shape) > 2 and img1.shape[2] > 1: # 检查是否是彩色图像
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) # 假设是 RGB，转为灰度
    if len(img2.shape) > 2 and img2.shape[2] > 1:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # 将图像转换为 float32 类型，以便进行傅里叶变换
    img1_float32 = np.float32(img1)
    img2_float32 = np.float32(img2)

    # 计算傅里叶变换
    fft_img1 = np.fft.fft2(img1_float32)
    fft_img2 = np.fft.fft2(img2_float32)

    # 计算互功率谱 (Cross-Power Spectrum)
    conjugate_fft_img2 = np.conjugate(fft_img2) # 共轭复数
    cross_power_spectrum = fft_img1 * conjugate_fft_img2

    # 归一化互功率谱 (为了增强峰值)
    magnitude = np.abs(cross_power_spectrum)
    normalized_cross_power_spectrum = cross_power_spectrum / np.maximum(magnitude, 1e-12)

    # 计算逆傅里叶变换，得到脉冲响应 (Inverse FFT)
    inverse_fft = np.fft.ifft2(normalized_cross_power_spectrum)

    # 找到峰值位置 (峰值位置对应偏移量)
    peak_position = np.unravel_index(np.argmax(np.abs(inverse_fft)), inverse_fft.shape)

    # 计算偏移量 (需要考虑图像尺寸和峰值位置的转换关系)
    shift_y = peak_position[0]
    shift_x = peak_position[1]

    if shift_y > img1.shape[0] // 2: # 如果峰值在图像的下半部分，则偏移为负
        shift_y -= img1.shape[0]
    if shift_x > img1.shape[1] // 2: # 如果峰值在图像的右半部分，则偏移为负
        shift_x -= img1.shape[1]

    offset = (shift_x, shift_y)

    # 平移图像 image1
    image1_aligned = shift(img1, shift=(-shift_y, -shift_x), order=0) # 注意 shift 的方向

    return offset, image1_aligned

def find_image_offset_phase_correlation_torch(image1_tensor, image2_tensor):
    """
    Calculate image offset using phase correlation method with PyTorch tensors.

    This function is the PyTorch implementation of phase correlation for image
    alignment, offering GPU acceleration when available.

    Arguments:
        image1_tensor: torch.Tensor
            First image as PyTorch tensor (image to be aligned).
            Expected shape: (H, W) or (1, H, W) or (3, H, W).
        image2_tensor: torch.Tensor
            Second image as PyTorch tensor (reference image).
            Expected shape: (H, W) or (1, H, W) or (3, H, W).

    Returns:
        tuple
            (offset, aligned_image) where:
            - offset: tuple (dx, dy) representing the displacement in pixels
            - aligned_image: torch.Tensor of the aligned first image

    Notes:
        - Images are automatically converted to grayscale if RGB
        - Computation is performed on GPU if available
        - More efficient than NumPy version for large images when using GPU
        - Maintains sub-pixel accuracy for precise alignment

    Examples:
        >>> import omicverse as ov
        >>> import torch
        >>> # Create sample images
        >>> img1 = torch.rand(100, 100)
        >>> img2 = torch.roll(img1, shifts=(5, 10), dims=(0, 1))
        >>> # Find offset
        >>> offset, aligned = ov.space.find_image_offset_phase_correlation_torch(
        ...     img1, img2
        ... )
        >>> print(f"Detected offset: {offset}")
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise build_optional_dependency_error(
            "omicverse.space.find_image_offset_phase_correlation_torch",
            ("torch",),
        ) from exc

    if image1_tensor.shape != image2_tensor.shape:
        raise ValueError(
            f"image tensors must have the same shape; got {tuple(image1_tensor.shape)} "
            f"and {tuple(image2_tensor.shape)}."
        )
    if image1_tensor.ndim not in (2, 3):
        raise ValueError("image tensors must have shape (H, W) or (C, H, W).")

    device = image1_tensor.device
    original_ndim = image1_tensor.ndim
    image1_chw = image1_tensor.unsqueeze(0) if original_ndim == 2 else image1_tensor
    image2_chw = image2_tensor.unsqueeze(0) if original_ndim == 2 else image2_tensor
    image1_chw = image1_chw.to(device=device, dtype=torch.float32)
    image2_chw = image2_chw.to(device=device, dtype=torch.float32)

    image1_gray = image1_chw.mean(dim=0, keepdim=True)
    image2_gray = image2_chw.mean(dim=0, keepdim=True)

    # 傅里叶变换
    fft1 = torch.fft.fft2(image1_gray)
    fft2 = torch.fft.fft2(image2_gray)
    
    # 计算互功率谱（带数值稳定性保护）
    cross_power_spectrum = fft1 * torch.conj(fft2)
    eps = 1e-8
    normalized_cross_power = cross_power_spectrum / (torch.abs(cross_power_spectrum) + eps)
    
    # 逆傅里叶变换
    inv_cross_power = torch.fft.ifft2(normalized_cross_power).real
    
    # 寻找峰值位置
    _, h, w = inv_cross_power.shape
    max_idx = int(torch.argmax(torch.abs(inv_cross_power)).item())
    shift_y, shift_x = np.unravel_index(max_idx, (h, w))
    
    # 计算循环位移量
    if shift_y > h // 2:
        shift_y -= h
    if shift_x > w // 2:
        shift_x -= w
    
    # affine_grid translations use normalized output-to-input coordinates. The
    # phase-correlation shift already has the correct inverse-sampling sign.
    tx = 0.0 if w <= 1 else 2.0 * float(shift_x) / float(w - 1)
    ty = 0.0 if h <= 1 else 2.0 * float(shift_y) / float(h - 1)
    theta = torch.tensor(
        [[1.0, 0.0, tx], [0.0, 1.0, ty]],
        dtype=torch.float32,
        device=device,
    )

    batch = image1_chw.unsqueeze(0)
    grid = F.affine_grid(theta.unsqueeze(0), batch.shape, align_corners=True)
    aligned = F.grid_sample(
        batch,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0)
    if original_ndim == 2:
        aligned = aligned.squeeze(0)

    return (int(shift_x), int(shift_y)), aligned

def _create_spatial_image(adata, ax, color=None, alpha=1, save_path=None):
    """
    Create a spatial plot of gene expression or feature values.

    Internal function to generate spatial plots with customizable appearance.

    Arguments:
        adata: AnnData
            Annotated data matrix containing spatial data.
        ax: matplotlib.axes.Axes
            Matplotlib axes object to plot on.
        color: str or None, optional (default=None)
            Key in adata.obs or adata.var for coloring points.
        alpha: float, optional (default=1)
            Transparency of the plotted points (0 to 1).
        save_path: str or None, optional (default=None)
            Path to save the plot. If None, plot is not saved.

    Returns:
        matplotlib.axes.Axes
            The axes object containing the plot.

    Notes:
        - This is an internal function used by other plotting functions
        - Handles both categorical and continuous coloring
        - Automatically scales point sizes based on data
    """
    if color is not None:
        adata.obs['temp_color'] = adata.obs[color] # 使用临时列，避免修改原始 adata.obs
    else:
        adata.obs['temp_color'] = '1' # 使用默认值，确保 `color` 参数可以处理 None

    library_id = list(adata.uns['spatial'].keys())[0]
    scalefactors = adata.uns['spatial'][library_id]['scalefactors']
    tissue_hires_scalef = scalefactors['tissue_hires_scalef']

    sc.pl.embedding(
        adata,
        basis='spatial',
        color='temp_color', # 使用临时列
        show=False,
        ax=ax,
        size=10,
        scale_factor=tissue_hires_scalef,
        legend_loc=None
    )

    cur_coords = np.concatenate([ax.get_xlim(), ax.get_ylim()])
    img = adata.uns["spatial"][library_id]["images"]["hires"]
    ax.imshow(img, cmap='gray', alpha=alpha) # 使用传入的 alpha 参数
    ax.set_xlim(cur_coords[0], cur_coords[1])
    ax.set_ylim(cur_coords[3], cur_coords[2])
    ax.set_xticks([]) # 使用 ax 设置，更简洁
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    ax.axis(False) # 隐藏坐标轴

    if save_path:
        plt.savefig(save_path, bbox_inches='tight') # 使用 plt.savefig, 更通用
    if 'temp_color' in adata.obs: # 清理临时列
        del adata.obs['temp_color']


def _map_spatial_img(adata_rotated, color=None,
                    method='phase_correlation'):
    """
    Map and align spatial transcriptomics data with tissue image.

    Internal function to perform image alignment between spatial data and tissue image.

    Arguments:
        adata_rotated: AnnData
            Annotated data matrix containing rotated spatial data.
        color: str or None, optional (default=None)
            Key in adata.obs or adata.var for visualization.
        method: str, optional (default='phase_correlation')
            Method for image alignment:
            - 'phase_correlation': Use phase correlation
            - 'feature': Use feature-based alignment

    Returns:
        tuple
            (offset, aligned_adata) where:
            - offset: tuple (dx, dy) representing the optimal alignment
            - aligned_adata: AnnData object with aligned coordinates

    Notes:
        - This is an internal function used by map_spatial_auto
        - Different alignment methods may work better for different data types
        - Phase correlation is generally more robust to intensity variations
    """
    import cv2
    scale_factor_denominator = pow(10, (len(str(int(adata_rotated.obsm['spatial'][:, 0].mean()))) - 2))
    fig_size_x = adata_rotated.obsm['spatial'][:, 0].mean() / scale_factor_denominator
    fig_size_y = adata_rotated.obsm['spatial'][:, 1].mean() / scale_factor_denominator

    # 确保 'temp' 目录存在
    os.makedirs('temp', exist_ok=True)

    # 创建并保存 image1 (带点图层)
    fig1, ax1 = plt.subplots(figsize=(fig_size_x, fig_size_y))
    _create_spatial_image(adata_rotated, ax1, color=color, alpha=0, save_path='temp/image1.png')
    image1_path = 'temp/image1.png'

    # 创建并保存 image2 (仅背景图像)
    fig2, ax2 = plt.subplots(figsize=(fig_size_x, fig_size_y))
    _create_spatial_image(adata_rotated, ax2, alpha=1, save_path='temp/image2.png') #  不传递 color, 仅背景图像
    image2_path = 'temp/image2.png'


    # 从文件读取图像 (实际应用中，如果可能，尽量避免读写文件，直接在内存中操作 NumPy 数组)
    img1_from_memory = cv2.imread(image1_path)
    img2_from_memory = cv2.imread(image2_path)
    if method == 'phase_correlation':
        # 使用相位相关法计算偏移量
        offset, _ = find_image_offset_phase_correlation_array_input(img1_from_memory, img2_from_memory)
        print(f"估计的偏移量 (dx, dy) 使用相位相关 (文件图像): {offset}")
    elif method == 'phase_correlation_torch':
        try:
            import torch
        except ImportError as exc:
            raise build_optional_dependency_error(
                "omicverse.space._map_spatial_img(method='phase_correlation_torch')",
                ("torch",),
            ) from exc
        # 转换到PyTorch张量 (需要添加的预处理步骤)
        img1_tensor = torch.from_numpy(img1_from_memory).permute(2, 0, 1).float() / 255.0  # [C, H, W] 格式
        img2_tensor = torch.from_numpy(img2_from_memory).permute(2, 0, 1).float() / 255.0  # 归一化到[0,1]

        # 可选：移动到GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img1_tensor = img1_tensor.to(device)
        img2_tensor = img2_tensor.to(device)

        # 调用函数
        offset, image1_aligned_pytorch = find_image_offset_phase_correlation_torch(
            img1_tensor, img2_tensor
        )

        # 转换回numpy用于显示 (需要添加的后处理步骤)
        image1_aligned_np = image1_aligned_pytorch.cpu().permute(1, 2, 0).numpy()  # [H, W, C]
        image1_aligned_np = (image1_aligned_np * 255).astype(np.uint8)  # 恢复[0,255]

        print(f"估计的偏移量 (dx, dy) 使用 PyTorch 相位相关 (文件图像): {offset}")

    # 应用偏移量到空间坐标
    adata_rotated.obsm['spatial1'] = adata_rotated.obsm['spatial'].copy()
    adata_rotated.obsm['spatial1'][:, 0] -= (offset[0]) * (
        adata_rotated.obsm['spatial'][:, 0].mean() / (img1_from_memory.shape[0] / 2)
    )
    adata_rotated.obsm['spatial1'][:, 1] -= (offset[1]) * (
        adata_rotated.obsm['spatial'][:, 1].mean() / (img1_from_memory.shape[1] / 2)
    )
    return adata_rotated


@register_function(
    aliases=["自动空间映射", "map_spatial_auto", "auto_spatial_mapping", "空间自动映射", "自动对齐"],
    category="space",
    description="Automatically map and align spatial transcriptomics data with tissue image",
    examples=[
        "# Basic auto mapping with phase correlation",
        "ov.space.map_spatial_auto(adata_rotated, method='phase')",
        "# Torch-accelerated phase correlation",
        "ov.space.map_spatial_auto(adata_rotated, method='torch')",
        "# After rotation, apply auto mapping",
        "adata_rotated = ov.space.rotate_space_visium(",
        "    adata, angle=45, library_id=library_id)",
        "ov.space.map_spatial_auto(adata_rotated)"
    ],
    related=["space.map_spatial_manual", "space.rotate_space_visium", "space.crop_space_visium"]
)
def map_spatial_auto(
    adata_rotated,
    method='phase',
    library_id=None,
    library_key=None,
    res='hires',
):
    """
    Automatically map and align spatial transcriptomics data.

    This function performs automatic alignment of spatial transcriptomics data
    with the corresponding tissue image using various alignment methods.

    Arguments:
        adata_rotated: AnnData
            Annotated data matrix containing spatial data to be aligned.
        method: str, optional (default='phase')
            Alignment method to use:
            - 'phase': Phase correlation-based alignment
            - 'torch': Torch-accelerated phase correlation
        library_id: str or None
            Spatial library whose image should be aligned. Required when more
            than one library is present.
        library_key: str or None
            Observation column identifying library membership. Required for a
            multi-library AnnData.
        res: str, default='hires'
            Image resolution used for registration.

    Returns:
        AnnData
            Aligned AnnData object with updated spatial coordinates.

    Notes:
        - The function automatically selects the best alignment
        - Results can be verified using spatial plotting functions
        - Different methods may work better for different data types

    Examples:
        >>> import scanpy as sc
        >>> import omicverse as ov
        >>> # Load and preprocess data
        >>> adata = sc.read_visium(...)
        >>> # Perform automatic alignment
        >>> adata_aligned = ov.space.map_spatial_auto(
        ...     adata,
        ...     method='phase'
        ... )
    """
    import tempfile

    method = str(method).lower()
    if method not in {'phase', 'torch'}:
        raise ValueError("method must be 'phase' or 'torch'.")
    library_id, library_mask = _resolve_spatial_library(
        adata_rotated,
        library_id=library_id,
        library_key=library_key,
    )
    plot_adata = adata_rotated[library_mask].copy()
    spatial_info = adata_rotated.uns['spatial'][library_id]
    if res not in spatial_info.get('images', {}):
        raise KeyError(f"Image resolution {res!r} was not found for library {library_id!r}.")
    scale_key = f'tissue_{res}_scalef'
    if scale_key not in spatial_info.get('scalefactors', {}):
        raise KeyError(f"Scalefactor {scale_key!r} was not found for library {library_id!r}.")
    img = np.asarray(spatial_info['images'][res])
    image_height, image_width = img.shape[:2]
    image_scale_factor = float(spatial_info['scalefactors'][scale_key])
    if not np.isfinite(image_scale_factor) or image_scale_factor <= 0:
        raise ValueError(f"Scalefactor {scale_key!r} must be a positive finite value.")
    plot_adata.obsm['spatial'] = (
        np.asarray(plot_adata.obsm['spatial'], dtype=np.float64)
        * image_scale_factor
    )
    max_inches = 8.0
    longest_side = max(image_height, image_width, 1)
    figsize = (
        max(2.0, max_inches * image_width / longest_side),
        max(2.0, max_inches * image_height / longest_side),
    )

    temp_dir = tempfile.TemporaryDirectory(prefix='omicverse_spatial_align_')
    image1_path = os.path.join(temp_dir.name, 'image1.png')
    image2_path = os.path.join(temp_dir.name, 'image2.png')

    fig1, ax1 = plt.subplots(figsize=figsize)
    ax=ax1
    sc.pl.embedding(
        plot_adata,
        basis='spatial',
        color=None,
        show=False,
        ax=ax1,
        size=10,
        #frameon=False,
        legend_loc=None
    )
    cur_coords = np.concatenate([ax1.get_xlim(), ax1.get_ylim()])
    ax.set_xlim(cur_coords[0], cur_coords[1])
    ax.set_ylim(cur_coords[3], cur_coords[2])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.title('')
    
    ax1.imshow(img, cmap='gray', alpha=0)
    plt.xlim(cur_coords[0], cur_coords[1])
    plt.ylim(cur_coords[3], cur_coords[2])
    plt.xticks([])
    plt.yticks([])
    plt.axis(False)

    fig1.savefig(image1_path,bbox_inches='tight', )

    fig2, ax2 = plt.subplots(figsize=figsize)
    ax=ax2
    
    ax2.imshow(img, cmap='gray', alpha=1)
    ax.set_xlim(cur_coords[0], cur_coords[1])
    ax.set_ylim(cur_coords[3], cur_coords[2])
    plt.xticks([])
    plt.yticks([])
    plt.axis(False)
    fig2.savefig(image2_path,bbox_inches='tight', )

    

    import cv2 # 导入 cv2 只是为了模拟读取 png, 实际您不需要读取文件了
    img1_from_memory = cv2.imread(image1_path) # 模拟从内存获取 image1 的 numpy array
    img2_from_memory = cv2.imread(image2_path) # 模拟从内存获取 image2 的 numpy array
    plt.close(fig1)
    plt.close(fig2)
    
    

    # 修改后的配准逻辑
    if method == 'torch':
        try:
            import torch
        except ImportError as exc:
            raise build_optional_dependency_error(
                "omicverse.space.map_spatial_auto(method='torch')",
                ("torch",),
            ) from exc
        # PyTorch处理路径
        img1_tensor = torch.from_numpy(img1_from_memory).permute(2, 0, 1).float() / 255.0
        img2_tensor = torch.from_numpy(img2_from_memory).permute(2, 0, 1).float() / 255.0
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img1_tensor = img1_tensor.to(device)
        img2_tensor = img2_tensor.to(device)
        
        offset, image1_aligned = find_image_offset_phase_correlation_torch(img1_tensor, img2_tensor)
        image1_aligned = image1_aligned.cpu().permute(1, 2, 0).numpy() * 255
        image1_aligned = image1_aligned.astype(np.uint8)
        
    elif method == 'phase':
        # 原相位相关法处理
        offset, image1_aligned = find_image_offset_phase_correlation_array_input(img1_from_memory, img2_from_memory)
    # 保持原有坐标修正逻辑不变 ...
    # 1) 先把 spatial1 转成 float64
    adata_rotated.obsm['spatial1'] = adata_rotated.obsm['spatial'].astype(np.float64)

    # Convert registration pixels back into plotted coordinate units. This is
    # derived from the rendered axes span, so centered/negative coordinates do
    # not collapse the translation to zero.
    x_span = abs(float(cur_coords[1] - cur_coords[0]))
    y_span = abs(float(cur_coords[3] - cur_coords[2]))
    x_per_pixel = (
        x_span / max(img1_from_memory.shape[1], 1) / image_scale_factor
    )
    y_per_pixel = (
        y_span / max(img1_from_memory.shape[0], 1) / image_scale_factor
    )
    adata_rotated.obsm['spatial1'][library_mask, 0] -= offset[0] * x_per_pixel
    adata_rotated.obsm['spatial1'][library_mask, 1] -= offset[1] * y_per_pixel
    temp_dir.cleanup()
    return adata_rotated

@register_function(
    aliases=["手动空间映射", "map_spatial_manual", "manual_spatial_mapping", "空间手动映射", "手动对齐"],
    category="space",
    description="Manually adjust spatial transcriptomics data alignment with specified offset",
    examples=[
        "# Apply manual offset",
        "ov.space.map_spatial_manual(adata_rotated, offset=(-1500, 1000),",
        "                            offset_mode='absolute')",
        "# Fine-tune alignment",
        "ov.space.map_spatial_manual(adata_rotated, offset=(50, -30))",
        "# Large adjustment",
        "ov.space.map_spatial_manual(adata_rotated, offset=(-500, 200))",
        "# Access manually aligned coordinates",
        "manual_coords = adata_rotated.obsm['spatial1']"
    ],
    related=["space.map_spatial_auto", "space.rotate_space_visium", "space.crop_space_visium"]
)
def map_spatial_manual(
        adata_rotated,
        offset,
        spatial_key='spatial',
        key_added='spatial1',
        offset_mode='legacy',
        library_id=None,
        library_key=None,
):
    """
    Manually adjust spatial transcriptomics data alignment.

    This function allows manual adjustment of the alignment between
    spatial transcriptomics data and the tissue image using specified offsets.

    Arguments:
        adata_rotated: AnnData
            Annotated data matrix containing spatial data to be aligned.
        offset: tuple
            (dx, dy) tuple specifying the manual offset to apply.
        spatial_key: str, default='spatial'
            Coordinate matrix to translate.
        key_added: str, default='spatial1'
            Coordinate key receiving the translated floating-point values.
        offset_mode: {'legacy', 'absolute'}, default='legacy'
            ``'legacy'`` preserves the historical mean/max-scaled subtraction
            and emits a deprecation warning. ``'absolute'`` applies ``(dx, dy)``
            directly in the coordinate units, with positive values moving right
            and down.
        library_id: str or None
            Spatial library to translate. Required together with ``library_key``
            for multi-library objects.
        library_key: str or None
            Observation column identifying library membership. Only observations
            matching ``library_id`` are translated.

    Returns:
        AnnData
            Aligned AnnData object with manually adjusted spatial coordinates.

    Notes:
        - Useful for fine-tuning automatic alignment results
        - In ``offset_mode='absolute'``, offsets use the coordinate units.
        - Positive absolute dx moves spots right; positive dy moves spots down.

    Examples:
        >>> import scanpy as sc
        >>> import omicverse as ov
        >>> # Load data
        >>> adata = sc.read_visium(...)
        >>> # Apply manual offset
        >>> adata_aligned = ov.space.map_spatial_manual(
        ...     adata,
        ...     offset=(10, -5),  # Move 10 pixels right, 5 pixels up
        ...     offset_mode='absolute'
        ... )
    """
    if spatial_key not in adata_rotated.obsm:
        raise KeyError(f"{spatial_key!r} was not found in adata.obsm.")
    offset = np.asarray(offset, dtype=np.float64)
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError("offset must contain two finite values (dx, dy).")
    translated = np.asarray(adata_rotated.obsm[spatial_key], dtype=np.float64).copy()
    if translated.ndim != 2 or translated.shape[1] < 2:
        raise ValueError(f"adata.obsm[{spatial_key!r}] must have at least two columns.")
    spatial_mapping = adata_rotated.uns.get("spatial")
    if isinstance(spatial_mapping, dict) and spatial_mapping:
        if len(spatial_mapping) > 1 or library_id is not None or library_key is not None:
            _, library_mask = _resolve_spatial_library(
                adata_rotated,
                library_id=library_id,
                library_key=library_key,
            )
        else:
            library_mask = np.ones(adata_rotated.n_obs, dtype=bool)
    elif library_id is not None or library_key is not None:
        raise KeyError(
            "library_id/library_key selection requires non-empty adata.uns['spatial']."
        )
    else:
        library_mask = np.ones(adata_rotated.n_obs, dtype=bool)

    selected = translated[library_mask, :2]
    offset_mode = str(offset_mode).lower()
    if offset_mode == 'absolute':
        translated[library_mask, :2] = selected + offset
    elif offset_mode == 'legacy':
        import warnings

        warnings.warn(
            "offset_mode='legacy' preserves the historical mean/max-scaled "
            "translation and is deprecated. Use offset_mode='absolute' for "
            "explicit coordinate-unit offsets.",
            FutureWarning,
            stacklevel=2,
        )
        maxima = np.max(selected, axis=0)
        if np.any(np.isclose(maxima, 0)):
            raise ValueError(
                "Legacy offset scaling is undefined when a coordinate-axis "
                "maximum is zero; use offset_mode='absolute'."
            )
        translated[library_mask, :2] = selected - offset * (
            np.mean(selected, axis=0) / maxima
        )
    else:
        raise ValueError("offset_mode must be 'legacy' or 'absolute'.")
    adata_rotated.obsm[key_added] = translated
    return adata_rotated


@register_function(
    aliases=["读取Visium数据", "read_visium_10x", "load_visium", "Visium数据读取", "空间数据载入"],
    category="space",
    description="Read and process 10x Visium spatial transcriptomics data",
    examples=[
        "# Basic Visium data loading",
        "adata = ov.space.read_visium_10x(adata_path)",
        "# With custom parameters",
        "adata = ov.space.read_visium_10x(adata_path, genome='GRCh38')",
        "# Select an alternate Space Ranger count matrix",
        "adata = ov.space.read_visium_10x(path, count_file='raw_feature_bc_matrix.h5')"
    ],
    related=["space.svg", "space.crop_space_visium"]
)
def read_visium_10x(
        adata,
        **kwargs,
):
    """Read and standardize 10x Visium data with bin2cell-compatible loader.

    Parameters
    ----------
    adata : str or AnnData
        Input Visium path/object accepted by ``bin2cell.read_visium``.
    **kwargs
        Additional arguments forwarded to ``read_visium``.

    Returns
    -------
    AnnData
        Visium AnnData with unique variable names.
    """
    from ..external.bin2cell import read_visium
    adata = read_visium(adata, **kwargs)
    adata.var_names_make_unique()
    return adata


@register_function(
    aliases=["Visium细胞分割HE", "visium_10x_hd_cellpose_he", "cellpose_he", "HE图像分割", "细胞核分割"],
    category="space",
    description="Cell segmentation on H&E images for high-resolution Visium data using CellPose",
    examples=[
        "# Basic H&E segmentation",
        "adata = ov.space.visium_10x_hd_cellpose_he(adata, mpp=0.3)",
        "# Custom parameters",
        "adata = ov.space.visium_10x_hd_cellpose_he(adata, mpp=0.5,",
        "                                           prob_thresh=0.1,",
        "                                           flow_threshold=0.3)",
        "# GPU acceleration",
        "adata = ov.space.visium_10x_hd_cellpose_he(adata, gpu=True,",
        "                                           buffer=200)"
    ],
    related=["space.visium_10x_hd_cellpose_gex", "space.bin2cell"]
)
def visium_10x_hd_cellpose_he(
        adata,
        mpp=0.3,
        he_save_path="stardist/he_colon.tiff",
        prob_thresh=0,
        flow_threshold=0.2,
        gpu=True,
        buffer=150,
        backend='tifffile',
        **kwargs,
):
    """
    Convert Visium 10x data to cell-level data.
    
    """
    from ..external.bin2cell import destripe, scaled_he_image, cellseg, insert_labels

    spatial_key = f"spatial_cropped_{buffer}_buffer"
    if not os.path.exists(he_save_path):
        destripe(adata)
        scaled_he_image(adata, mpp=mpp, buffer=buffer, save_path=he_save_path,
                        backend=backend)
    else:
        print(f"he_save_path {he_save_path} already exists, skipping image generation")
        # destripe and scaled_he_image create spatial metadata needed downstream
        if spatial_key not in adata.obsm:
            destripe(adata)
            scaled_he_image(adata, mpp=mpp, buffer=buffer, save_path=None,
                            backend=backend)
    cellseg(image_path=he_save_path    , 
             labels_npz_path=he_save_path.replace(".tiff", ".npz"), 
             stardist_model="2D_versatile_he", 
             prob_thresh=prob_thresh,
             flow_threshold=flow_threshold,
             gpu=gpu,
             **kwargs,
            )
    insert_labels(adata, 
                  labels_npz_path=he_save_path.replace(".tiff", ".npz"), 
                  basis="spatial", 
                  spatial_key=f"spatial_cropped_{buffer}_buffer",
                  mpp=mpp, 
                  labels_key="labels_he"
                 )
    return adata
    
@register_function(
    aliases=["Visium细胞扩展", "visium_10x_hd_cellpose_expand", "cellpose_expand", "细胞标签扩展", "空间扩展"],
    category="space",
    description="Expand cell segmentation labels to nearby bins for improved coverage",
    examples=[
        "# Basic label expansion",
        "adata = ov.space.visium_10x_hd_cellpose_expand(adata,",
        "                                               max_bin_distance=4)",
        "# Custom parameters",
        "adata = ov.space.visium_10x_hd_cellpose_expand(adata,",
        "                                               max_bin_distance=6,",
        "                                               labels_key='cellpose_labels')",
        "# Different expansion keys",
        "adata = ov.space.visium_10x_hd_cellpose_expand(adata,",
        "                                               expanded_labels_key='expanded_cells')"
    ],
    related=["space.visium_10x_hd_cellpose_he", "space.visium_10x_hd_cellpose_gex"]
)
def visium_10x_hd_cellpose_expand(
        adata,
        max_bin_distance=4,
        labels_key="labels_he",
        expanded_labels_key="labels_he_expanded",
        **kwargs,
):
    """Expand segmentation labels from nuclei to nearby bins.

    Parameters
    ----------
    adata : AnnData
        Visium HD AnnData containing primary segmentation labels.
    max_bin_distance : int, default=4
        Maximum bin distance for expansion.
    labels_key : str, default='labels_he'
        Source label column/key.
    expanded_labels_key : str, default='labels_he_expanded'
        Output label key for expanded labels.
    **kwargs
        Extra arguments forwarded to ``expand_labels``.

    Returns
    -------
    AnnData
        The same object after updating labels in place.
    """
    from ..external.bin2cell import expand_labels
    expand_labels(adata, 
                  labels_key=labels_key, 
                  expanded_labels_key=expanded_labels_key,
                  max_bin_distance=max_bin_distance,
                  **kwargs,
                 )
    return adata
    
@register_function(
    aliases=["Visium细胞基因表达", "visium_10x_hd_cellpose_gex", "cellpose_gex", "细胞基因表达映射", "细胞水平表达"],
    category="space",
    description="Map gene expression to cell level using CellPose segmentation",
    examples=[
        "# Basic gene expression mapping",
        "adata = ov.space.visium_10x_hd_cellpose_gex(adata)",
        "# Custom parameters",
        "adata = ov.space.visium_10x_hd_cellpose_gex(adata,",
        "                                            obs_key='total_counts',",
        "                                            mpp=0.5, sigma=3)",
        "# With log transformation",
        "adata = ov.space.visium_10x_hd_cellpose_gex(adata, log1p=True,",
        "                                            prob_thresh=0.1)"
    ],
    related=["space.visium_10x_hd_cellpose_he", "space.bin2cell"]
)
def visium_10x_hd_cellpose_gex(
        adata,
        obs_key="n_counts_adjusted",
        log1p=False,
        mpp=0.3,
        sigma=5,
        gex_save_path="stardist/gex_colon.tiff",
        prob_thresh=0.01,
        nms_thresh=0.1,
        gpu=True,
        buffer=150,
        **kwargs,
):
    """Run expression-image segmentation and map labels back to spatial bins.

    Parameters
    ----------
    adata : AnnData
        Visium HD AnnData.
    obs_key : str, default='n_counts_adjusted'
        Observation value used to generate gene-expression image.
    log1p : bool, default=False
        Whether to log-transform values when generating image.
    mpp : float, default=0.3
        Microns-per-pixel scale.
    sigma : int, default=5
        Gaussian smoothing parameter for grid image generation.
    gex_save_path : str, default='stardist/gex_colon.tiff'
        Output path of generated expression image.
    prob_thresh : float, default=0.01
        StarDist probability threshold.
    nms_thresh : float, default=0.1
        StarDist non-max-suppression threshold.
    gpu : bool, default=True
        Whether to use GPU in StarDist inference.
    buffer : int, default=150
        Crop buffer used for spatial coordinate key.
    **kwargs
        Additional StarDist arguments.

    Returns
    -------
    AnnData
        The same object after writing ``labels_gex`` in place.
    """
    from ..external.bin2cell import grid_image, cellseg, insert_labels,destripe
    #if gex_save_path's file exist, jump grid_image to stardist
    if obs_key not in adata.obs.keys():
        destripe(adata)
    if not os.path.exists(gex_save_path):
        grid_image(adata, obs_key, log1p=log1p,
                mpp=mpp, sigma=sigma, save_path=gex_save_path)
    else:
        print(f"gex_save_path {gex_save_path} already exists, skipping grid_image")
    cellseg(image_path=gex_save_path, 
             labels_npz_path=gex_save_path.replace(".tiff", ".npz"), 
             stardist_model="2D_versatile_fluo", 
             prob_thresh=prob_thresh, 
             nms_thresh=nms_thresh,gpu=gpu,
             **kwargs,
            )
    insert_labels(adata, 
                  labels_npz_path=gex_save_path.replace(".tiff", ".npz"), 
                  basis="spatial", 
                  spatial_key=f"spatial_cropped_{buffer}_buffer",
                  mpp=mpp, 
                  labels_key="labels_gex"
                 )
    return adata
    
@register_function(
    aliases=["挂失次级标签", "salvage_secondary_labels", "rescue_labels", "标签救救", "次级标签恢复"],
    category="space",
    description="Salvage secondary labels by combining primary and secondary segmentation results",
    examples=[
        "# Basic label salvaging",
        "ov.space.salvage_secondary_labels(adata, primary_label='labels_he',",
        "                                   secondary_label='labels_gex')",
        "# Custom label keys",
        "ov.space.salvage_secondary_labels(adata, primary_label='cellpose_he',",
        "                                   secondary_label='cellpose_gex',",
        "                                   labels_key='combined_labels')",
        "# Access combined labels",
        "joint_labels = adata.obs['labels_joint']"
    ],
    related=["space.visium_10x_hd_cellpose_he", "space.visium_10x_hd_cellpose_gex", "space.bin2cell"]
)
def salvage_secondary_labels(
        adata,
        primary_label="labels_he", 
        secondary_label="labels_gex",
        labels_key="labels_joint",
        **kwargs,
):
    """Merge primary and secondary segmentation labels.

    Parameters
    ----------
    adata : AnnData
        Visium HD AnnData with multiple label layers.
    primary_label : str, default='labels_he'
        Primary segmentation label key.
    secondary_label : str, default='labels_gex'
        Secondary segmentation label key.
    labels_key : str, default='labels_joint'
        Output merged-label key.
    **kwargs
        Reserved keyword arguments.

    Returns
    -------
    AnnData
        The same object after updating merged labels in place.
    """
    from ..external.bin2cell import salvage_secondary_labels
    salvage_secondary_labels(adata, primary_label=primary_label,
                             secondary_label=secondary_label,
                             labels_key=labels_key)
    return adata
    
    
    
@register_function(
    aliases=["空间段到细胞", "bin2cell", "bin_to_cell", "空间细胞化", "段级到细胞级"],
    category="space",
    description="Convert spatial bins to single-cell resolution using segmentation labels",
    examples=[
        "# Basic bin to cell conversion",
        "adata_cell = ov.space.bin2cell(adata, labels_key='labels_joint')",
        "# Custom spatial keys",
        "adata_cell = ov.space.bin2cell(adata, labels_key='segmentation',",
        "                               spatial_keys=['spatial'])",
        "# With diameter scaling",
        "adata_cell = ov.space.bin2cell(adata, labels_key='cellpose_labels',",
        "                               diameter_scale_factor=1.5)"
    ],
    related=["space.visium_10x_hd_cellpose_he", "space.visium_10x_hd_cellpose_gex"]
)
def bin2cell(
        adata,
        labels_key="labels_joint",
        spatial_keys=None,
        diameter_scale_factor=None,
        add_geometry: bool = True,
        geometry_key: str = "geometry",
        geometry_spatial_key: str = "spatial",
        geometry_force_polygon: bool = False,
        rename_obs_to_cellid: bool = True,
        show_progress: bool = True,
):
    """Aggregate binned Visium signals into cell-level profiles.

    Parameters
    ----------
    adata : AnnData
        Spatial bin-level AnnData.
    labels_key : str, default='labels_joint'
        Label key assigning bins to cells.
    spatial_keys : list or None, default=None
        Spatial coordinate keys to aggregate.
    diameter_scale_factor : float, optional
        Optional scaling factor for estimated cell diameters.
    add_geometry : bool, default=True
        Whether to generate polygon geometry from labeled bins and store
        WKT strings in ``obs[geometry_key]`` of the returned cell-level AnnData.
    geometry_key : str, default='geometry'
        Observation column name used to store generated geometry WKT.
    geometry_spatial_key : str, default='spatial'
        Coordinate key in ``adata.obsm`` used to reconstruct polygons.
    geometry_force_polygon : bool, default=False
        If ``True``, convert ``MultiPolygon`` geometries to their largest
        polygon component so each cell gets a single polygon contour.
    rename_obs_to_cellid : bool, default=True
        If ``True``, rename output ``obs_names`` to ``cellid_XXXXXXXXX-1``
        using ``obs['object_id']`` and also write ``obs['cellid']``.
    show_progress : bool, default=True
        Whether to display progress bars during aggregation and (if enabled)
        geometry reconstruction.

    Returns
    -------
    AnnData
        Cell-level AnnData generated from labeled bins.
    """
    from ..external.bin2cell import bin_to_cell

    if spatial_keys is None:
        spatial_keys = ["spatial"]
    cell_adata = bin_to_cell(
        adata,
        labels_key=labels_key,
        spatial_keys=spatial_keys,
        diameter_scale_factor=diameter_scale_factor,
        show_progress=show_progress,
    )

    if add_geometry:
        import os
        import warnings
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import pandas as pd
        try:
            import shapely
        except ImportError as exc:
            raise ImportError(
                "`bin2cell(add_geometry=True)` requires Shapely. Install it "
                "with `pip install shapely`."
            ) from exc
        from shapely.errors import GEOSException
        try:
            from tqdm.auto import tqdm
        except Exception:
            def tqdm(x, **kwargs):
                return x

        if labels_key not in adata.obs.columns:
            raise ValueError(f"`labels_key='{labels_key}'` not found in `adata.obs`.")
        if geometry_spatial_key not in adata.obsm:
            raise ValueError(f"`geometry_spatial_key='{geometry_spatial_key}'` not found in `adata.obsm`.")

        coords = np.asarray(adata.obsm[geometry_spatial_key])
        labels = pd.to_numeric(adata.obs[labels_key], errors="coerce").fillna(0).astype(np.int64).to_numpy()
        valid_mask = labels > 0
        if not np.any(valid_mask):
            warnings.warn("No positive labels found; skipping geometry generation.")
            cell_adata.obs[geometry_key] = ""
            return cell_adata

        x_vals = np.unique(coords[valid_mask, 0])
        y_vals = np.unique(coords[valid_mask, 1])
        dx = np.diff(np.sort(x_vals))
        dy = np.diff(np.sort(y_vals))
        step_candidates = np.concatenate([dx[dx > 0], dy[dy > 0]])
        bin_size = float(np.median(step_candidates)) if step_candidates.size else 1.0
        half = max(bin_size * 0.5, 0.5)

        labels_valid = labels[valid_mask]
        coords_valid = coords[valid_mask]

        # Prefer exact bin grid geometry when array coordinates are available.
        rows_valid = None
        cols_valid = None
        if "array_row" in adata.obs.columns and "array_col" in adata.obs.columns:
            rows_raw = pd.to_numeric(adata.obs["array_row"], errors="coerce").to_numpy()[valid_mask]
            cols_raw = pd.to_numeric(adata.obs["array_col"], errors="coerce").to_numpy()[valid_mask]
            finite_mask = np.isfinite(rows_raw) & np.isfinite(cols_raw)
            if np.any(finite_mask):
                labels_valid = labels_valid[finite_mask]
                coords_valid = coords_valid[finite_mask]
                rows_valid = rows_raw[finite_mask].astype(np.int64, copy=False)
                cols_valid = cols_raw[finite_mask].astype(np.int64, copy=False)
            else:
                warnings.warn(
                    "array_row/array_col exist but contain no finite values; "
                    "falling back to center-based bin boxes."
                )

        # Group bins by label once to avoid O(n_labels * n_bins) boolean scans.
        order = np.argsort(labels_valid, kind="mergesort")
        labels_sorted = labels_valid[order]
        coords_sorted = coords_valid[order]
        if rows_valid is not None and cols_valid is not None:
            rows_sorted = rows_valid[order]
            cols_sorted = cols_valid[order]
        unique_labels, starts, counts = np.unique(
            labels_sorted, return_index=True, return_counts=True
        )

        boxes = None
        if rows_valid is not None and cols_valid is not None:
            try:
                row_to_y = (
                    pd.DataFrame({"row": rows_sorted, "y": coords_sorted[:, 1]})
                    .groupby("row", sort=True)["y"]
                    .median()
                )
                col_to_x = (
                    pd.DataFrame({"col": cols_sorted, "x": coords_sorted[:, 0]})
                    .groupby("col", sort=True)["x"]
                    .median()
                )

                def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
                    centers = np.asarray(centers, dtype=np.float64)
                    if centers.size == 1:
                        return np.array([centers[0] - half, centers[0] + half], dtype=np.float64)
                    mids = (centers[:-1] + centers[1:]) * 0.5
                    first = centers[0] - (mids[0] - centers[0])
                    last = centers[-1] + (centers[-1] - mids[-1])
                    return np.concatenate(([first], mids, [last]))

                row_vals = row_to_y.index.to_numpy()
                col_vals = col_to_x.index.to_numpy()
                y_edges = _centers_to_edges(row_to_y.to_numpy())
                x_edges = _centers_to_edges(col_to_x.to_numpy())

                row_idx = np.searchsorted(row_vals, rows_sorted)
                col_idx = np.searchsorted(col_vals, cols_sorted)

                x0 = x_edges[col_idx]
                x1 = x_edges[col_idx + 1]
                y0 = y_edges[row_idx]
                y1 = y_edges[row_idx + 1]

                boxes = shapely.box(
                    np.minimum(x0, x1),
                    np.minimum(y0, y1),
                    np.maximum(x0, x1),
                    np.maximum(y0, y1),
                )
            except Exception as exc:
                warnings.warn(
                    "Failed to construct array-grid bin polygons; "
                    f"falling back to center-based boxes. Error: {exc}"
                )

        if boxes is None:
            boxes = shapely.box(
                coords_sorted[:, 0] - half,
                coords_sorted[:, 1] - half,
                coords_sorted[:, 0] + half,
                coords_sorted[:, 1] + half,
            )

        def _build_geometry(lab, start, count):
            part = boxes[start:start + count]
            try:
                # Faster for grid-cell coverages (touching but non-overlapping polygons).
                geom = shapely.coverage_union_all(part)
            except GEOSException:
                # Fallback for unexpected overlaps/invalids.
                geom = shapely.union_all(part)
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geometry_force_polygon and geom.geom_type == "MultiPolygon":
                try:
                    geom = max(geom.geoms, key=lambda g: g.area)
                except Exception:
                    pass
            return str(int(lab)), "" if geom.is_empty else geom.wkt

        geometry_map = {}
        tasks = list(zip(unique_labels, starts, counts))
        max_workers = min(len(tasks), max(1, os.cpu_count() or 1))
        if max_workers <= 1:
            label_iter = tasks
            if show_progress:
                label_iter = tqdm(
                    label_iter,
                    total=len(tasks),
                    desc="bin2cell geometry",
                    leave=False,
                )
            for lab, start, count in label_iter:
                key, wkt = _build_geometry(lab, start, count)
                geometry_map[key] = wkt
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_build_geometry, lab, start, count) for lab, start, count in tasks]
                done_iter = as_completed(futures)
                if show_progress:
                    done_iter = tqdm(
                        done_iter,
                        total=len(futures),
                        desc="bin2cell geometry",
                        leave=False,
                    )
                for future in done_iter:
                    key, wkt = future.result()
                    geometry_map[key] = wkt

        object_ids = cell_adata.obs["object_id"].astype(np.int64).astype(str)
        cell_adata.obs[geometry_key] = object_ids.map(geometry_map).fillna("")

        if rename_obs_to_cellid:
            object_id_vals = pd.to_numeric(cell_adata.obs["object_id"], errors="coerce")
            if np.all(np.isfinite(object_id_vals)) and np.all(object_id_vals > 0):
                cellid_vals = object_id_vals.astype(np.int64).map(
                    lambda x: f"cellid_{x:09d}-1"
                )
                if len(set(cellid_vals.tolist())) == len(cellid_vals):
                    cell_adata.obs["cellid"] = cellid_vals.to_numpy()
                    cell_adata.obs_names = cellid_vals.to_numpy()

        cell_adata.uns.setdefault("omicverse_io", {})
        cell_adata.uns["omicverse_io"]["type"] = "bin2cell_seg"

    return cell_adata


@register_function(
    aliases=[
        "同步VisiumHD分割几何",
        "sync_visium_hd_seg_geometries",
        "sync_hd_seg_geometries",
    ],
    category="space",
    description="Sync Visium HD segmentation geometries in adata.uns['spatial'] after AnnData subsetting",
    examples=[
        "# Subset and sync segmentation geometries",
        "adata_sub = adata[adata.obs['classification'] == 'Cluster-1'].copy()",
        "ov.space.sync_visium_hd_seg_geometries(adata_sub, sample='sample1')",
    ],
    related=["io.spatial.read_visium_hd"],
)
def sync_visium_hd_seg_geometries(adata, sample=None):
    """
    Synchronize ``adata.uns["spatial"][sample]["geometries"]`` with current ``adata.obs_names``.

    This utility is intended for AnnData objects loaded by ``read_visium_hd_seg``.
    After subsetting AnnData, ``adata.obs`` and ``adata.obsm`` are subset automatically,
    but the geometry table stored in ``adata.uns["spatial"][sample]["geometries"]`` may
    still contain rows for cells that are no longer present. This function filters that
    GeoDataFrame by current cell IDs and updates it in place.

    Parameters
    ----------
    adata : sc.AnnData
        A (possibly subsetted) AnnData object containing spatial segmentation metadata.
    sample : str, optional
        Sample key under ``adata.uns["spatial"]``. If ``None``, the first available key
        is used (with a warning when multiple samples exist).

    Returns
    -------
    sc.AnnData
        The same ``adata`` object, modified in place.
    """
    import warnings
    import geopandas as gpd

    if "spatial" not in adata.uns:
        warnings.warn(
            "No spatial information found in adata.uns['spatial']. "
            "This function is intended for data loaded with read_visium_hd_seg()."
        )
        return adata

    if sample is None:
        available_samples = list(adata.uns["spatial"].keys())
        if len(available_samples) == 0:
            warnings.warn("No samples found in adata.uns['spatial'].")
            return adata
        sample = available_samples[0]
        if len(available_samples) > 1:
            warnings.warn(
                f"Multiple samples found: {available_samples}. "
                f"Using '{sample}'. Specify `sample` explicitly to use a different one."
            )

    if sample not in adata.uns["spatial"]:
        warnings.warn(f"Sample '{sample}' not found in adata.uns['spatial'].")
        return adata

    spatial_info = adata.uns["spatial"][sample]
    if "geometries" not in spatial_info:
        warnings.warn(
            f"No geometries found in adata.uns['spatial']['{sample}']['geometries']. "
            "This function is intended for data loaded with read_visium_hd_seg()."
        )
        return adata

    geometries = spatial_info["geometries"]
    current_cell_ids = set(adata.obs_names)

    if isinstance(geometries, gpd.GeoDataFrame):
        try:
            geometries_subset = geometries.loc[geometries.index.isin(current_cell_ids)]
        except (KeyError, IndexError) as e:
            common_indices = geometries.index.intersection(current_cell_ids)
            if len(common_indices) > 0:
                geometries_subset = geometries.loc[common_indices]
            else:
                warnings.warn(
                    f"Could not filter geometries by index. "
                    f"Geometries may not be properly indexed by cell IDs. "
                    f"Error: {e}"
                )
                return adata
        spatial_info["geometries"] = geometries_subset
    else:
        warnings.warn(
            f"Unexpected geometry format in adata.uns['spatial']['{sample}']['geometries']. "
            "Expected GeoDataFrame."
        )

    return adata


def write_visium_hd_cellseg(adata, path, sample=None):
    """Export cell-level AnnData to SpaceRanger v4-compatible directory.

    Creates ``filtered_feature_cell_matrix.h5``,
    ``graphclust_annotated_cell_segmentations.geojson``, and ``spatial/``
    with images and scalefactors.

    Parameters
    ----------
    adata : AnnData
        Cell-level AnnData from ``bin2cell()``.
    path : str or Path
        Output directory.
    sample : str, optional
        Sample key in ``adata.uns["spatial"]``.
    """
    from ..io.spatial import write_visium_hd_cellseg as _write
    _write(adata, path, sample=sample)

