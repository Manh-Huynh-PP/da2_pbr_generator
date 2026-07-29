import numpy as np


def guided_filter(guide: np.ndarray, src: np.ndarray,
                  radius: int = 16, eps: float = 0.01) -> np.ndarray:
    """Edge-preserving guided filter using the original RGB image as guide.

    Uses box filtering (mean filter) for O(N) complexity regardless of radius.
    The guide image's edges steer the filtering so that depth edges align
    precisely with color edges in the original photo.

    Args:
        guide: (H, W) float32 grayscale guide image (luminance of RGB), [0,1].
        src: (H, W) float32 source image to filter (e.g. coarse depth), [0,1].
        radius: Window radius for box filter (kernel size = 2*radius + 1).
        eps: Regularization. Smaller = sharper edges, larger = smoother.

    Returns:
        (H, W) float32 filtered image.
    """
    ksize = 2 * radius + 1

    mean_I = _box_filter(guide, ksize)
    mean_p = _box_filter(src, ksize)
    mean_II = _box_filter(guide * guide, ksize)
    mean_Ip = _box_filter(guide * src, ksize)

    var_I = mean_II - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = _box_filter(a, ksize)
    mean_b = _box_filter(b, ksize)

    return mean_a * guide + mean_b


def guided_upscale(depth_low: np.ndarray, rgb_high: np.ndarray,
                   radius: int = 16, eps: float = 0.01) -> np.ndarray:
    """Upscale a low-resolution depth map using the high-res RGB as edge guide.

    Pipeline:
    1. Bilinear upscale depth_low → coarse_depth at rgb_high resolution
    2. Convert rgb_high → luminance guide
    3. Apply guided filter(guide=luminance, src=coarse_depth)

    Args:
        depth_low: (Hl, Wl) float32 raw depth from model inference.
        rgb_high: (Hh, Wh, 3) float32 original high-res RGB [0,1].
        radius: Guided filter radius.
        eps: Guided filter epsilon.

    Returns:
        (Hh, Wh) float32 refined depth map with sharp, edge-aligned boundaries.
    """
    from PIL import Image

    h_target, w_target = rgb_high.shape[:2]

    # Step 1: Bilinear upscale depth to target resolution
    depth_pil = Image.fromarray(depth_low)
    coarse = np.array(
        depth_pil.resize((w_target, h_target), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )

    # Step 2: RGB → luminance guide (BT.601)
    lum = (
        0.299 * rgb_high[..., 0]
        + 0.587 * rgb_high[..., 1]
        + 0.114 * rgb_high[..., 2]
    ).astype(np.float32)

    # Step 3: Guided filter
    refined = guided_filter(lum, coarse, radius=radius, eps=eps)

    return refined


def compute_joint_guide(
    rgb_high: np.ndarray,
    depth_map: np.ndarray = None,
    depth_weight: float = 0.5,
) -> np.ndarray:
    """Compute a multi-modal guide map combining RGB luminance and DA2 depth gradients.

    Args:
        rgb_high: (H, W, 3) float32 RGB [0, 1]
        depth_map: Optional (H, W) float32 depth [0, 1] from DA2
        depth_weight: Weight for depth gradients relative to RGB luminance [0, 1]

    Returns:
        (H, W) float32 joint guide map normalized [0, 1]
    """
    lum = (
        0.299 * rgb_high[..., 0]
        + 0.587 * rgb_high[..., 1]
        + 0.114 * rgb_high[..., 2]
    ).astype(np.float32)

    if depth_map is None:
        return lum

    # Resize depth map to match RGB image resolution if needed
    h, w = rgb_high.shape[:2]
    dh, dw = depth_map.shape[:2]
    if (dh, dw) != (h, w):
        from PIL import Image
        d_pil = Image.fromarray(depth_map.astype(np.float32), mode='F')
        depth_norm = np.array(d_pil.resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)
    else:
        depth_norm = depth_map.astype(np.float32)

    # Compute Depth Sobel Gradient magnitude to capture 3D geometric step edges
    gy = np.abs(np.diff(depth_norm, axis=0, prepend=depth_norm[:1, :]))
    gx = np.abs(np.diff(depth_norm, axis=1, prepend=depth_norm[:, :1]))
    d_grad = np.sqrt(gx * gx + gy * gy)

    # Normalize depth gradient
    max_g = d_grad.max()
    if max_g > 1e-6:
        d_grad = d_grad / max_g

    w_d = np.clip(depth_weight, 0.0, 1.0)
    w_rgb = 1.0 - w_d

    # Joint guide combines smooth depth map + depth gradient + RGB luminance
    joint = w_rgb * lum + w_d * (0.5 * depth_norm + 0.5 * d_grad)
    return np.clip(joint, 0.0, 1.0).astype(np.float32)



def _box_filter(img: np.ndarray, ksize: int = None, radius: int = None) -> np.ndarray:
    """Fast O(N) box filter using integral image (cumulative sum).

    Accepts either `ksize` (kernel width) or `radius` (window radius).
    Guarantees exact output dimensions matching input (H, W).
    """
    h, w = img.shape[:2]

    if radius is not None:
        ksize = 2 * radius + 1
    elif ksize is None:
        ksize = 5

    # Ensure kernel size is odd for symmetric padding
    if ksize % 2 == 0:
        ksize += 1

    pad = ksize // 2
    padded = np.pad(img, pad, mode='edge')

    # Axis 0 (rows)
    cum = np.cumsum(padded, axis=0)
    cum = np.vstack([np.zeros((1, cum.shape[1]), dtype=cum.dtype), cum])
    row_sum = cum[ksize:, :] - cum[:-ksize, :]

    # Axis 1 (cols)
    cum = np.cumsum(row_sum, axis=1)
    cum = np.hstack([np.zeros((cum.shape[0], 1), dtype=cum.dtype), cum])
    result = cum[:, ksize:] - cum[:, :-ksize]

    # Crop to exact original input shape (H, W)
    return (result[:h, :w] / (ksize * ksize)).astype(img.dtype)
