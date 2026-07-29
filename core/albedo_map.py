"""Generate Albedo (Base Color / Delighted) Map from RGB Color Image + DA2 Depth.

Delighting Algorithm (De-shading & De-AO):
  1. Estimate Ambient Occlusion (AO) from depth map.
  2. Estimate Surface Shading (directional lighting) from surface normals/depth gradient.
  3. Combine AO and Shading to form an Illumination Map (Luminance of baked light/shadow).
  4. Delight: Albedo = RGB / Illumination (with edge-preserving guided filter).
  5. Retain original color saturation and hue while neutralizing shadows and specular hot-spots.

Returns:
  albedo_rgb: (H, W, 3) float32 [0, 1]
"""
import numpy as np
from .guided_filter import _box_filter, guided_filter


def generate_albedo_map(rgb: np.ndarray, depth: np.ndarray,
                        delight_strength: float = 0.70,
                        delight_saturation: float = 0.30) -> np.ndarray:
    """Generate delighting Albedo (Base Color) Map from RGB and Depth.

    Args:
        rgb: (H, W, 3) float32 [0, 1] original color image.
        depth: (H, W) float32 [0, 1] depth map.
        delight_strength: float [0, 1] strength of shadow/AO removal.
        delight_saturation: float [0, 1] color saturation recovery strength.

    Returns:
        (H, W, 3) float32 [0, 1] delighting Albedo map.
    """
    h, w = depth.shape
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    # --- 1. Compute Ambient Occlusion (AO) from Depth ---
    ksize = 19
    depth_blur = _box_filter(depth, ksize)
    depth_recess = np.clip((depth_blur - depth) * 5.0, 0.0, 1.0)
    ao_illumination = 1.0 - depth_recess * 0.5

    # --- 2. Compute Directional Surface Shading from Depth Gradient ---
    gy = (np.roll(depth, -1, axis=0) - np.roll(depth, 1, axis=0)) * 2.0
    gx = (np.roll(depth, -1, axis=1) - np.roll(depth, 1, axis=1)) * 2.0
    shading_raw = np.clip(1.0 - (gx * 0.4 - gy * 0.4), 0.5, 1.5)

    # --- 3. Combined Raw Illumination Field ---
    illumination_raw = ao_illumination * shading_raw
    illumination_raw = np.clip(illumination_raw, 0.3, 1.5).astype(np.float32)

    illumination_guided = guided_filter(guide=lum, src=illumination_raw, radius=16, eps=0.01)
    illumination_guided = np.clip(illumination_guided, 0.4, 1.4).astype(np.float32)

    # --- 4. Delighting Operation: Albedo = RGB / Illumination ---
    effective_illum = 1.0 + (illumination_guided - 1.0) * delight_strength

    albedo = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        albedo[..., c] = rgb[..., c] / (effective_illum + 1e-6)

    albedo = np.clip(albedo, 0.0, 1.0)

    # --- 5. Saturation & Tone Recovery ---
    if delight_saturation > 0:
        c_max = np.max(albedo, axis=-1, keepdims=True)
        c_min = np.min(albedo, axis=-1, keepdims=True)
        sat = np.where(c_max > 1e-6, (c_max - c_min) / (c_max + 1e-6), 0.0)

        desat_mask = np.clip((1.0 - effective_illum) * delight_saturation, 0.0, 0.5)
        mean_color = np.mean(albedo, axis=-1, keepdims=True)
        albedo = albedo + (albedo - mean_color) * desat_mask[..., None]

    return np.clip(albedo, 0.0, 1.0).astype(np.float32)
