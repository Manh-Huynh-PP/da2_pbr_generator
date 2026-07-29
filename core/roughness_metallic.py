"""Generate Roughness and Metallic maps from RGB + DA2 Depth — pure algorithmic.

Physics-based roughness estimation:
  - Crevices/cavities (deep in depth map) → ROUGH (white) — accumulate dirt, not polished
  - Peaks/exposed surfaces → SMOOTH (dark) — wear and polishing
  - High texture detail → ROUGH — complex micro-surface
  - Specular highlights (bright + low saturation) → SMOOTH — mirror-like reflection
  - High depth curvature → ROUGH — edges, corners, transitions

No AI model needed. Full resolution. Pure numpy.
"""
import numpy as np
from typing import Optional, Tuple, Dict
from .guided_filter import _box_filter, guided_filter


def _gaussian_blur_2d(src: np.ndarray, sigma: float) -> np.ndarray:
    """2D Gaussian blur using 3-pass box filter approximation (Central Limit Theorem)."""
    radius = max(1, int(sigma * 2.5))
    result = src.copy()
    for _ in range(3):
        result = _box_filter(result, radius=radius)
    return result


def _local_std(src: np.ndarray, radius: int = 5) -> np.ndarray:
    """Compute local standard deviation (texture variance) at each pixel.

    High local std = high-frequency texture = rough surface.
    Low local std = uniform/smooth surface = glossy.
    """
    ksize = 2 * radius + 1
    mean = _box_filter(src, ksize)
    mean_sq = _box_filter(src * src, ksize)
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(variance)


def _detect_specular(rgb: np.ndarray) -> np.ndarray:
    """Detect specular highlights — bright pixels with low color saturation.

    Specular highlights indicate smooth/glossy surface → low roughness.
    Returns: (H, W) float32 [0, 1], higher = more specular.
    """
    # Luminance
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    # Saturation (max - min across channels)
    c_max = np.max(rgb, axis=-1)
    c_min = np.min(rgb, axis=-1)
    sat = np.where(c_max > 1e-6, (c_max - c_min) / (c_max + 1e-6), 0.0)

    # Specular = bright AND low saturation (white/gray highlights)
    bright_mask = np.clip((lum - 0.6) / 0.3, 0.0, 1.0)  # ramp from 0.6 to 0.9
    low_sat_mask = np.clip(1.0 - sat * 3.0, 0.0, 1.0)    # low saturation
    specular = bright_mask * low_sat_mask

    # Smooth slightly to avoid noise
    specular = _gaussian_blur_2d(specular.astype(np.float32), sigma=2.0)
    return np.clip(specular, 0.0, 1.0).astype(np.float32)


def _detect_specular_peaks(rgb: np.ndarray, radius: int = 15) -> np.ndarray:
    """Detect true local specular peaks — bright spots that stand out from surrounding background.

    A flat white wall/paper is bright everywhere (no local peak).
    A specular highlight on a metal surface is locally much brighter than its surroundings.
    """
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    c_max = np.max(rgb, axis=-1)
    c_min = np.min(rgb, axis=-1)
    sat = np.where(c_max > 1e-6, (c_max - c_min) / (c_max + 1e-6), 0.0)

    local_lum_mean = _box_filter(lum, radius=radius)
    specular_contrast = np.clip((lum - local_lum_mean) * 4.0, 0.0, 1.0)

    bright_mask = np.clip((lum - 0.50) / 0.30, 0.0, 1.0)
    low_sat_mask = np.clip(1.0 - sat * 3.0, 0.0, 1.0)

    return np.clip(bright_mask * low_sat_mask * specular_contrast, 0.0, 1.0).astype(np.float32)


def _compute_ambient_occlusion(depth: np.ndarray, radius: int = 9) -> np.ndarray:
    """Approximate screen-space ambient occlusion from depth map.

    Compares each pixel's depth to the local average.
    Pixels deeper than neighbors → occluded → rough.
    Pixels higher than neighbors → exposed → smooth.
    Returns: (H, W) float32 [0, 1], higher = more occluded = rougher.
    """
    ksize = 2 * radius + 1
    local_avg = _box_filter(depth, ksize)
    # How much this pixel is recessed relative to neighbors
    # Negative = pixel is deeper (in a crevice) → high AO → rough
    # Positive = pixel is elevated (on a peak) → low AO → smooth
    ao = np.clip((local_avg - depth) * 8.0 + 0.5, 0.0, 1.0)
    return ao.astype(np.float32)


def _compute_metallic_advanced(rgb: np.ndarray, depth: np.ndarray,
                                metallic_sensitivity: float = 0.50,
                                metallic_shadow_fill: float = 0.85,
                                metallic_pattern_boost: float = 0.90) -> np.ndarray:
    """Advanced Metallic Detection supporting Gold, Copper, Brass & Achromatic Metals, with White Diffuse Marble Suppression."""
    h, w = depth.shape
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    c_max = np.max(rgb, axis=-1)
    c_min = np.min(rgb, axis=-1)
    sat = np.where(c_max > 1e-6, (c_max - c_min) / (c_max + 1e-6), 0.0)

    # --- 1. Colored Metal Detection (Gold, Brass, Copper, Bronze) ---
    # Gold & Yellow-Brass: R > G, G > B, Saturation > 0.12, R > 0.35
    gold_hue_score = (
        np.clip((r - b) * 2.5, 0.0, 1.0) *
        np.clip((g - b) * 3.0, 0.0, 1.0) *
        np.clip(sat * 3.0, 0.0, 1.0)
    )
    gold_lum_score = np.clip((lum - 0.12) / 0.25, 0.0, 1.0)
    gold_metallic = gold_hue_score * gold_lum_score

    # Copper & Reddish-Bronze: R > G, R > B, Saturation > 0.18
    copper_hue_score = (
        np.clip((r - g) * 3.0, 0.0, 1.0) *
        np.clip((r - b) * 2.0, 0.0, 1.0) *
        np.clip(sat * 2.5, 0.0, 1.0)
    )
    copper_metallic = copper_hue_score * gold_lum_score

    warm_metal_score = np.maximum(gold_metallic * 1.25, copper_metallic * 1.25)

    # --- 2. Achromatic Metal Detection (Silver, Chrome, Steel, Iron) ---
    achromatic_mask = np.clip(1.0 - sat * 3.5, 0.0, 1.0)
    specular_peaks = _detect_specular_peaks(rgb, radius=15)
    luminance_gate = np.clip((lum - 0.20) / (0.20 + (1.0 - metallic_sensitivity) * 0.3), 0.0, 1.0)

    # Achromatic metals require specular highlights or environment reflection peaks
    chrome_score = achromatic_mask * luminance_gate * specular_peaks

    # --- 3. Diffuse Non-Metal Penalty (Marble, Ceramic, Paper, Plaster, Wall) ---
    # Non-metallic white surfaces have uniform high luminance with low local specular contrast and low saturation.
    lum_mean_large = _box_filter(lum, radius=15)
    lum_std_large = _local_std(lum, radius=15)

    diffuse_white_flatness = (
        np.clip((lum_mean_large - 0.45) / 0.35, 0.0, 1.0) *
        np.clip(1.0 - lum_std_large * 8.0, 0.0, 1.0) *
        np.clip(1.0 - sat * 3.5, 0.0, 1.0)
    )
    diffuse_white_penalty = np.clip(diffuse_white_flatness * (1.0 - specular_peaks * 0.90), 0.0, 1.0)

    # --- 4. Surface Normal Map & Structure Tensor Analysis ---
    dx = (np.roll(depth, -1, axis=1) - np.roll(depth, 1, axis=1)) * 3.5 + \
         (np.roll(lum, -1, axis=1) - np.roll(lum, 1, axis=1)) * 0.8
    dy = (np.roll(depth, -1, axis=0) - np.roll(depth, 1, axis=0)) * 3.5 + \
         (np.roll(lum, -1, axis=0) - np.roll(lum, 1, axis=0)) * 0.8

    j_xx = _box_filter(dx * dx, radius=5)
    j_yy = _box_filter(dy * dy, radius=5)
    j_xy = _box_filter(dx * dy, radius=5)

    trace = j_xx + j_yy
    det = j_xx * j_yy - j_xy * j_xy
    diff = np.sqrt(np.maximum(trace * trace * 0.25 - det, 0.0))
    lambda1 = trace * 0.5 + diff
    lambda2 = trace * 0.5 - diff

    anisotropy = np.clip((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-6), 0.0, 1.0)
    grad_norm = np.sqrt(dx * dx + dy * dy)
    normal_smoothness = np.exp(-_box_filter(grad_norm, radius=7) * 2.0)
    organic_noise = np.clip(_box_filter(lambda2, radius=7) * 4.0, 0.0, 1.0)

    # --- 5. Pattern & Score Combination ---
    brushed_pattern = anisotropy * luminance_gate * achromatic_mask
    base_metallic_color = np.maximum(chrome_score, warm_metal_score)

    polished_pattern = normal_smoothness * base_metallic_color
    pattern_boost = np.maximum(polished_pattern, brushed_pattern * metallic_pattern_boost)

    metallic_raw = np.maximum(base_metallic_color, pattern_boost)

    # Suppress non-metallic organic textures (wood/fabric/stone)
    metallic_raw = metallic_raw * (1.0 - organic_noise * 0.85)

    # Hard-suppress diffuse white non-metals (unless gold/copper or brushed metal)
    is_colored_metal = (warm_metal_score > 0.20)
    metallic_raw = np.where(is_colored_metal, metallic_raw, metallic_raw * (1.0 - diffuse_white_penalty * 0.98))

    # --- 6. Morphological Hole Filling for Shadow Gradients ---
    if metallic_shadow_fill > 0:
        metallic_dilated = _box_filter(metallic_raw, radius=21)
        shadow_fill_mask = (
            (metallic_dilated > 0.15) &
            (organic_noise < 0.45) &
            (lum > 0.05) &
            (diffuse_white_penalty < 0.50)
        )
        final_metallic = np.where(shadow_fill_mask, np.maximum(metallic_raw, metallic_dilated * metallic_shadow_fill), metallic_raw)
    else:
        final_metallic = metallic_raw

    final_metallic = _gaussian_blur_2d(final_metallic.astype(np.float32), sigma=1.5)
    return np.clip(final_metallic, 0.0, 1.0).astype(np.float32)


def generate_roughness_metallic(
    rgb: np.ndarray,
    depth: np.ndarray,
    labels: Optional[np.ndarray] = None,
    region_materials: Optional[Dict[int, dict]] = None,
    material_info: Optional[dict] = None,
    roughness_offset: float = 0.20,
    roughness_cavity: float = 0.30,
    roughness_texture: float = 0.25,
    metallic_sensitivity: float = 0.50,
    metallic_shadow_fill: float = 0.85,
    metallic_pattern_boost: float = 0.90,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Roughness and Metallic maps from RGB + DA2 Depth, optionally driven by AI MaterialClassifier."""
    h, w = depth.shape
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    if labels is not None and region_materials is not None:
        base_roughness = np.full((h, w), 0.50, dtype=np.float32)
        base_metallic = np.zeros((h, w), dtype=np.float32)

        for region_id, mat in region_materials.items():
            mask = (labels == region_id)
            base_roughness[mask] = mat["roughness"]
            base_metallic[mask] = mat["metallic"]

        base_roughness = _gaussian_blur_2d(base_roughness, sigma=3.0)
        base_metallic = _gaussian_blur_2d(base_metallic, sigma=1.5)
        base_metallic = np.where(base_metallic > 0.3, 0.95, 0.0).astype(np.float32)

        ao = _compute_ambient_occlusion(depth, radius=9)
        texture_var = _local_std(lum, radius=5)
        tv_norm = np.clip(texture_var / (np.percentile(texture_var, 95) + 1e-6), 0.0, 1.0)
        specular = _detect_specular(rgb)

        combined = (
            base_roughness * 0.65 +
            ao * roughness_cavity +
            tv_norm * roughness_texture +
            (1.0 - specular) * 0.08
        )
        final_roughness = guided_filter(guide=lum, src=combined.astype(np.float32), radius=8, eps=0.005)
        final_roughness = np.clip(final_roughness, 0.0, 1.0).astype(np.float32)

        final_metallic = guided_filter(guide=lum, src=base_metallic, radius=4, eps=0.02)
        final_metallic = np.clip(final_metallic, 0.0, 1.0).astype(np.float32)
        final_metallic[final_metallic < 0.15] = 0.0

        return final_roughness, final_metallic

    # --- Physics-based + AI Material Classifier ---
    ao = _compute_ambient_occlusion(depth, radius=9)

    gy = np.abs(np.diff(depth, axis=0, prepend=depth[:1, :]))
    gx = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
    grad_mag = np.sqrt(gx**2 + gy**2)
    p95_grad = np.percentile(grad_mag, 95)
    depth_gradient = np.clip(grad_mag / (p95_grad + 1e-6), 0.0, 1.0)

    smoothed = _box_filter(depth, 7)
    depth_curvature = np.clip(np.abs(depth - smoothed) * 6.0, 0.0, 1.0)

    texture_var = _local_std(lum, radius=5)
    tv_p95 = np.percentile(texture_var, 95)
    texture_detail = np.clip(texture_var / (tv_p95 + 1e-6), 0.0, 1.0)

    specular = _detect_specular(rgb)

    # Effective base roughness offset
    effective_base_roughness = roughness_offset
    metal_prob_gate = 1.0

    if material_info is not None:
        ai_base_r = material_info.get('base_roughness', roughness_offset)
        effective_base_roughness = ai_base_r * 0.7 + roughness_offset * 0.3
        metal_prob_gate = material_info.get('metal_prob', 1.0)

    roughness_raw = (
        ao * roughness_cavity +
        texture_detail * roughness_texture +
        depth_gradient * 0.15 +
        depth_curvature * 0.10 +
        effective_base_roughness
    )

    roughness_raw = roughness_raw * (1.0 - specular * 0.6)
    roughness_raw = np.clip(roughness_raw, 0.0, 1.0).astype(np.float32)

    final_roughness = guided_filter(guide=lum, src=roughness_raw, radius=8, eps=0.005)
    final_roughness = np.clip(final_roughness, 0.0, 1.0).astype(np.float32)

    # Metallic Calculation
    if metal_prob_gate < 0.15:
        # AI confirms non-metallic material (paper, wall, wood, plastic, ceramic, etc.)
        final_metallic = np.zeros((h, w), dtype=np.float32)
    else:
        raw_metallic = _compute_metallic_advanced(
            rgb, depth,
            metallic_sensitivity=metallic_sensitivity,
            metallic_shadow_fill=metallic_shadow_fill,
            metallic_pattern_boost=metallic_pattern_boost,
        )
        if metal_prob_gate < 0.60:
            # Scale down metallic confidence if AI detects low metal probability
            raw_metallic = raw_metallic * (metal_prob_gate / 0.60)

        final_metallic = guided_filter(guide=lum, src=raw_metallic, radius=4, eps=0.02)
        final_metallic = np.clip(final_metallic, 0.0, 1.0).astype(np.float32)
        final_metallic[final_metallic < 0.15] = 0.0

    return final_roughness, final_metallic
