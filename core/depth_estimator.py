import numpy as np
from PIL import Image
from .plane_fitting import remove_plane_bias
from .guided_filter import guided_upscale

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 518  # DA2 standard input size


class DepthEstimator:
    """Singleton ONNX inference session for Depth Anything V2."""

    _session = None
    _loaded_path = None

    @classmethod
    def load_model(cls, model_path: str, use_gpu: bool = True):
        """Load ONNX model. Cache session if path has not changed."""
        import onnxruntime as ort

        if cls._session is not None and cls._loaded_path == model_path:
            return

        providers = (
            ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if use_gpu
            else ['CPUExecutionProvider']
        )
        cls._session = ort.InferenceSession(model_path, providers=providers)
        cls._loaded_path = model_path

    @classmethod
    def release(cls):
        """Release the cached ONNX session to free memory and unlock DLL files."""
        cls._session = None
        cls._loaded_path = None

    @classmethod
    def estimate(cls, image_rgb: np.ndarray,
                 apply_plane_fix: bool = True,
                 enhance_mode: str = 'NONE',
                 guided_radius: int = 16,
                 guided_epsilon: float = 0.02,
                 sharpen: bool = False,
                 sharpen_strength: float = 0.5,
                 smooth_depth: bool = True,
                 smooth_radius: int = 3) -> np.ndarray:
        """
        Input:  numpy (H, W, 3) float32 RGB [0,1]
        Output: numpy (H, W) float32 normalized [0,1]

        Pipeline:
        1. Resize -> 518x518
        2. Normalize ImageNet mean/std
        3. ONNX inference -> raw depth (518x518)
        4. Upscale to original resolution:
           - NONE: Bilinear interpolation
           - GUIDED: Guided Filter using RGB edges (recommended)
        5. Plane Fitting subtraction (optional)
        6. Unsharp Mask sharpening (optional)
        7. Min-Max normalize -> [0,1]
        """
        if cls._session is None:
            raise RuntimeError("Model not loaded. Please download the model in Addon Preferences first.")

        orig_h, orig_w = image_rgb.shape[:2]

        # --- Preprocessing ---
        pil_img = Image.fromarray((image_rgb * 255).astype(np.uint8)).resize(
            (INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR
        )
        resized_arr = np.array(pil_img, dtype=np.float32) / 255.0

        normalized = (resized_arr - IMAGENET_MEAN) / IMAGENET_STD
        tensor = normalized.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

        # --- Inference ---
        input_name = cls._session.get_inputs()[0].name
        raw_depth = cls._session.run(None, {input_name: tensor})[0]
        raw_depth = np.squeeze(raw_depth)  # (518, 518)

        # --- Upscale to original resolution ---
        if enhance_mode == 'GUIDED' and (orig_w > INPUT_SIZE or orig_h > INPUT_SIZE):
            # Edge-Guided Upsampling: use original RGB as edge guide
            depth = guided_upscale(
                raw_depth, image_rgb,
                radius=guided_radius, eps=guided_epsilon,
            )
        else:
            # Standard bilinear upscale
            depth_pil = Image.fromarray(raw_depth).resize(
                (orig_w, orig_h), Image.Resampling.BILINEAR
            )
            depth = np.array(depth_pil, dtype=np.float32)

        # --- Plane Fitting ---
        if apply_plane_fix:
            depth = remove_plane_bias(depth)
        else:
            d_min, d_max = depth.min(), depth.max()
            rng = d_max - d_min
            depth = (depth - d_min) / rng if rng > 1e-6 else np.full_like(depth, 0.5)

        # --- Unsharp Mask Sharpening ---
        if sharpen and sharpen_strength > 0:
            depth = _unsharp_mask(depth, strength=sharpen_strength)

        # --- Smooth Depth Edges (reduce displacement artifacts) ---
        if smooth_depth and smooth_radius > 0:
            depth = _smooth_depth_edges(depth, radius=smooth_radius)

        return depth


def _unsharp_mask(depth: np.ndarray, strength: float = 0.5,
                  sigma_pixels: int = 5) -> np.ndarray:
    """Apply unsharp mask to enhance depth edges.

    sharpened = depth + strength * (depth - blurred)
    Then re-normalize to [0, 1].
    """
    from .guided_filter import _box_filter

    ksize = sigma_pixels * 2 + 1
    blurred = _box_filter(depth, ksize)
    detail = depth - blurred
    sharpened = depth + strength * detail

    # Re-normalize to [0, 1]
    d_min, d_max = sharpened.min(), sharpened.max()
    rng = d_max - d_min
    if rng > 1e-6:
        return ((sharpened - d_min) / rng).astype(np.float32)
    return np.full_like(depth, 0.5)


def _smooth_depth_edges(depth: np.ndarray, radius: int = 3) -> np.ndarray:
    """Smooth extreme depth gradients to prevent displacement artifacts.

    Identifies pixels with steep gradients (outliers) and blends them with
    a locally smoothed version. Preserves flat/gentle areas untouched.
    """
    from .guided_filter import _box_filter

    ksize = radius * 2 + 1
    smoothed = _box_filter(depth, ksize)

    # Compute gradient magnitude (Sobel-like)
    gy = np.abs(np.diff(depth, axis=0, prepend=depth[:1, :]))
    gx = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Find gradient threshold: pixels above 95th percentile are extreme
    threshold = np.percentile(grad_mag, 95)
    if threshold < 1e-6:
        return depth

    # Blend: extreme gradient areas use smoothed, rest use original
    # Soft blend to avoid hard transitions
    blend = np.clip((grad_mag - threshold * 0.5) / (threshold * 0.5 + 1e-6), 0.0, 1.0)
    result = depth * (1.0 - blend) + smoothed * blend

    return result.astype(np.float32)
