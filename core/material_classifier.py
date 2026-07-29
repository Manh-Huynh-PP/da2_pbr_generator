"""Material Classifier module using ONNX inference with MINC-23 (Materials in Context) model.

Classifies 23 material categories from RGB images to drive physically-accurate
Roughness and Metallic map generation in Blender.
"""
import os
import numpy as np
from PIL import Image
from typing import Optional, Dict, Tuple

MINC_CLASSES = [
    'brick', 'carpet', 'ceramic', 'fabric', 'foliage', 'food', 'glass', 'hair',
    'leather', 'metal', 'mirror', 'other', 'painted', 'paper', 'plastic',
    'polishedstone', 'skin', 'sky', 'stone', 'tile', 'wallpaper', 'water', 'wood'
]

# Physical PBR Presets: (base_metallic, base_roughness) per material class
MATERIAL_PBR_PRESETS: Dict[str, Tuple[float, float]] = {
    'metal': (1.00, 0.25),
    'mirror': (1.00, 0.05),
    'paper': (0.00, 0.75),
    'wallpaper': (0.00, 0.75),
    'painted': (0.00, 0.50),
    'wood': (0.00, 0.55),
    'plastic': (0.00, 0.35),
    'ceramic': (0.00, 0.20),
    'tile': (0.00, 0.20),
    'polishedstone': (0.00, 0.18),
    'brick': (0.00, 0.85),
    'stone': (0.00, 0.80),
    'fabric': (0.00, 0.85),
    'carpet': (0.00, 0.90),
    'leather': (0.00, 0.50),
    'glass': (0.00, 0.10),
    'water': (0.00, 0.05),
    'food': (0.00, 0.60),
    'foliage': (0.00, 0.65),
    'skin': (0.00, 0.55),
    'hair': (0.00, 0.60),
    'sky': (0.00, 0.10),
    'other': (0.00, 0.50),
}


class MaterialClassifier:
    """Singleton ONNX inference session for MINC-23 Material Recognition."""

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
        """Release cached session to free memory."""
        cls._session = None
        cls._loaded_path = None

    @classmethod
    def predict_material(cls, image_rgb: np.ndarray) -> Dict[str, any]:
        """Predict material probabilities and PBR parameters from RGB image.

        Input: numpy (H, W, 3) float32 RGB [0, 1] or uint8 [0, 255]
        Output: Dict containing:
            - 'probs': Dict[str, float] probabilities for all 23 classes
            - 'top_class': str label of highest probability class
            - 'top_prob': float probability of top class
            - 'metal_prob': float probability of metal + mirror
            - 'base_metallic': float weighted base metallic score
            - 'base_roughness': float weighted base roughness score
        """
        if cls._session is None:
            raise RuntimeError("MaterialClassifier model is not loaded. Call load_model() first.")

        # Ensure float32 [0, 1]
        if image_rgb.dtype == np.uint8:
            rgb_float = image_rgb.astype(np.float32) / 255.0
        else:
            rgb_float = np.clip(image_rgb, 0.0, 1.0).astype(np.float32)

        # Convert to PIL Image for high-quality resize to 224x224
        img_pil = Image.fromarray((rgb_float * 255.0).astype(np.uint8))
        img_resized = img_pil.resize((224, 224), Image.Resampling.BILINEAR)
        arr_resized = np.array(img_resized, dtype=np.float32) / 255.0

        # SigLIP Normalization: (x - 0.5) / 0.5
        norm_img = (arr_resized - 0.5) / 0.5
        # Transpose (H, W, C) -> (1, C, H, W)
        tensor_in = np.transpose(norm_img, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        # Run ONNX Inference
        outputs = cls._session.run(None, {"pixel_values": tensor_in})
        logits = outputs[0][0]

        # Softmax over logits
        exp_logits = np.exp(logits - np.max(logits))
        probs_array = exp_logits / np.sum(exp_logits)

        probs_dict = {cls_name: float(p) for cls_name, p in zip(MINC_CLASSES, probs_array)}

        top_idx = int(np.argmax(probs_array))
        top_class = MINC_CLASSES[top_idx]
        top_prob = float(probs_array[top_idx])

        metal_prob = probs_dict.get('metal', 0.0) + probs_dict.get('mirror', 0.0)

        # Calculate weighted base PBR parameters
        weighted_metallic = 0.0
        weighted_roughness = 0.0
        for cls_name, p in probs_dict.items():
            preset_m, preset_r = MATERIAL_PBR_PRESETS.get(cls_name, (0.0, 0.50))
            weighted_metallic += p * preset_m
            weighted_roughness += p * preset_r

        return {
            'probs': probs_dict,
            'top_class': top_class,
            'top_prob': top_prob,
            'metal_prob': metal_prob,
            'base_metallic': float(weighted_metallic),
            'base_roughness': float(weighted_roughness),
        }
