import numpy as np

def generate_hybrid_normal(depth: np.ndarray, color_rgb: np.ndarray,
                           depth_strength: float = 3.5,
                           color_detail_strength: float = 0.8) -> np.ndarray:
    """
    Hybrid Normal Map = Depth Gradient + Color Luminance Gradient.
    
    Input:
      depth: (H, W) float32 [0,1]
      color_rgb: (H, W, 3) float32 [0,1]
    Output:
      (H, W, 3) uint8 - tangent-space normal map (OpenGL convention)
      
    Port từ pbr.ts generateHybridNormalMap()
    """
    # Color -> Luminance (BT.601)
    lum = 0.299 * color_rgb[..., 0] + 0.587 * color_rgb[..., 1] + 0.114 * color_rgb[..., 2]
    
    # Sobel-like gradients (wrap edges cho seamless)
    dx_d = (np.roll(depth, -1, axis=1) - np.roll(depth, 1, axis=1)) * depth_strength
    dy_d = (np.roll(depth, -1, axis=0) - np.roll(depth, 1, axis=0)) * depth_strength
    
    dx_c = (np.roll(lum, -1, axis=1) - np.roll(lum, 1, axis=1)) * color_detail_strength
    dy_c = (np.roll(lum, -1, axis=0) - np.roll(lum, 1, axis=0)) * color_detail_strength
    
    dx = dx_d + dx_c
    dy = dy_d + dy_c
    dz = np.ones_like(dx)
    
    # Normalize vector
    length = np.sqrt(dx**2 + dy**2 + dz**2)
    length = np.maximum(length, 1e-8)
    
    nx = dx / length
    ny = dy / length
    nz = dz / length
    
    # Map [-1,1] -> [0,255]
    normal = np.stack([
        np.clip((nx * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8),
        np.clip((ny * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8),
        np.clip((nz * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8),
    ], axis=-1)
    
    return normal
