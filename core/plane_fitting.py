import numpy as np

def remove_plane_bias(depth: np.ndarray) -> np.ndarray:
    """
    Loại bỏ gradient bias (trên-xa, dưới-gần) bằng Plane Subtraction.
    
    1. Fit plane z = ax + by + c qua least squares
    2. Subtract plane từ depth
    3. Min-Max normalize kết quả về [0, 1]
    
    Port từ depth.ts solvePlane3x3() + estimateDepth()
    """
    h, w = depth.shape
    
    # Tạo grid coordinates [0, 1]
    ys, xs = np.mgrid[0:h, 0:w]
    x_norm = xs.astype(np.float64) / max(w - 1, 1)
    y_norm = ys.astype(np.float64) / max(h - 1, 1)
    
    z = depth.astype(np.float64).ravel()
    x_flat = x_norm.ravel()
    y_flat = y_norm.ravel()
    
    n = len(z)
    if n == 0:
        return depth
        
    # Least squares: z = ax + by + c
    A = np.array([
        [np.dot(x_flat, x_flat), np.dot(x_flat, y_flat), x_flat.sum()],
        [np.dot(x_flat, y_flat), np.dot(y_flat, y_flat), y_flat.sum()],
        [x_flat.sum(),           y_flat.sum(),           n           ]
    ], dtype=np.float64)
    
    B = np.array([np.dot(x_flat, z), np.dot(y_flat, z), z.sum()], dtype=np.float64)
    
    try:
        coeffs = np.linalg.lstsq(A, B, rcond=None)[0]
        a, b, c = coeffs
    except Exception:
        a, b, c = 0.0, 0.0, 0.0

    # Subtract plane
    plane = a * x_norm + b * y_norm + c
    detail = depth.astype(np.float64) - plane
    
    # Min-Max normalize -> [0, 1]
    d_min, d_max = detail.min(), detail.max()
    rng = d_max - d_min
    if rng > 1e-6:
        result = ((detail - d_min) / rng).astype(np.float32)
    else:
        result = np.full_like(depth, 0.5, dtype=np.float32)
        
    return result
