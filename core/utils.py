import os
import glob
import numpy as np
from PIL import Image

def load_image_as_numpy(filepath: str) -> np.ndarray:
    """Load RGB/RGBA image file -> (H, W, 3) float32 RGB [0,1]."""
    img = Image.open(filepath).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0

def depth_to_16bit_png(depth: np.ndarray, output_path: str):
    """Float32 (H,W) [0,1] -> 16-bit grayscale PNG."""
    depth_clipped = np.clip(depth, 0.0, 1.0)
    arr_16 = (depth_clipped * 65535).astype(np.uint16)
    Image.fromarray(arr_16, mode='I;16').save(output_path)

def normal_to_png(normal: np.ndarray, output_path: str):
    """uint8 (H,W,3) normal map -> PNG."""
    Image.fromarray(normal).save(output_path)

def depth_to_exr(depth: np.ndarray, output_path: str):
    """Float32 (H,W) [0,1] -> 32-bit float EXR.

    Uses image.save() directly — no scene render settings needed.
    """
    import bpy

    h, w = depth.shape
    name = "__da2_tmp_depth_exr__"

    if name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[name])

    img = bpy.data.images.new(name, width=w, height=h, alpha=False, float_buffer=True)

    rgba = np.ones((h, w, 4), dtype=np.float32)
    val = np.clip(depth, 0.0, 1.0).astype(np.float32)
    rgba[..., 0] = val
    rgba[..., 1] = val
    rgba[..., 2] = val

    flipped = np.flipud(rgba).ravel()
    img.pixels.foreach_set(flipped)

    img.file_format = 'OPEN_EXR'
    img.filepath_raw = output_path
    img.save()
    bpy.data.images.remove(img)

def normal_to_exr(normal: np.ndarray, output_path: str):
    """Normal map -> 32-bit float EXR.

    Uses image.save() directly — no scene render settings needed.
    """
    import bpy

    h, w = normal.shape[:2]
    name = "__da2_tmp_normal_exr__"

    if name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[name])

    img = bpy.data.images.new(name, width=w, height=h, alpha=False, float_buffer=True)

    rgba = np.ones((h, w, 4), dtype=np.float32)
    if normal.dtype == np.uint8:
        rgba[..., :3] = normal.astype(np.float32) / 255.0
    else:
        rgba[..., :3] = normal.astype(np.float32)

    flipped = np.flipud(rgba).ravel()
    img.pixels.foreach_set(flipped)

    img.file_format = 'OPEN_EXR'
    img.filepath_raw = output_path
    img.save()
    bpy.data.images.remove(img)

def numpy_to_blender_image(array: np.ndarray, name: str):
    """
    Tạo hoặc cập nhật bpy.data.images từ numpy array.
    Hỗ trợ Grayscale (H,W) và RGB (H,W,3) uint8 hoặc float32.
    """
    import bpy

    h, w = array.shape[:2]
    
    # Check if image already exists in Blender data
    if name in bpy.data.images:
        image = bpy.data.images[name]
        img_w, img_h = image.size[0], image.size[1]
        if img_w != w or img_h != h:
            image.scale(w, h)
    else:
        image = bpy.data.images.new(name, width=w, height=h, alpha=False, float_buffer=True)
    
    rgba = np.ones((h, w, 4), dtype=np.float32)
    if array.ndim == 2:
        # Grayscale
        val = array.astype(np.float32)
        rgba[..., 0] = val
        rgba[..., 1] = val
        rgba[..., 2] = val
    else:
        # RGB
        if array.dtype == np.uint8:
            rgba[..., :3] = array.astype(np.float32) / 255.0
        else:
            rgba[..., :3] = array.astype(np.float32)
            
    # Blender images are bottom-up Y axis
    flipped = np.flipud(rgba).ravel()
    image.pixels.foreach_set(flipped)
    image.pack()
    image.update()
    return image

def collect_image_files(directory: str, extensions=('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp', '.exr')) -> list[str]:
    """Thu thập tất cả ảnh trong thư mục, sắp xếp theo tên."""
    if not os.path.exists(directory):
        return []
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        files.extend(glob.glob(os.path.join(directory, f"*{ext.upper()}")))
    return sorted(list(set(files)))


def resolve_output_path(output_setting: str, input_filepath: str, default_filename: str, suffix: str, ext: str) -> str:
    """
    Smartly resolve destination filepath for saving depth/normal/roughness/metallic maps.

    Handles:
    - Empty setting or setting == input_filepath -> saves in input directory with default_filename + suffix + ext.
    - Valid directory path -> saves inside that directory.
    - Text prefix (no slashes) -> saves in input directory with prefix added to filename.
    - Path + prefix (e.g. 'C:/Output/myprefix_') -> saves in 'C:/Output' with prefix 'myprefix_'.
    """
    setting = (output_setting or "").strip()
    input_dir = os.path.dirname(input_filepath) if input_filepath else ""
    input_name = os.path.basename(input_filepath) if input_filepath else ""

    if not setting or setting == input_filepath or setting == input_dir:
        out_dir = input_dir
        prefix = ""
    elif os.path.isdir(setting):
        out_dir = setting
        prefix = ""
    else:
        # Check if setting contains directory path components
        dir_part = os.path.dirname(setting)
        base_part = os.path.basename(setting)

        # If base_part matches input filename exactly, don't use as prefix
        if input_name and base_part == input_name:
            out_dir = dir_part if dir_part else input_dir
            prefix = ""
        elif dir_part and (os.path.exists(dir_part) or os.path.isabs(dir_part)):
            out_dir = dir_part
            prefix = base_part
        elif dir_part:
            out_dir = dir_part
            prefix = base_part
        else:
            # Plain text prefix without directory path
            out_dir = input_dir
            prefix = setting

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = "."

    clean_ext = ext.lstrip('.')
    final_filename = f"{prefix}{default_filename}{suffix}.{clean_ext}"
    return os.path.join(out_dir, final_filename)

