# MIT License
# Copyright (c) 2026 Manh Huynh

import os
import ssl
import shutil
import urllib.request
import threading
import traceback
from typing import Optional

import bpy
from bpy.props import EnumProperty

MODEL_URLS = {
    'SMALL': "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx",
    'BASE': "https://huggingface.co/onnx-community/depth-anything-v2-base/resolve/main/onnx/model.onnx",
    'LARGE': "https://huggingface.co/onnx-community/depth-anything-v2-large/resolve/main/onnx/model.onnx",
}

MODEL_HW_NOTES = {
    'SMALL': "~98MB | RAM 4GB+ | Any GPU",
    'BASE': "~392MB | RAM 8GB+ | VRAM 4GB+",
    'LARGE': "~1.3GB | RAM 16GB+ | VRAM 6GB+ recommended",
}


def get_possible_model_dirs(create=False) -> list:
    """Return all possible model storage directories across Extension & standard Addon paths."""
    dirs = []

    # 1. Extension user data path (Blender 4.2+ extensions)
    try:
        if __package__:
            p = bpy.utils.extension_path_user(__package__, path="models", create=create)
            if p and p not in dirs:
                dirs.append(p)
    except Exception:
        pass

    # 2. Standard DATAFILES user path
    try:
        base_dir = os.path.join(bpy.utils.user_resource('DATAFILES'), "da2_pbr_generator", "models")
        if create and not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
        if base_dir not in dirs:
            dirs.append(base_dir)
    except Exception:
        pass

    # 3. Addon root folder fallback
    try:
        addon_dir = os.path.join(os.path.dirname(__file__), "models")
        if addon_dir not in dirs:
            dirs.append(addon_dir)
    except Exception:
        pass

    return dirs


def get_user_data_path(subfolder="models", create=True) -> str:
    """Get the primary user data directory for saving downloaded models."""
    dirs = get_possible_model_dirs(create=create)
    for d in dirs:
        if os.path.isdir(d):
            return d
    target = dirs[0] if dirs else os.path.expanduser("~")
    if create and target:
        os.makedirs(target, exist_ok=True)
    return target


def find_existing_model_file(filename: str, min_bytes: int = 10 * 1024 * 1024) -> Optional[str]:
    """Search for an existing model file across all candidate directories."""
    dirs = get_possible_model_dirs(create=False)
    for d in dirs:
        full_path = os.path.join(d, filename)
        if os.path.isfile(full_path):
            try:
                if os.path.getsize(full_path) >= min_bytes:
                    return full_path
            except Exception:
                pass
    return None


def force_ui_redraw(context):
    """Trigger redraw of 3D View and Preferences areas so UI immediately reflects model status changes."""
    try:
        if hasattr(context, "window_manager") and context.window_manager:
            for window in context.window_manager.windows:
                if window.screen:
                    for area in window.screen.areas:
                        if area.type in ('VIEW_3D', 'PREFERENCES'):
                            area.tag_redraw()
    except Exception:
        pass


def remove_all_downloaded_models():
    """Remove downloaded model files."""
    try:
        dirs = get_possible_model_dirs(create=False)
        for model_dir in dirs:
            if os.path.isdir(model_dir):
                shutil.rmtree(model_dir, ignore_errors=True)
    except Exception as e:
        print(f"[DA2 PBR Generator] Error removing models: {e}")


# ---------------------------------------------------------------------------
# Operator: Remove Model
# ---------------------------------------------------------------------------
class DA2_OT_remove_model(bpy.types.Operator):
    bl_idname = "da2.remove_model"
    bl_label = "Remove Downloaded Model"
    bl_description = "Delete the downloaded ONNX model file from disk to free up space"

    def execute(self, context):
        try:
            from .core.depth_estimator import DepthEstimator
            DepthEstimator.release()
        except Exception:
            pass

        prefs = context.preferences.addons[__package__].preferences
        path = prefs.get_model_path()

        removed = False
        if os.path.exists(path):
            try:
                os.remove(path)
                removed = True
            except Exception as e:
                self.report({'ERROR'}, f"Failed to remove model: {e}")
                return {'CANCELLED'}

        force_ui_redraw(context)
        if removed:
            self.report({'INFO'}, "Downloaded model removed successfully.")
        else:
            self.report({'WARNING'}, "No downloaded model file found.")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Operator: Download Model (non-blocking modal with background thread)
# ---------------------------------------------------------------------------
class DA2_OT_download_model(bpy.types.Operator):
    bl_idname = "da2.download_model"
    bl_label = "Download DA2 Model Weights"
    bl_description = "Download Depth Anything V2 ONNX model weights from HuggingFace"

    _timer = None
    _thread = None
    _download_progress = 0.0
    _download_done = False
    _download_error = None
    _status_msg = ""

    def modal(self, context, event):
        wm = context.window_manager

        if context.area:
            context.area.tag_redraw()

        if event.type == 'TIMER':
            wm.da2_progress = self._download_progress

            if self._download_done:
                self._cleanup_timer(context)
                wm.da2_progress = 0.0
                force_ui_redraw(context)

                if self._download_error:
                    self.report({'ERROR'}, f"Download failed: {self._download_error}")
                    return {'CANCELLED'}
                else:
                    self.report({'INFO'}, "Model downloaded successfully!")
                    return {'FINISHED'}

        return {'PASS_THROUGH'}

    def _cleanup_timer(self, context):
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)
            self._timer = None

    def execute(self, context):
        return self.invoke(context, None)

    def invoke(self, context, event):
        if hasattr(bpy.app, 'online_access') and not bpy.app.online_access:
            self.report({'ERROR'}, "Please enable 'Allow Online Access' in Blender Preferences > System.")
            return {'CANCELLED'}

        prefs = context.preferences.addons[__package__].preferences
        variant = prefs.model_variant
        url = MODEL_URLS.get(variant, MODEL_URLS['SMALL'])

        model_dir = get_user_data_path(subfolder="models", create=True)
        dest_path = os.path.join(model_dir, f"depth_anything_v2_{variant.lower()}.onnx")

        self._download_done = False
        self._download_error = None
        self._download_progress = 0.0
        self._status_msg = "Connecting to HuggingFace..."

        def _download_worker():
            try:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE

                req = urllib.request.Request(
                    url,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'},
                )

                with urllib.request.urlopen(req, context=ctx, timeout=60) as response:
                    total_size = int(response.info().get('Content-Length', 0))
                    downloaded = 0
                    block_size = 256 * 1024

                    temp_path = dest_path + ".tmp"
                    with open(temp_path, 'wb') as f:
                        while True:
                            buffer = response.read(block_size)
                            if not buffer:
                                break
                            f.write(buffer)
                            downloaded += len(buffer)
                            if total_size > 0:
                                self._download_progress = (downloaded / total_size) * 100.0
                                self._status_msg = (
                                    f"Downloading: {self._download_progress:.1f}% "
                                    f"({downloaded // (1024 * 1024)}MB / {total_size // (1024 * 1024)}MB)"
                                )
                            else:
                                self._status_msg = f"Downloading: {downloaded // (1024 * 1024)}MB"

                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    os.rename(temp_path, dest_path)
                    self._download_progress = 100.0

            except Exception as e:
                self._download_error = str(e)
                traceback.print_exc()
            finally:
                self._download_done = True

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.2, window=context.window)
        wm.modal_handler_add(self)

        self._thread = threading.Thread(target=_download_worker, daemon=True)
        self._thread.start()

        return {'RUNNING_MODAL'}


# ---------------------------------------------------------------------------
# AddonPreferences
# ---------------------------------------------------------------------------
class DA2Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    def _on_variant_change(self, context):
        force_ui_redraw(context)

    model_variant: EnumProperty(
        name="Model Variant",
        items=[
            ('SMALL', "DA2 Small (~98MB)", "Lightweight and fast, recommended for most tasks"),
            ('BASE', "DA2 Base (~392MB)", "Sharper details, requires more VRAM/RAM"),
            ('LARGE', "DA2 Large (~1.3GB)", "Highest quality, requires 16GB+ RAM and 6GB+ VRAM"),
        ],
        default='SMALL',
        update=_on_variant_change,
    )

    device: EnumProperty(
        name="Inference Device",
        items=[
            ('AUTO', "Auto (CUDA if available)", "Automatically use CUDA GPU if available, fallback to CPU"),
            ('CUDA', "GPU (CUDA)", "Force using NVIDIA GPU via CUDA"),
            ('CPU', "CPU Only", "Force using CPU execution"),
        ],
        default='AUTO',
    )

    def get_model_path(self) -> str:
        filename = f"depth_anything_v2_{self.model_variant.lower()}.onnx"
        existing = find_existing_model_file(filename, min_bytes=10 * 1024 * 1024)
        if existing:
            return existing
        model_dir = get_user_data_path(subfolder="models", create=False)
        return os.path.join(model_dir, filename)

    def is_model_downloaded(self) -> bool:
        path = self.get_model_path()
        return os.path.exists(path) and os.path.getsize(path) > 10 * 1024 * 1024

    def draw(self, context):
        layout = self.layout
        wm = context.window_manager

        box = layout.box()
        box.label(text="AI Model Management (Depth Anything V2)", icon='MODIFIER')
        box.prop(self, "model_variant")
        box.prop(self, "device")

        is_ready = self.is_model_downloaded()

        row = box.row()
        if is_ready:
            model_path = self.get_model_path()
            row.label(text=f"Model Ready ({os.path.basename(model_path)})", icon='CHECKMARK')
            row.operator("da2.remove_model", text="Remove Model", icon='TRASH')
        else:
            row.label(text="ONNX Weights Not Downloaded", icon='ERROR')

        col = box.column(align=True)
        btn_row = col.row()
        btn_row.operator("da2.download_model", text="Download Model Now", icon='IMPORT')

        if 0.0 < wm.da2_progress < 100.0:
            col.prop(wm, "da2_progress", text="Downloading Progress", slider=True)

        if hasattr(bpy.app, 'online_access') and not bpy.app.online_access:
            btn_row.enabled = False
            col.label(text="Please enable 'Allow Online Access' in Preferences > System to download.", icon='INFO')

        hw_note = MODEL_HW_NOTES.get(self.model_variant, "")
        if hw_note:
            box.label(text=f"Requirements: {hw_note}", icon='SYSTEM')

        # Credits & License Box
        box_cred = layout.box()
        box_cred.label(text="Credits & License", icon='HELP')
        col_cred = box_cred.column(align=True)
        col_cred.label(text="Addon: DA2 + PBR texture Generator from Image")
        col_cred.label(text="Author: Manh Huynh")
        col_cred.label(text="License: MIT License (Open Source)")
        col_cred.label(text="Core AI Model: Depth Anything V2 (HKU / ByteDance)")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def register():
    bpy.utils.register_class(DA2_OT_download_model)
    bpy.utils.register_class(DA2_OT_remove_model)
    bpy.utils.register_class(DA2Preferences)


def unregister():
    try:
        from .core.depth_estimator import DepthEstimator
        DepthEstimator.release()
    except Exception:
        pass

    bpy.utils.unregister_class(DA2Preferences)
    bpy.utils.unregister_class(DA2_OT_remove_model)
    bpy.utils.unregister_class(DA2_OT_download_model)
