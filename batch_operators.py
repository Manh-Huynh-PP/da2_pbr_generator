import os
import time
import threading
import traceback

import bpy
from .core.depth_estimator import DepthEstimator
from .core.normal_map import generate_hybrid_normal
from .core.roughness_metallic import generate_roughness_metallic
from .core.albedo_map import generate_albedo_map
from .core.utils import (load_image_as_numpy, collect_image_files,
                          depth_to_16bit_png, normal_to_png,
                          depth_to_exr, normal_to_exr,
                          resolve_output_path)


class DA2_OT_batch_process(bpy.types.Operator):
    """Non-blocking batch operator: processes images in a background thread
    one-at-a-time and writes results per-image. The modal timer only checks
    the thread status — it never runs inference on the main thread."""

    bl_idname = "da2.batch_process"
    bl_label = "Start Batch Processing"
    bl_description = "Process folder batch, sequence, or video frames without blocking the UI"

    _timer = None
    _thread = None
    _cancel_flag = False
    _finished = False
    _error_msg = None
    _status_msg = ""
    _current_idx = 0
    _total = 0

    # ------------------------------------------------------------------
    # Modal: only reads progress from the worker thread
    # ------------------------------------------------------------------
    def modal(self, context, event):
        wm = context.window_manager

        if context.area:
            context.area.tag_redraw()

        if event.type == 'ESC':
            self._cancel_flag = True
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)
            self._cleanup(context)
            wm.da2_status = "Cancelled."
            wm.da2_progress = 0.0
            self.report({'WARNING'}, "Batch processing cancelled by user.")
            return {'CANCELLED'}

        if event.type == 'TIMER':
            # Push status to UI
            if self._total > 0:
                wm.da2_progress = (self._current_idx / self._total) * 100.0
            wm.da2_status = self._status_msg

            if self._finished:
                self._cleanup(context)
                if self._error_msg:
                    wm.da2_status = f"Error: {self._error_msg}"
                    wm.da2_progress = 0.0
                    self.report({'ERROR'}, f"Batch error: {self._error_msg}")
                    return {'CANCELLED'}
                wm.da2_progress = 100.0
                wm.da2_status = f"Done! Processed {self._total} items."
                self.report({'INFO'}, f"Successfully processed all {self._total} items!")
                return {'FINISHED'}

        return {'PASS_THROUGH'}

    # ------------------------------------------------------------------
    # Invoke: validate inputs, then spawn the background worker thread
    # ------------------------------------------------------------------
    def invoke(self, context, event):
        props = context.scene.da2_props
        prefs = context.preferences.addons[__package__].preferences

        if not prefs.is_model_downloaded():
            self.report({'ERROR'}, "ONNX model weights not downloaded. Click 'Download Model Now' in Addon Preferences.")
            return {'CANCELLED'}

        mode = props.input_mode
        files = []
        out_dir = ""

        if mode == 'BATCH':
            if not props.input_directory or not os.path.isdir(props.input_directory):
                self.report({'ERROR'}, "Invalid input folder path.")
                return {'CANCELLED'}
            files = collect_image_files(props.input_directory)
            out_dir = props.output_directory if props.output_directory else os.path.join(props.input_directory, "depth_output")

        elif mode == 'SEQUENCE':
            if not props.input_directory or not os.path.isdir(props.input_directory):
                self.report({'ERROR'}, "Invalid image sequence folder path.")
                return {'CANCELLED'}
            files = collect_image_files(props.input_directory)
            out_dir = props.output_directory if props.output_directory else os.path.join(props.input_directory, "depth_sequence")

        elif mode == 'VIDEO':
            video_path = props.input_filepath
            if not video_path or not os.path.exists(video_path):
                self.report({'ERROR'}, "Invalid video input file path.")
                return {'CANCELLED'}
            video_dir = os.path.dirname(video_path)
            out_dir = props.output_directory if props.output_directory else os.path.join(video_dir, "video_depth_output")

        if mode != 'VIDEO' and not files:
            self.report({'ERROR'}, "No input images found to process.")
            return {'CANCELLED'}

        os.makedirs(out_dir, exist_ok=True)

        # Gather parameters (avoid bpy inside thread)
        model_path = prefs.get_model_path()
        use_gpu = prefs.device in ('AUTO', 'CUDA')
        apply_plane_fix = props.apply_plane_fix
        generate_albedo = props.generate_albedo
        generate_normal = props.generate_normal
        generate_roughness = props.generate_roughness
        generate_metallic = props.generate_metallic
        depth_strength = props.depth_strength
        color_detail_strength = props.color_detail_strength
        enhance_mode = props.enhance_mode
        guided_radius = props.guided_radius
        guided_epsilon = props.guided_epsilon
        sharpen = props.sharpen_depth
        sharpen_strength = props.sharpen_strength
        smooth_depth = props.smooth_depth
        smooth_radius = props.smooth_radius
        delight_strength = props.delight_strength
        delight_saturation = props.delight_saturation
        roughness_offset = props.roughness_offset
        roughness_cavity = props.roughness_cavity
        roughness_texture = props.roughness_texture
        metallic_sensitivity = props.metallic_sensitivity
        metallic_shadow_fill = props.metallic_shadow_fill
        metallic_pattern_boost = props.metallic_pattern_boost
        out_fmt = props.output_format
        out_dir_setting = props.output_directory
        video_path_for_worker = props.input_filepath if mode == 'VIDEO' else None

        # Reset state
        self._cancel_flag = False
        self._finished = False
        self._error_msg = None
        self._status_msg = "Initializing..."
        self._current_idx = 0
        self._total = len(files) if files else 0

        # ---- Background worker (NO bpy calls) ----
        def _worker():
            try:
                nonlocal files

                # Load model
                self._status_msg = "Loading AI model..."
                t0 = time.perf_counter()
                DepthEstimator.load_model(model_path, use_gpu=use_gpu)
                dt = time.perf_counter() - t0
                self._status_msg = f"Model loaded ({dt:.1f}s)."

                # Video: extract frames
                if video_path_for_worker:
                    self._status_msg = "Extracting video frames..."
                    files = self._extract_video_frames(video_path_for_worker)
                    if not files:
                        self._error_msg = "No frames extracted. Ensure opencv-python is available."
                        return
                    self._total = len(files)
                    self._status_msg = f"Extracted {len(files)} frames."

                # Process each file
                t_batch = time.perf_counter()
                for i, filepath in enumerate(files):
                    if self._cancel_flag:
                        self._status_msg = f"Cancelled at {i}/{self._total}."
                        break

                    fname = os.path.basename(filepath)
                    self._status_msg = f"[{i + 1}/{self._total}] {fname}"

                    try:
                        rgb = load_image_as_numpy(filepath)
                        depth = DepthEstimator.estimate(
                            rgb,
                            apply_plane_fix=apply_plane_fix,
                            enhance_mode=enhance_mode,
                            guided_radius=guided_radius,
                            guided_epsilon=guided_epsilon,
                            sharpen=sharpen,
                            sharpen_strength=sharpen_strength,
                            smooth_depth=smooth_depth,
                            smooth_radius=smooth_radius,
                        )

                        filename = os.path.splitext(fname)[0]
                        ext = 'exr' if out_fmt == 'EXR' else 'png'

                        depth_path = resolve_output_path(
                            output_setting=out_dir_setting,
                            input_filepath=filepath,
                            default_filename=filename,
                            suffix="_depth",
                            ext=ext
                        )
                        if out_fmt == 'EXR':
                            depth_to_exr(depth, depth_path)
                        else:
                            depth_to_16bit_png(depth, depth_path)

                        if generate_albedo:
                            albedo = generate_albedo_map(
                                rgb, depth,
                                delight_strength=delight_strength,
                                delight_saturation=delight_saturation,
                            )
                            albedo_path = resolve_output_path(
                                output_setting=out_dir_setting,
                                input_filepath=filepath,
                                default_filename=filename,
                                suffix="_albedo",
                                ext=ext
                            )
                            if out_fmt == 'EXR':
                                normal_to_exr(albedo, albedo_path)
                            else:
                                normal_to_png((np.clip(albedo, 0.0, 1.0) * 255).astype(np.uint8), albedo_path)

                        if generate_normal:
                            normal = generate_hybrid_normal(
                                depth, rgb,
                                depth_strength=depth_strength,
                                color_detail_strength=color_detail_strength,
                            )
                            normal_path = resolve_output_path(
                                output_setting=out_dir_setting,
                                input_filepath=filepath,
                                default_filename=filename,
                                suffix="_normal",
                                ext=ext
                            )
                            if out_fmt == 'EXR':
                                normal_to_exr(normal, normal_path)
                            else:
                                normal_to_png(normal, normal_path)

                        if generate_roughness or generate_metallic:
                            r_map, m_map = generate_roughness_metallic(
                                rgb, depth,
                                roughness_offset=roughness_offset,
                                roughness_cavity=roughness_cavity,
                                roughness_texture=roughness_texture,
                                metallic_sensitivity=metallic_sensitivity,
                                metallic_shadow_fill=metallic_shadow_fill,
                                metallic_pattern_boost=metallic_pattern_boost,
                            )

                            if generate_roughness:
                                rough_path = resolve_output_path(
                                    output_setting=out_dir_setting,
                                    input_filepath=filepath,
                                    default_filename=filename,
                                    suffix="_roughness",
                                    ext=ext
                                )
                                if out_fmt == 'EXR':
                                    depth_to_exr(r_map, rough_path)
                                else:
                                    depth_to_16bit_png(r_map, rough_path)

                            if generate_metallic:
                                metal_path = resolve_output_path(
                                    output_setting=out_dir_setting,
                                    input_filepath=filepath,
                                    default_filename=filename,
                                    suffix="_metallic",
                                    ext=ext
                                )
                                if out_fmt == 'EXR':
                                    depth_to_exr(m_map, metal_path)
                                else:
                                    depth_to_16bit_png(m_map, metal_path)
                    except Exception as e:
                        print(f"[DA2 Batch] Error processing {filepath}: {e}")

                    self._current_idx = i + 1

                    # Estimate remaining time
                    elapsed = time.perf_counter() - t_batch
                    if self._current_idx > 0:
                        avg = elapsed / self._current_idx
                        remaining = avg * (self._total - self._current_idx)
                        if remaining > 60:
                            self._status_msg = f"[{self._current_idx}/{self._total}] ~{remaining / 60:.1f}min left"
                        else:
                            self._status_msg = f"[{self._current_idx}/{self._total}] ~{remaining:.0f}s left"

            except Exception as e:
                self._error_msg = str(e)
                traceback.print_exc()
            finally:
                self._finished = True

        wm = context.window_manager
        wm.da2_progress = 0.0
        wm.da2_status = "Starting batch..."
        self._timer = wm.event_timer_add(0.15, window=context.window)
        wm.modal_handler_add(self)

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

        self.report({'INFO'}, "Batch processing started in background...")
        return {'RUNNING_MODAL'}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_video_frames(video_path: str) -> list[str]:
        """Extract all frames from a video file. Runs in worker thread only."""
        try:
            import cv2
        except ImportError:
            print("[DA2 Batch] opencv-python (cv2) not available for video decoding.")
            return []

        temp_dir = os.path.join(os.path.dirname(video_path), "_da2_temp_frames")
        os.makedirs(temp_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        extracted = []
        idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_path = os.path.join(temp_dir, f"frame_{idx:06d}.png")
                cv2.imwrite(frame_path, frame)
                extracted.append(frame_path)
                idx += 1
        finally:
            cap.release()

        return extracted

    def _cleanup(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None


def register():
    bpy.utils.register_class(DA2_OT_batch_process)


def unregister():
    bpy.utils.unregister_class(DA2_OT_batch_process)
