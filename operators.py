import os
import time
import threading
import traceback

import bpy
import mathutils
from .core.depth_estimator import DepthEstimator
from .core.clipseg_material import CLIPSegMaterial
from .core.normal_map import generate_hybrid_normal
from .core.roughness_metallic import generate_roughness_metallic
from .core.albedo_map import generate_albedo_map
from .core.utils import (load_image_as_numpy, numpy_to_blender_image,
                          depth_to_16bit_png, normal_to_png,
                          depth_to_exr, normal_to_exr,
                          resolve_output_path)


def _create_pbr_plane(name: str, width_px: int, height_px: int,
                      color_img: bpy.types.Image,
                      depth_img: bpy.types.Image,
                      normal_img: bpy.types.Image = None,
                      roughness_img: bpy.types.Image = None,
                      metallic_img: bpy.types.Image = None,
                      albedo_img: bpy.types.Image = None,
                      pixels_per_meter: float = 1000.0,
                      displacement_method: str = 'MODIFIER'):
    """Create a plane sized to match the input image and assign a full PBR
    material with Color/Albedo, Normal, Roughness, Metallic, and Displacement."""
    plane_w = width_px / pixels_per_meter
    plane_h = height_px / pixels_per_meter

    # Create mesh plane
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
    plane_obj = bpy.context.active_object
    plane_obj.name = f"DA2_{name}"
    plane_obj.scale = (plane_w, plane_h, 1.0)
    bpy.ops.object.transform_apply(scale=True)

    # Subdivide mesh geometry for displacement
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=64)
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.shade_smooth()

    # Add Subdivision Surface modifier (Simple) for displacement detail
    subsurf = plane_obj.modifiers.new(name="Subsurf_Displacement", type='SUBSURF')
    subsurf.subdivision_type = 'SIMPLE'
    subsurf.levels = 2
    subsurf.render_levels = 4

    # --- Create PBR Material ---
    mat = bpy.data.materials.new(name=f"DA2_PBR_{name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_output = nodes.new('ShaderNodeOutputMaterial')
    node_output.location = (600, 0)

    node_bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    node_bsdf.location = (200, 0)
    links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])

    # Base Color (Albedo Map if generated, else original Color Map)
    target_color_img = albedo_img if albedo_img is not None else color_img
    node_color_tex = nodes.new('ShaderNodeTexImage')
    node_color_tex.location = (-400, 200)
    node_color_tex.image = target_color_img
    node_color_tex.label = "Albedo Map" if albedo_img is not None else "Color Map"
    node_color_tex.image.colorspace_settings.name = 'sRGB'
    links.new(node_color_tex.outputs['Color'], node_bsdf.inputs['Base Color'])

    # --- Displacement ---
    if displacement_method == 'MATERIAL':
        node_depth_tex = nodes.new('ShaderNodeTexImage')
        node_depth_tex.location = (-400, -200)
        node_depth_tex.image = depth_img
        node_depth_tex.label = "Depth Map"
        node_depth_tex.image.colorspace_settings.name = 'Non-Color'

        node_disp = nodes.new('ShaderNodeDisplacement')
        node_disp.location = (200, -300)
        node_disp.inputs['Scale'].default_value = 0.03
        node_disp.inputs['Midlevel'].default_value = 0.5
        links.new(node_depth_tex.outputs['Color'], node_disp.inputs['Height'])
        links.new(node_disp.outputs['Displacement'], node_output.inputs['Displacement'])

        if hasattr(mat, 'displacement_method'):
            mat.displacement_method = 'BOTH'
        elif hasattr(mat, 'cycles') and hasattr(mat.cycles, 'displacement_method'):
            mat.cycles.displacement_method = 'BOTH'

    else:
        tex_name = f"DA2_DispTex_{name}"
        if tex_name in bpy.data.textures:
            bpy.data.textures.remove(bpy.data.textures[tex_name])
        tex = bpy.data.textures.new(tex_name, type='IMAGE')
        tex.image = depth_img
        tex.image.colorspace_settings.name = 'Non-Color'

        disp_mod = plane_obj.modifiers.new(name="DA2_Displace", type='DISPLACE')
        disp_mod.texture = tex
        disp_mod.texture_coords = 'UV'
        disp_mod.direction = 'NORMAL'
        disp_mod.strength = 0.03
        disp_mod.mid_level = 0.5

    # Normal Map
    if normal_img is not None:
        node_normal_tex = nodes.new('ShaderNodeTexImage')
        node_normal_tex.location = (-400, -600)
        node_normal_tex.image = normal_img
        node_normal_tex.label = "Normal Map"
        node_normal_tex.image.colorspace_settings.name = 'Non-Color'

        node_normal_map = nodes.new('ShaderNodeNormalMap')
        node_normal_map.location = (-100, -500)
        node_normal_map.inputs['Strength'].default_value = 1.0
        links.new(node_normal_tex.outputs['Color'], node_normal_map.inputs['Color'])
        links.new(node_normal_map.outputs['Normal'], node_bsdf.inputs['Normal'])

    # Roughness Map
    if roughness_img is not None:
        node_rough_tex = nodes.new('ShaderNodeTexImage')
        node_rough_tex.location = (-400, 0)
        node_rough_tex.image = roughness_img
        node_rough_tex.label = "Roughness Map"
        node_rough_tex.image.colorspace_settings.name = 'Non-Color'
        links.new(node_rough_tex.outputs['Color'], node_bsdf.inputs['Roughness'])

    # Metallic Map
    if metallic_img is not None:
        node_metal_tex = nodes.new('ShaderNodeTexImage')
        node_metal_tex.location = (-400, -400)
        node_metal_tex.image = metallic_img
        node_metal_tex.label = "Metallic Map"
        node_metal_tex.image.colorspace_settings.name = 'Non-Color'
        links.new(node_metal_tex.outputs['Color'], node_bsdf.inputs['Metallic'])

    plane_obj.data.materials.append(mat)

    return plane_obj


class DA2_OT_generate_depth(bpy.types.Operator):
    """Non-blocking operator: runs ONNX inference in a background thread,
    reports progress via modal timer, and never blocks the Blender UI."""

    bl_idname = "da2.generate_depth"
    bl_label = "Generate Depth & Normal Map"
    bl_description = "Generate depth map and normal map for the selected image (non-blocking)"

    _timer = None
    _thread = None
    _finished = False
    _error_msg = None
    _status_msg = ""
    _progress = 0.0
    _result_depth = None
    _result_normal = None
    _result_rgb = None
    _result_filename = ""
    _img_width = 0
    _img_height = 0

    # ------------------------------------------------------------------
    # Modal loop
    # ------------------------------------------------------------------
    def modal(self, context, event):
        wm = context.window_manager

        if context.area:
            context.area.tag_redraw()

        if event.type == 'TIMER':
            wm.da2_progress = self._progress
            wm.da2_status = self._status_msg

            if self._finished:
                self._cleanup_timer(context)
                if self._error_msg:
                    wm.da2_status = f"Error: {self._error_msg}"
                    wm.da2_progress = 0.0
                    self.report({'ERROR'}, f"Processing error: {self._error_msg}")
                    return {'CANCELLED'}

                # --- Write results on main thread (bpy calls) ---
                try:
                    wm.da2_status = "Writing results to Blender..."
                    props = context.scene.da2_props
                    fname = self._result_filename
                    out_fmt = props.output_format  # 'PNG' or 'EXR'

                    # Load color image into Blender
                    color_bl_img = numpy_to_blender_image(self._result_rgb, f"Color_{fname}")

                    # Load depth into Blender
                    depth_bl_img = numpy_to_blender_image(self._result_depth, f"Depth_{fname}")

                    # Save to disk
                    if props.save_to_disk:
                        ext = 'exr' if out_fmt == 'EXR' else 'png'
                        depth_path = resolve_output_path(
                            output_setting=props.output_directory,
                            input_filepath=props.input_filepath,
                            default_filename=fname,
                            suffix="_depth",
                            ext=ext
                        )
                        if out_fmt == 'EXR':
                            depth_to_exr(self._result_depth, depth_path)
                        else:
                            depth_to_16bit_png(self._result_depth, depth_path)
                        self.report({'INFO'}, f"Saved Depth Map: {depth_path}")

                    # Normal map
                    normal_bl_img = None
                    if self._result_normal is not None:
                        normal_bl_img = numpy_to_blender_image(self._result_normal, f"Normal_{fname}")
                        if props.save_to_disk:
                            ext = 'exr' if out_fmt == 'EXR' else 'png'
                            normal_path = resolve_output_path(
                                output_setting=props.output_directory,
                                input_filepath=props.input_filepath,
                                default_filename=fname,
                                suffix="_normal",
                                ext=ext
                            )
                            if out_fmt == 'EXR':
                                normal_to_exr(self._result_normal, normal_path)
                            else:
                                normal_to_png(self._result_normal, normal_path)

                    # Roughness map
                    roughness_bl_img = None
                    if self._result_roughness is not None:
                        roughness_bl_img = numpy_to_blender_image(self._result_roughness, f"Roughness_{fname}")
                        if props.save_to_disk:
                            ext = 'exr' if out_fmt == 'EXR' else 'png'
                            rough_path = resolve_output_path(
                                output_setting=props.output_directory,
                                input_filepath=props.input_filepath,
                                default_filename=fname,
                                suffix="_roughness",
                                ext=ext
                            )
                            if out_fmt == 'EXR':
                                depth_to_exr(self._result_roughness, rough_path)
                            else:
                                depth_to_16bit_png(self._result_roughness, rough_path)

                    # Metallic map
                    metallic_bl_img = None
                    if self._result_metallic is not None:
                        metallic_bl_img = numpy_to_blender_image(self._result_metallic, f"Metallic_{fname}")
                        if props.save_to_disk:
                            ext = 'exr' if out_fmt == 'EXR' else 'png'
                            metal_path = resolve_output_path(
                                output_setting=props.output_directory,
                                input_filepath=props.input_filepath,
                                default_filename=fname,
                                suffix="_metallic",
                                ext=ext
                            )
                            if out_fmt == 'EXR':
                                depth_to_exr(self._result_metallic, metal_path)
                            else:
                                depth_to_16bit_png(self._result_metallic, metal_path)

                    # Albedo map (Base Color delighted)
                    albedo_bl_img = None
                    if self._result_albedo is not None:
                        albedo_bl_img = numpy_to_blender_image(self._result_albedo, f"Albedo_{fname}")
                        if props.save_to_disk:
                            ext = 'exr' if out_fmt == 'EXR' else 'png'
                            albedo_path = resolve_output_path(
                                output_setting=props.output_directory,
                                input_filepath=props.input_filepath,
                                default_filename=fname,
                                suffix="_albedo",
                                ext=ext
                            )
                            if out_fmt == 'EXR':
                                normal_to_exr(self._result_albedo, albedo_path)
                            else:
                                normal_to_png((np.clip(self._result_albedo, 0.0, 1.0) * 255).astype(np.uint8), albedo_path)
                            self.report({'INFO'}, f"Saved Albedo Map: {albedo_path}")

                    # Auto-create plane with PBR material
                    if props.auto_create_plane:
                        wm.da2_status = "Creating textured plane..."
                        _create_pbr_plane(
                            name=fname,
                            width_px=self._img_width,
                            height_px=self._img_height,
                            color_img=color_bl_img,
                            depth_img=depth_bl_img,
                            normal_img=normal_bl_img,
                            roughness_img=roughness_bl_img,
                            metallic_img=metallic_bl_img,
                            albedo_img=albedo_bl_img,
                            displacement_method=props.displacement_method,
                        )
                        self.report({'INFO'}, f"Created PBR plane: DA2_{fname}")

                    wm.da2_status = "Done!"
                    wm.da2_progress = 100.0
                    self.report({'INFO'}, "Successfully generated Depth Map & Normal Map!")
                except Exception as e:
                    wm.da2_status = f"Error: {e}"
                    wm.da2_progress = 0.0
                    self.report({'ERROR'}, f"Error writing results: {e}")
                    traceback.print_exc()
                    return {'CANCELLED'}

                return {'FINISHED'}

        return {'PASS_THROUGH'}

    # ------------------------------------------------------------------
    # Execute & Invoke
    # ------------------------------------------------------------------
    def execute(self, context):
        return self.invoke(context, None)

    def invoke(self, context, event):
        props = context.scene.da2_props
        prefs = context.preferences.addons[__package__].preferences

        if not prefs.is_model_downloaded():
            self.report({'ERROR'}, "ONNX model weights not downloaded. Click 'Download Model Now' in Addon Preferences.")
            return {'CANCELLED'}

        filepath = props.input_filepath
        if not filepath or not os.path.exists(filepath):
            self.report({'ERROR'}, "Invalid image filepath or file does not exist.")
            return {'CANCELLED'}

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
        input_basename = os.path.basename(filepath)

        # Reset state
        self._finished = False
        self._error_msg = None
        self._status_msg = "Initializing..."
        self._progress = 0.0
        self._result_depth = None
        self._result_normal = None
        self._result_roughness = None
        self._result_metallic = None
        self._result_albedo = None
        self._result_rgb = None
        self._result_filename = os.path.splitext(input_basename)[0]
        self._img_width = 0
        self._img_height = 0

        # ---- Background worker (NO bpy calls inside) ----
        def _worker():
            try:
                self._status_msg = "Loading AI model..."
                self._progress = 5.0
                t0 = time.perf_counter()
                DepthEstimator.load_model(model_path, use_gpu=use_gpu)
                dt_model = time.perf_counter() - t0

                self._status_msg = f"Model loaded ({dt_model:.1f}s). Reading image..."
                self._progress = 20.0
                rgb = load_image_as_numpy(filepath)
                h, w = rgb.shape[:2]
                self._img_width = w
                self._img_height = h
                self._result_rgb = rgb

                mode_label = "Edge-Guided" if enhance_mode == 'GUIDED' else "Standard"
                self._status_msg = f"Estimating depth [{mode_label}] ({w}x{h})..."
                self._progress = 30.0
                t1 = time.perf_counter()
                self._result_depth = DepthEstimator.estimate(
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
                dt_depth = time.perf_counter() - t1

                self._status_msg = f"Depth done ({dt_depth:.1f}s)."
                self._progress = 70.0

                if generate_albedo:
                    self._status_msg = "Generating Albedo (Base Color) map..."
                    self._progress = 73.0
                    self._result_albedo = generate_albedo_map(
                        rgb, self._result_depth,
                        delight_strength=delight_strength,
                        delight_saturation=delight_saturation,
                    )

                if generate_normal:
                    self._status_msg = "Generating normal map..."
                    self._progress = 75.0
                    self._result_normal = generate_hybrid_normal(
                        self._result_depth, rgb,
                        depth_strength=depth_strength,
                        color_detail_strength=color_detail_strength,
                    )
                    self._status_msg = "Normal map done."

                if generate_roughness or generate_metallic:
                    self._status_msg = "Calculating Roughness & Metallic maps..."
                    self._progress = 85.0
                    r_map, m_map = generate_roughness_metallic(
                        rgb, self._result_depth,
                        roughness_offset=roughness_offset,
                        roughness_cavity=roughness_cavity,
                        roughness_texture=roughness_texture,
                        metallic_sensitivity=metallic_sensitivity,
                        metallic_shadow_fill=metallic_shadow_fill,
                        metallic_pattern_boost=metallic_pattern_boost,
                    )

                    if generate_roughness:
                        self._result_roughness = r_map
                    if generate_metallic:
                        self._result_metallic = m_map

                self._progress = 95.0

            except Exception as e:
                self._error_msg = str(e)
                traceback.print_exc()
            finally:
                self._finished = True

        wm = context.window_manager
        wm.da2_progress = 0.0
        wm.da2_status = f"Processing: {input_basename}"
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

        self.report({'INFO'}, "Depth estimation started in background...")
        return {'RUNNING_MODAL'}

    def _cleanup_timer(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None


def register():
    bpy.utils.register_class(DA2_OT_generate_depth)


def unregister():
    bpy.utils.unregister_class(DA2_OT_generate_depth)
