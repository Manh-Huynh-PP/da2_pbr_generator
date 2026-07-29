import os
import bpy


class DA2_PT_main_panel(bpy.types.Panel):
    bl_label = "DA2 DepthMap & PBR Generator"
    bl_idname = "DA2_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "DA2 Depth"

    def draw(self, context):
        layout = self.layout
        props = context.scene.da2_props
        wm = context.window_manager

        # Addon Preferences status check
        try:
            prefs = context.preferences.addons[__package__].preferences
        except KeyError:
            layout.label(text="Addon preferences not found.", icon='ERROR')
            return

        is_model_ready = prefs.is_model_downloaded()
        is_processing = 0.0 < wm.da2_progress < 100.0

        # Model Status Header
        if not is_model_ready:
            box_err = layout.box()
            box_err.label(text="AI Model Weights Not Downloaded!", icon='ERROR')
            box_err.operator("da2.download_model", text="Download AI Model", icon='IMPORT')
            layout.separator()

        # =========================================================================
        # STEP 1: Input Image & PBR Maps Settings
        # =========================================================================
        box_step1 = layout.box()
        box_step1.label(text="Step 1: Input & PBR Settings", icon='IMAGE_DATA')
        box_step1.prop(props, "input_mode", text="Mode")

        if props.input_mode == 'SINGLE':
            box_step1.prop(props, "input_filepath", text="Image")
        elif props.input_mode in ('BATCH', 'SEQUENCE'):
            box_step1.prop(props, "input_directory", text="Folder")
        elif props.input_mode == 'VIDEO':
            box_step1.prop(props, "input_filepath", text="Video")

        # Create Plane at the top (default active)
        if props.input_mode == 'SINGLE':
            box_step1.separator()
            box_step1.prop(props, "auto_create_plane")
            if props.auto_create_plane:
                box_step1.prop(props, "displacement_method")
            box_step1.separator()

        box_step1.prop(props, "apply_plane_fix")

        # PBR Maps Checkboxes with Collapsible Fine-Tuning Sliders
        col_maps = box_step1.column(align=True)
        col_maps.label(text="PBR Maps to Generate:", icon='TEXTURE')
        
        # Albedo Map
        row_a = col_maps.row(align=True)
        row_a.prop(props, "generate_albedo")
        if props.generate_albedo:
            icon_a = 'TRIA_DOWN' if props.show_albedo_tuning else 'TRIA_RIGHT'
            row_a.prop(props, "show_albedo_tuning", text="", icon=icon_a, icon_only=True, emboss=False)
        if props.generate_albedo and props.show_albedo_tuning:
            sub_a = col_maps.column(align=True)
            sub_a.prop(props, "delight_strength")
            sub_a.prop(props, "delight_saturation")

        # Normal Map
        row_n = col_maps.row(align=True)
        row_n.prop(props, "generate_normal")
        if props.generate_normal:
            icon_n = 'TRIA_DOWN' if props.show_normal_tuning else 'TRIA_RIGHT'
            row_n.prop(props, "show_normal_tuning", text="", icon=icon_n, icon_only=True, emboss=False)
        if props.generate_normal and props.show_normal_tuning:
            sub_n = col_maps.column(align=True)
            sub_n.prop(props, "depth_strength")
            sub_n.prop(props, "color_detail_strength")

        # Roughness Map
        row_r = col_maps.row(align=True)
        row_r.prop(props, "generate_roughness")
        if props.generate_roughness:
            icon_r = 'TRIA_DOWN' if props.show_roughness_tuning else 'TRIA_RIGHT'
            row_r.prop(props, "show_roughness_tuning", text="", icon=icon_r, icon_only=True, emboss=False)
        if props.generate_roughness and props.show_roughness_tuning:
            sub_r = col_maps.column(align=True)
            sub_r.prop(props, "roughness_offset")
            sub_r.prop(props, "roughness_cavity")
            sub_r.prop(props, "roughness_texture")

        # Metallic Map
        row_m = col_maps.row(align=True)
        row_m.prop(props, "generate_metallic")
        if props.generate_metallic:
            icon_m = 'TRIA_DOWN' if props.show_metallic_tuning else 'TRIA_RIGHT'
            row_m.prop(props, "show_metallic_tuning", text="", icon=icon_m, icon_only=True, emboss=False)
        if props.generate_metallic and props.show_metallic_tuning:
            sub_m = col_maps.column(align=True)
            sub_m.prop(props, "metallic_sensitivity")
            sub_m.prop(props, "metallic_shadow_fill")
            sub_m.prop(props, "metallic_pattern_boost")

        layout.separator()

        # =========================================================================
        # STEP 2: Quality & Output Settings
        # =========================================================================
        box_step2 = layout.box()
        box_step2.label(text="Step 2: Quality & Output Settings", icon='MODIFIER')

        box_step2.prop(props, "enhance_mode")
        if props.enhance_mode == 'GUIDED':
            sub_q = box_step2.column(align=True)
            sub_q.prop(props, "guided_radius")
            sub_q.prop(props, "guided_epsilon")

        box_step2.prop(props, "sharpen_depth")
        if props.sharpen_depth:
            box_step2.prop(props, "sharpen_strength")

        box_step2.prop(props, "smooth_depth")
        if props.smooth_depth:
            box_step2.prop(props, "smooth_radius")

        box_step2.separator()
        box_step2.prop(props, "save_to_disk")
        if props.save_to_disk:
            if not props.output_directory or not props.output_directory.strip():
                if props.input_mode in ('SINGLE', 'VIDEO') and props.input_filepath:
                    props.output_directory = props.input_filepath
                elif props.input_mode in ('BATCH', 'SEQUENCE') and props.input_directory:
                    props.output_directory = props.input_directory

            box_step2.prop(props, "output_format")
            box_step2.prop(props, "output_directory", text="Folder")

        layout.separator()

        # =========================================================================
        # MAIN GENERATE BUTTON
        # =========================================================================
        col_btn = layout.column()
        col_btn.enabled = is_model_ready and not is_processing
        col_btn.scale_y = 1.6

        if props.input_mode == 'SINGLE':
            col_btn.operator("da2.generate_depth", text="Generate PBR Maps Now", icon='RENDER_STILL')
        else:
            col_btn.operator("da2.batch_process", text=f"Start Batch ({props.input_mode})", icon='RENDER_ANIMATION')

        # Live Progress Display
        if is_processing or wm.da2_status:
            layout.separator()
            progress_box = layout.box()

            if is_processing:
                progress_box.prop(wm, "da2_progress", text="Progress", slider=True)
                if wm.da2_status:
                    progress_box.label(text=wm.da2_status, icon='TIME')
                progress_box.label(text="Press ESC in viewport to cancel.", icon='CANCEL')

            elif wm.da2_status:
                if wm.da2_status.startswith("Done"):
                    progress_box.label(text=wm.da2_status, icon='CHECKMARK')
                elif wm.da2_status.startswith("Error"):
                    progress_box.label(text=wm.da2_status, icon='ERROR')
                elif wm.da2_status == "Cancelled.":
                    progress_box.label(text=wm.da2_status, icon='CANCEL')
                else:
                    progress_box.label(text=wm.da2_status, icon='INFO')


classes = (
    DA2_PT_main_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
