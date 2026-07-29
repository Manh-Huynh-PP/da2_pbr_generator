# MIT License
# Copyright (c) 2026 ANTIGRAVITY AI & Open Source Contributors

import os
import bpy
from bpy.props import (StringProperty, BoolProperty, FloatProperty,
                       EnumProperty, IntProperty, PointerProperty)


def _update_input_filepath(self, context):
    if self.input_filepath:
        self.output_directory = self.input_filepath


def _update_input_directory(self, context):
    if self.input_directory:
        self.output_directory = self.input_directory


class DA2Properties(bpy.types.PropertyGroup):

    input_mode: EnumProperty(
        name="Input Mode",
        items=[
            ('SINGLE', "Single Image", "Process a single input image"),
            ('BATCH', "Batch Folder", "Process all images in a directory"),
            ('SEQUENCE', "Image Sequence", "Process a numbered image sequence"),
            ('VIDEO', "Video File", "Extract frames from a video file and process"),
        ],
        default='SINGLE'
    )

    input_filepath: StringProperty(
        name="Input Image/Video",
        description="Path to input image or video file",
        subtype='FILE_PATH',
        update=_update_input_filepath
    )

    input_directory: StringProperty(
        name="Input Directory",
        description="Path to directory containing input images",
        subtype='DIR_PATH',
        update=_update_input_directory
    )

    apply_plane_fix: BoolProperty(
        name="Plane Fix (Remove Bias)",
        description="Remove depth gradient bias (camera slope bias artifact)",
        default=True
    )

    generate_albedo: BoolProperty(
        name="Albedo (Base Color)",
        description="Generate delighting Albedo map by removing baked shadows and ambient occlusion from input image",
        default=True
    )

    generate_normal: BoolProperty(
        name="Normal Map",
        description="Generate hybrid tangent-space normal map",
        default=True
    )

    generate_roughness: BoolProperty(
        name="Roughness Map",
        description="Generate Roughness Map using Depth Cavity, Gradient, and Surface Texture Analysis",
        default=True
    )

    generate_metallic: BoolProperty(
        name="Metallic Map",
        description="Generate Metallic Map using Surface Normal Pattern Analysis + RGB Color Analysis",
        default=True
    )

    depth_strength: FloatProperty(
        name="Normal Depth Strength",
        description="Depth gradient strength for Normal Map generation",
        default=3.5,
        min=0.1,
        max=10.0
    )

    color_detail_strength: FloatProperty(
        name="Normal Color Detail",
        description="Color luminance detail strength for Normal Map generation",
        default=0.8,
        min=0.0,
        max=5.0
    )

    # --- Fine-Tuning PBR Map Controls ---
    delight_strength: FloatProperty(
        name="Shadow Removal",
        description="Intensity of baked shadow & AO removal for Albedo map",
        default=0.70,
        min=0.0,
        max=1.0,
    )

    delight_saturation: FloatProperty(
        name="Shadow Saturation",
        description="Color saturation recovery in shadow regions",
        default=0.30,
        min=0.0,
        max=1.0,
    )

    roughness_offset: FloatProperty(
        name="Base Roughness",
        description="Base roughness level across the surface",
        default=0.20,
        min=0.0,
        max=1.0,
    )

    roughness_cavity: FloatProperty(
        name="Crevice Roughness",
        description="Roughness intensity in deep crevices and depth hollows",
        default=0.30,
        min=0.0,
        max=1.0,
    )

    roughness_texture: FloatProperty(
        name="Texture Detail Roughness",
        description="Roughness sensitivity to high-frequency color texture detail",
        default=0.25,
        min=0.0,
        max=1.0,
    )

    metallic_sensitivity: FloatProperty(
        name="Metallic Sensitivity",
        description="Sensitivity threshold for detecting metallic reflections",
        default=0.50,
        min=0.0,
        max=1.0,
    )

    metallic_shadow_fill: FloatProperty(
        name="Shadow Hole Fill",
        description="Bridge and fill dark shadow gradients across continuous metal surfaces",
        default=0.85,
        min=0.0,
        max=1.0,
    )

    metallic_pattern_boost: FloatProperty(
        name="Pattern Boost",
        description="Confidence boost for brushed metal streaks & smooth chrome patterns",
        default=0.90,
        min=0.0,
        max=1.0,
    )

    # Collapsible UI fold toggles (default False = collapsed)
    show_albedo_tuning: BoolProperty(name="Fine-Tune Albedo", default=False)
    show_normal_tuning: BoolProperty(name="Fine-Tune Normal", default=False)
    show_roughness_tuning: BoolProperty(name="Fine-Tune Roughness", default=False)
    show_metallic_tuning: BoolProperty(name="Fine-Tune Metallic", default=False)

    # --- Quality Enhancement ---
    enhance_mode: EnumProperty(
        name="Upscale Mode",
        description="Method to upscale depth from 518×518 to original resolution",
        items=[
            ('NONE', "Standard (Bilinear)", "Basic bilinear interpolation"),
            ('GUIDED', "Edge-Guided (Recommended)", "Guided Filter — uses RGB edges for sharp depth boundaries"),
        ],
        default='GUIDED'
    )

    guided_radius: IntProperty(
        name="Filter Radius",
        description="Guided filter window radius. Larger = smoother depth, smaller = more detail",
        default=16,
        min=4,
        max=64
    )

    guided_epsilon: FloatProperty(
        name="Edge Sensitivity",
        description="Lower = sharper edges (may keep noise), Higher = smoother (may blur edges)",
        default=0.02,
        min=0.001,
        max=0.1,
        precision=3
    )

    sharpen_depth: BoolProperty(
        name="Sharpen Depth",
        description="Apply Unsharp Mask to enhance depth edge contrast",
        default=False
    )

    sharpen_strength: FloatProperty(
        name="Sharpen Strength",
        description="Sharpening intensity. Too high may cause halo artifacts",
        default=0.5,
        min=0.0,
        max=2.0
    )

    smooth_depth: BoolProperty(
        name="Smooth Depth Edges",
        description="Smooth extreme depth gradients to reduce black artifacts in displacement. Recommended for displacement rendering",
        default=True
    )

    smooth_radius: IntProperty(
        name="Smooth Radius",
        description="Smoothing kernel size. Larger = smoother but may lose fine detail",
        default=3,
        min=1,
        max=10
    )

    # --- Output ---
    save_to_disk: BoolProperty(
        name="Save to Disk",
        description="Automatically save depth and normal maps to disk",
        default=True
    )

    output_format: EnumProperty(
        name="Output Format",
        description="File format for saved depth and normal maps",
        items=[
            ('PNG', "PNG (16-bit)", "16-bit PNG — compatible, smaller files"),
            ('EXR', "EXR (32-bit float)", "OpenEXR — full precision, no banding, industry standard"),
        ],
        default='EXR'
    )

    output_directory: StringProperty(
        name="Output Path / Prefix",
        description="Output folder path or filename prefix. If empty, saves alongside input image",
        subtype='FILE_PATH'
    )

    auto_create_plane: BoolProperty(
        name="Create Textured Plane",
        description="Auto-create a plane with PBR material (Color + Depth Displacement + Normal Map) sized to match the input image",
        default=True
    )

    displacement_method: EnumProperty(
        name="Displacement Type",
        description="How depth displacement is applied to the plane",
        items=[
            ('MATERIAL', "Material (Cycles)", "Shader-based displacement via Displacement node. Renders in Cycles only, low memory"),
            ('MODIFIER', "Modifier (Viewport)", "Displace modifier — deforms actual geometry. Visible in viewport, works in Eevee, exportable"),
        ],
        default='MODIFIER'
    )


def register():
    bpy.utils.register_class(DA2Properties)
    bpy.types.Scene.da2_props = PointerProperty(type=DA2Properties)
    bpy.types.WindowManager.da2_progress = FloatProperty(
        name="DA2 Progress",
        default=0.0,
        min=0.0,
        max=100.0,
    )
    bpy.types.WindowManager.da2_status = StringProperty(
        name="DA2 Status",
        default="",
    )


def unregister():
    del bpy.types.WindowManager.da2_status
    del bpy.types.WindowManager.da2_progress
    del bpy.types.Scene.da2_props
    bpy.utils.unregister_class(DA2Properties)
