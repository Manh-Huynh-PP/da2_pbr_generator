# DA2 + PBR texture Generator from Image

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Blender](https://img.shields.io/badge/Blender-4.2%20%7C%204.3%20%7C%204.4%20%7C%204.5%20%7C%205.0%2B-orange.svg)](https://www.blender.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/manhhuynhsd/Minc-Materials-23-ONNX)

An advanced open-source **Blender Extension / Add-on** that generates high-precision **Depth Maps** and complete **PBR Material Map Suites** (`Albedo`, `Normal`, `Roughness`, `Metallic`) from a single 2D RGB image or video sequence using **Depth Anything V2** AI, **MINC-23 AI Material Recognition**, and physics-based texture analysis.

---

## 🌟 Key Features

- 🧠 **AI Depth Estimation (Depth Anything V2)**: Powered by ONNX Runtime with support for `Small` (~98MB), `Base` (~392MB), and `Large` (~1.3GB) models. Supports CUDA GPU acceleration and CPU fallback.
- 🏷️ **AI Material Recognition (MINC-23 ONNX)**: Automatically classifies 23 visual material categories (Gold/Metal, Wood, Ceramic, Paper, Plastic, Stone, Tile, etc.) using [manhhuynhsd/Minc-Materials-23-ONNX](https://huggingface.co/manhhuynhsd/Minc-Materials-23-ONNX) to drive physically-accurate PBR parameters.
- 🪙 **Gold, Copper & Colored Metal Detection**: Advanced spectral color heuristics (`R > G > B`, saturation & hue gating) to accurately capture Gold, Brass, Copper, and Bronze at full `Metallic = 1.0`.
- 🏛️ **White Marble & Diffuse Non-Metal Suppression**: Eliminates false-positive metallic reflections on diffuse white non-metals (carved marble, plaster, white ceramic, paper).
- 🌊 **Physically-Coupled Roughness Engine**: Dynamically couples Metallic confidence and specular reflection peaks directly into Roughness maps for smooth, mirror-like metallic & polished reflections.
- 🎨 **Albedo (Base Color) Delighting**: Detects ambient occlusion and baked shadows from input images to extract clean, unlit Albedo maps.
- 📐 **Hybrid Tangent Normal Map**: Combines macro depth gradients with micro-frequency luminance details for crisp surface normals.
- 🖼️ **One-Click 3D Textured Plane**: Automatically creates a 3D plane sized to the image's aspect ratio with a pre-wired `Principled BSDF` material and Cycles/Eevee Displacement setup.
- 🗂️ **Batch & Video Frame Processing**: Process folders of images or video files asynchronously without freezing Blender's UI.
- 🎛️ **Collapsible Fine-Tuning UI**: Interactive, foldable sliders under each map for manual adjustment of delighting, roughness cavity, and metallic sensitivity.

---

## 📦 Installation

1. Download the latest `da2_pbr_generator_v1.0.1.zip` release.
2. Open Blender, navigate to **Edit > Preferences > Get Extensions** (or **Add-ons**).
3. Click the menu dropdown at top-right, select **Install from Disk...** and choose `da2_pbr_generator_v1.0.1.zip`.
4. Enable **DA2 + PBR texture Generator from Image**.
5. Go to Add-on Preferences and click **Download Model Now** to fetch the ONNX model weights.

---

## 🚀 Usage Guide

1. Open the 3D Viewport Sidebar (`N` panel) and click on the **DA2 Depth** tab.
2. Select your **Input Image** (or Folder / Video File).
3. Check **AI Material Recognition** to enable automatic MINC-23 material classification.
4. Choose which PBR Maps to generate (`Albedo`, `Normal`, `Roughness`, `Metallic`).
5. Keep **Create Textured Plane** checked to automatically generate a 3D textured mesh in your scene.
6. Click **Generate PBR Maps Now**!

---

## 🛠️ System Requirements

| Model Variant | File Size | Recommended System RAM | VRAM (GPU) |
|---|---|---|---|
| **DA2 Small** | ~98 MB | 4 GB+ | Integrated / Any GPU |
| **DA2 Base** | ~392 MB | 8 GB+ | 4 GB+ VRAM |
| **DA2 Large** | ~1.3 GB | 16 GB+ | 6 GB+ VRAM |
| **MINC-23 Material Model** | ~84.7 MB (INT8 ONNX) | 1 GB+ | Any / CPU Fast |

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Credits & Acknowledgments

- **Depth Estimation AI**: [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (HKU / ByteDance).
- **Material Classifier Model**: [Minc-Materials-23](https://huggingface.co/prithivMLmods/Minc-Materials-23) based on [google/siglip2](https://huggingface.co/google/siglip2-base-patch16-224) (Apache-2.0 License).
- **ONNX Model Repository**: [manhhuynhsd/Minc-Materials-23-ONNX](https://huggingface.co/manhhuynhsd/Minc-Materials-23-ONNX).
- Built for the Blender Open Source Community by **Manh Huynh**.
