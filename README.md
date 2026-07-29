# DA2 + PBR texture Generator from Image

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Blender](https://img.shields.io/badge/Blender-4.0%20%7C%204.1%20%7C%204.2%20%7C%205.0%2B-orange.svg)](https://www.blender.org/)

An advanced open-source **Blender Add-on** that generates high-precision **Depth Maps** and complete **PBR Material Map Suites** (`Albedo`, `Normal`, `Roughness`, `Metallic`) from a single 2D RGB image or video sequence using **Depth Anything V2** AI and physics-based texture analysis.

---

## 🌟 Key Features

- 🧠 **AI Depth Estimation (Depth Anything V2)**: Powered by ONNX Runtime with support for `Small` (~98MB), `Base` (~392MB), and `Large` (~1.3GB) models. Supports GPU acceleration (CUDA) and CPU fallback.
- 🎨 **Albedo (Base Color) Delighting**: Automatically detects ambient occlusion and baked shadows from input images to extract clean, unlit Albedo maps.
- 📐 **Hybrid Tangent Normal Map**: Combines macro depth gradients with micro-frequency luminance details for crisp surface normals.
- 🧊 **Physics-Based Roughness Map**: Analyzes screen-space ambient occlusion (crevice roughness), depth curvature, and texture variance.
- 🪙 **Anisotropic Metallic Map with Shadow Fill**: Uses Structure Tensor anisotropy ($\lambda_1, \lambda_2$) for brushed metal streaks & chrome, paired with morphological region closing to fill dark shadow gradient holes.
- 🖼️ **One-Click 3D Textured Plane**: Automatically creates a 3D plane sized to the image's aspect ratio with a pre-wired `Principled BSDF` material and Cycles/Eevee Displacement setup.
- 🗂️ **Batch & Video Frame Processing**: Process folders of images or video files asynchronously without freezing Blender's UI.
- 🎛️ **Collapsible Fine-Tuning UI**: Interactive, foldable sliders under each map for manual adjustment of shadow removal, roughness cavity, and metallic sensitivity.

---

## 📦 Installation

1. Download the latest `.zip` release of the add-on.
2. Open Blender, navigate to **Edit > Preferences > Add-ons**.
3. Click **Install...** (or **Install from Disk**) and select the downloaded zip file.
4. Enable **DA2 + PBR texture Generator from Image**.
5. Go to the Add-on Preferences and click **Download Model Now** to fetch the ONNX model weights from HuggingFace.

---

## 🚀 Usage Guide

1. Open the 3D Viewport Sidebar (`N` panel) and click on the **DA2 Depth** tab.
2. Select your **Input Image** (or Folder / Video File).
3. Choose which PBR Maps to generate (`Albedo`, `Normal`, `Roughness`, `Metallic`). Click `▶` to adjust fine-tuning sliders if needed.
4. Keep **Create Textured Plane** checked to automatically generate a 3D textured mesh in your scene.
5. Click **Generate PBR Maps Now**!

---

## 🛠️ System Requirements

| Model Variant | File Size | Recommended System RAM | VRAM (GPU) |
|---|---|---|---|
| **DA2 Small** | ~98 MB | 4 GB+ | Integrated / Any GPU |
| **DA2 Base** | ~392 MB | 8 GB+ | 4 GB+ VRAM |
| **DA2 Large** | ~1.3 GB | 16 GB+ | 6 GB+ VRAM |

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Credits & Acknowledgments

- Core Depth Estimation AI model: [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (HKU / ByteDance).
- Built for the Blender Open Source Community by **Manh Huynh**.
