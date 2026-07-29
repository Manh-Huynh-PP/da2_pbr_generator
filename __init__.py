# MIT License
# Copyright (c) 2026 Manh Huynh
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction beach...

bl_info = {
    "name": "DA2 + PBR texture Generator from Image",
    "author": "Manh Huynh",
    "version": (2, 0, 0),
    "blender": (4, 0, 0),
    "location": "3D View > Sidebar > DA2 Depth Tab",
    "description": "Generates high-precision Depth maps and full PBR material maps (Albedo, Normal, Roughness, Metallic) from a single 2D image using Depth Anything V2 AI and physics-based surface texture analysis.",
    "warning": "",
    "doc_url": "https://github.com/user/da2-pbr-generator",
    "tracker_url": "https://github.com/user/da2-pbr-generator/issues",
    "category": "3D View",
}

from . import preferences
from . import properties
from . import operators
from . import batch_operators
from . import panels


def register():
    preferences.register()
    properties.register()
    operators.register()
    batch_operators.register()
    panels.register()


def unregister():
    panels.unregister()
    batch_operators.unregister()
    operators.unregister()
    properties.unregister()
    preferences.unregister()
