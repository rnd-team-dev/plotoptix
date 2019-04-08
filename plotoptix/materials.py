"""
PlotOptiX predefined materials.

Copyright (C) 2019 R&D Team. All Rights Reserved.

Have a look at examples on GitHub: https://github.com/rnd-team-dev/plotoptix.
"""

import os

from plotoptix.enums import RtFormat

__pkg_dir__ = os.path.dirname(__file__)

m_flat = {
      "ClosestHitPrograms": [ "0::path_tracing_materials.cu::flat_closest_hit" ],
      "AnyHitPrograms": [ "1::path_tracing_materials.cu::any_hit" ]
    }
"""
Super-fast material, color is not shaded anyhow. Use color components range <0; 1>.
"""

m_eye_normal_cos = {
      "ClosestHitPrograms": [ "0::path_tracing_materials.cu::cos_closest_hit" ],
      "AnyHitPrograms": [ "1::path_tracing_materials.cu::any_hit" ]
    }
"""
Fast material, color is shaded by the cos(eye-hit-normal). Use color components range
<0; 1>.
"""

m_diffuse = {
      "ClosestHitPrograms": [ "0::path_tracing_materials.cu::diffuse_closest_hit" ],
      "AnyHitPrograms": [ "1::path_tracing_materials.cu::any_hit" ],
      "VarInt": { "material_flags": 2 }
    }
"""
Standard diffuse material. Note it is available by default under the name "diffuse".
Use color components range <0; 1>.
"""

m_clear_glass = {
      "ClosestHitPrograms": [ "0::path_tracing_materials.cu::glass_closest_hit" ],
      "AnyHitPrograms": [ "1::path_tracing_materials.cu::any_hit" ],
      "VarInt": { "material_flags": 12 },
      "VarFloat": {
        "refraction_index": 1.4,
        "radiation_length": 0.0,
        "vol_scattering": 1.0,
        "light_emission": 0.0
      },
      "VarFloat3": {
        "surface_albedo": {
          "X": 1.0,
          "Y": 1.0,
          "Z": 1.0
        }
      }
    }
"""
Glass, with reflection and refraction. Color components meaning is "attenuation length"
and the range is <0; inf>.
"""
