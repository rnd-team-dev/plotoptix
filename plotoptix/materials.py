"""PlotOptiX predefined materials.
"""

import os

from plotoptix.enums import RtFormat

__pkg_dir__ = os.path.dirname(__file__)

m_flat = {
      "RadianceProgram": "materials7.ptx::__closesthit__radiance__flat",
      "OcclusionProgram": "materials7.ptx::__closesthit__occlusion",
    }
"""
Super-fast material, color is not shaded anyhow. Use color components range <0; 1>.
"""

m_eye_normal_cos = {
      "RadianceProgram": "materials7.ptx::__closesthit__radiance__cos",
      "OcclusionProgram": "materials7.ptx::__closesthit__occlusion",
    }
"""
Fast material, color is shaded by the cos(eye-hit-normal). Use color components range
<0; 1>.
"""

m_diffuse = {
      "RadianceProgram": "materials7.ptx::__closesthit__radiance__diffuse",
      "OcclusionProgram": "materials7.ptx::__closesthit__occlusion",
      "VarUInt": { "flags": 2 }
    }
"""
Lambertian diffuse material. Note it is available by default under the name "diffuse".
Use color components range <0; 1>.
"""

m_matt_diffuse = {
      "RadianceProgram": "materials7.ptx::__closesthit__radiance__diffuse",
      "OcclusionProgram": "materials7.ptx::__closesthit__occlusion",
      "VarUInt": { "flags": 2 },
      "VarFloat": { "base_roughness": 1 }
    }
"""
Oren-Nayar diffuse material. Surface roughness range is <0; inf), 0 is equivalent to
the Lambertian "diffuse" material.
"""

m_mirror = {
      "RadianceProgram": "materials7.ptx::__closesthit__radiance__reflective",
      "OcclusionProgram": "materials7.ptx::__closesthit__occlusion",
      "VarUInt": { "flags": 6 },
      "VarFloat3": {
        "surface_albedo": [ 1.0, 1.0, 1.0 ]
      }
}
"""
Reflective mirror, quite simple to calculate and therefore fast material. Note, this material
has default values: ``reflectivity_index = 1`` and ``reflectivity_range = 1``. In this configuration
the shading algorithm overrides ``surface_albedo`` with the color assigned to each primitive (RGB
range <0; 1>), which results with colorized reflections.
"""

m_metalic = {
      "RadianceProgram": "materials7.ptx::__closesthit__radiance__reflective",
      "OcclusionProgram": "materials7.ptx::__closesthit__occlusion",
      "VarUInt": { "flags": 6 },
      "VarFloat": { "base_roughness": 0.002 },
}
"""
Strongly reflective, metalic material. Note, this material has default values: ``reflectivity_index = 1``
and ``reflectivity_range = 1``. In this configuration the shading algorithm overrides ``surface_albedo``
with the color assigned to each primitive (RGB range <0; 1>), which results with colorized reflections.
Roughness of the surface should be usually small.
"""

m_plastic = {
      "RadianceProgram": "materials7.ptx::__closesthit__radiance__reflective",
      "OcclusionProgram": "materials7.ptx::__closesthit__occlusion",
      "VarUInt": { "flags": 6 },
      "VarFloat": {
        "reflectivity_index": 0.0,
        "reflectivity_range": 0.5,
      },
      "VarFloat3": {
        "refraction_index": [ 2.0, 2.0, 2.0 ],
        "surface_albedo": [ 1.0, 1.0, 1.0 ]
      }
}
"""
Combined reflective and diffuse surface. Reflection fraction may be boosted with reflectivity_index
set above 0 (up to 1, resulting with mirror-like appearance) or minimized with a lower than default
reflectivity_range value (down to 0). Higher refraction_index gives a more glossy look.
"""

m_matt_plastic = {
      "RadianceProgram": "materials7.ptx::__closesthit__radiance__reflective",
      "OcclusionProgram": "materials7.ptx::__closesthit__occlusion",
      "VarUInt": { "flags": 6 },
      "VarFloat": {
        "reflectivity_index": 0.0,
        "reflectivity_range": 0.5,
        "base_roughness": 0.001
      },
      "VarFloat3": {
        "refraction_index": [ 2.0, 2.0, 2.0 ],
        "surface_albedo": [ 1.0, 1.0, 1.0 ]
      }
}
"""
Similar to :attr:`plotoptix.materials.m_plastic` but slightly rough surface.
"""

m_clear_glass = {
      "RadianceProgram": "materials7.ptx::__closesthit__radiance__glass",
      "OcclusionProgram": "materials7.ptx::__closesthit__occlusion",
      "VarUInt": { "flags": 12 },
      "VarFloat": {
        "radiation_length": 0.0,
        "vol_scattering": 1.0,
        "light_emission": 0.0
      },
      "VarFloat3": {
        "refraction_index": [ 1.4, 1.4, 1.4 ],
        "surface_albedo": [ 1.0, 1.0, 1.0 ]
      }
    }
"""
Glass, with reflection and refraction simulated. Color components meaning is "attenuation length"
and the color range is <0; inf>.
"""

m_matt_glass = {
      "RadianceProgram": "materials7.ptx::__closesthit__radiance__glass",
      "OcclusionProgram": "materials7.ptx::__closesthit__occlusion",
      "VarUInt": { "flags": 12 },
      "VarFloat": {
        "radiation_length": 0.0,
        "vol_scattering": 1.0,
        "light_emission": 0.0,
        "base_roughness": 0.2
      },
      "VarFloat3": {
        "refraction_index": [ 1.4, 1.4, 1.4 ],
        "surface_albedo": [ 1.0, 1.0, 1.0 ]
      }
    }
"""
Glass with surface roughness configured to obtain matt appearance. Color components meaning is
"attenuation length" and the color range is <0; inf>.
"""

m_dispersive_glass = {
      "RadianceProgram": "materials7.ptx::__closesthit__radiance__glass",
      "OcclusionProgram": "materials7.ptx::__closesthit__occlusion",
      "VarUInt": { "flags": 12 },
      "VarFloat": {
        "radiation_length": 0.0,
        "vol_scattering": 1.0,
        "light_emission": 0.0
      },
      "VarFloat3": {
        "refraction_index": [ 1.4, 1.42, 1.45 ],
        "surface_albedo": [ 1.0, 1.0, 1.0 ]
      }
    }
"""
Clear glass, with reflection and refraction simulated. Refraction index is varying with the wavelength,
resulting with the light dispersion. Color components meaning is "attenuation length"
and the range is <0; inf>.
"""

m_thin_walled = {
      "RadianceProgram": "materials7.ptx::__closesthit__radiance__glass",
      "OcclusionProgram": "materials7.ptx::__closesthit__occlusion",
      "VarUInt": { "flags": 44 },
      "VarFloat": {
        "radiation_length": 0.0,
        "vol_scattering": 1.0,
        "light_emission": 0.0
      },
      "VarFloat3": {
        "refraction_index": [ 1.9, 1.9, 1.9 ],
      }
    }
"""
Ideal for the soap-like bubbles. Reflection amount depends on the refraction index, however, there is
no refraction on crossing the surface. Reflections can be textured or colorized with the primitive
colors, and the color values range is <0; inf>.
"""
