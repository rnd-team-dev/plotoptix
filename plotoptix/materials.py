"""Set of basic configurations and an utility method :meth:`plotoptix.materials.make_material` to create user defined materials.
"""

import os

from typing import Any, Union

from plotoptix.enums import MaterialType

__pkg_dir__ = os.path.dirname(__file__)

def make_material(shader: Union[MaterialType, str],
                  color: Any = [1.0, 1.0, 1.0, 1.0],
                  color_tex: Union[list, str] = [],
                  roughness: float = 0.0,
                  roughness_tex: Union[list, str] = [],
                  metalness: float = 0.0,
                  metalness_tex: Union[list, str] = [],
                  normal_tex: Union[list, str] = [],
                  normaltilt_iar: float = 1.0,
                  specular: float = 0.0,
                  albedo_color: Any = [1.0, 1.0, 1.0],
                  subsurface_color: Any = [1.0, 1.0, 1.0],
                  radiation_length: float = 0.0,
                  light_emission: float = 0.0,
                  refraction_index: Any = [1.41, 1.41, 1.41]) -> dict:
    """Create a dictionary of material parameters.

    Resulting dictionary should be used as an input to :meth:`plotoptix.NpOptiX.setup_material`.

    Parameters
    ----------
    shader : MaterialType or string
        Name of the material type, selects the shader programs and material capabilities, see
        :class:`plotoptix.enums.MaterialType`.

        Dictionary parameter names: ``RadianceProgram`` and ``flags``.
    color : Any, optional
        Base color of the material, modulated by an optional texture and geometry primitive color.
        Single value means a constant gray level. 3(4)-component array means constant RGB(RGBA) color.

        Dictionary parameter name: ``base_color``.
    color_tex: list or string, optional
        List of color texture names or a single texture name. Each texture should be RGBA, and uploaded
        with :meth:`plotoptix.NpOptiX.set_texture_2d` or :meth:`plotoptix.NpOptiX.load_texture` before
        using :meth:`plotoptix.NpOptiX.setup_material`.

        Dictionary parameter name: ``ColorTextures``.
    roughness: float, optional
        Base roughness value, modulated by an optional texture.

        Dictionary parameter name: ``base_roughness``.
    roughness_tex: list or string, optional
        List of roughness texture names or a single texture name. Each texture should be grayscale, and
        uploaded with :meth:`plotoptix.NpOptiX.set_texture_2d` or :meth:`plotoptix.NpOptiX.load_texture`
        before using :meth:`plotoptix.NpOptiX.setup_material`.

        Dictionary parameter name: ``RoughnessTextures``.
    metalness: float, optional
        Base metalness value, modulated by an optional texture.

        Dictionary parameter name: ``reflectivity_index``.
    metalness_tex: list or string, optional
        List of metalness texture names or a single texture name. Each texture should be grayscale, and
        uploaded with :meth:`plotoptix.NpOptiX.set_texture_2d` or :meth:`plotoptix.NpOptiX.load_texture`
        before using :meth:`plotoptix.NpOptiX.setup_material`.

        Dictionary parameter name: ``MetalnessTextures``.
    normal_tex: list or string, optional
        List of normal texture names or a single texture name. Each texture should be :attr:`plotoptix.enums.RtFormat.Float2`,
        encoding UV normal tilt in the tangent space, and uploaded with :meth:`plotoptix.NpOptiX.set_texture_2d`
        before using :meth:`plotoptix.NpOptiX.setup_material`.

        Dictionary parameter name: ``NormalTextures``.
    normaltilt_iar: float, optional
        Inverse aspect ratio of the normal texture, :math:`height / width`.

        Dictionary parameter name: ``normaltilt_iar``.
    specular: float, optional
        Amount of specular reflection.

        Dictionary parameter name: ``reflectivity_range``.
    albedo_color: Any, optional
        Surface albedo color, modulates the reflected rays color. Single value means a gray level.
        3-component array means an RGB color.

        Dictionary parameter name: ``surface_albedo``.
    subsurface_color: Any, optional
        Sub-surface scattering color, modulates color of rays scattered inside the volume. Single value means
        a gray level. 3-component array means an RGB color.

        Dictionary parameter name: ``subsurface_color``.
    radiation_length: float, optional
        Mean free path length for the sub-surface scattering (free path length has an exponential distribution).

        Dictionary parameter name: ``radiation_length``.
    light_emission: float, optional
        Amount of light emission from the diffuse scattering on surfaces or in the sub-surface scattering for
        transmissive materials.

        Dictionary parameter name: ``light_emission``.
    refraction_index: Any, optional
        Refraction index for transmissive materials. Single value means an uniform refraction of all colors.
        3-component array allows for dispersion simulation (individual refraction index values fror each RGB
        component).

        Dictionary parameter name: ``refraction_index``.

    Returns
    -------
    out : Dictionary of parameters, ready to use with :meth:`plotoptix.NpOptiX.setup_material`.
    """
    if isinstance(shader, str): shader = MaterialType[shader]

    if shader == MaterialType.Flat:
        radianceProgram = "materials7.ptx::__closesthit__radiance__flat"
        flags = 0
    elif shader == MaterialType.Cosine:
        radianceProgram = "materials7.ptx::__closesthit__radiance__cos"
        flags = 0
    elif shader == MaterialType.Diffuse:
        radianceProgram = "materials7.ptx::__closesthit__radiance__diffuse"
        flags = 2
    elif shader == MaterialType.TransparentDiffuse:
        radianceProgram = "materials7.ptx::__closesthit__radiance__diffuse_masked"
        flags = 2
    elif shader == MaterialType.Reflective:
        radianceProgram = "materials7.ptx::__closesthit__radiance__reflective"
        flags = 6
    elif shader == MaterialType.TransparentReflective:
        radianceProgram = "materials7.ptx::__closesthit__radiance__reflective_masked"
        flags = 6
    elif shader == MaterialType.Transmissive:
        radianceProgram = "materials7.ptx::__closesthit__radiance__glass"
        flags = 12
    elif shader == MaterialType.ThinWalled:
        radianceProgram = "materials7.ptx::__closesthit__radiance__glass"
        flags = 44
    elif shader == MaterialType.ShadowCatcher:
        radianceProgram = "materials7.ptx::__closesthit__radiance__shadow_catcher"
        flags = 2
    else:
        radianceProgram = "materials7.ptx::__closesthit__radiance__diffuse"
        flags = 2

    occlusionProgram = "materials7.ptx::__closesthit__occlusion"

    c = [0.0, 0.0, 0.0, 1.0]
    if isinstance(color, float) or isinstance(color, int):
        c[0] = c[1] = c[2] = float(color)
    else:
        c[0] = float(color[0])
        c[1] = float(color[1])
        c[2] = float(color[2])
        if len(color) == 4:
            c[3] = float(color[3])

    if isinstance(albedo_color, float) or isinstance(albedo_color, int):
        a = [float(albedo_color), float(albedo_color), float(albedo_color)]
    else:
        a = [float(albedo_color[0]), float(albedo_color[1]), float(albedo_color[2])]

    if isinstance(subsurface_color, float) or isinstance(subsurface_color, int):
        s = [float(subsurface_color), float(subsurface_color), float(subsurface_color)]
    else:
        s = [float(subsurface_color[0]), float(subsurface_color[1]), float(subsurface_color[2])]

    if isinstance(refraction_index, float) or isinstance(refraction_index, int):
        r = [float(refraction_index), float(refraction_index), float(refraction_index)]
    else:
        r = [float(refraction_index[0]), float(refraction_index[1]), float(refraction_index[2])]

    if isinstance(color_tex, str): color_tex = [color_tex,]
    if isinstance(roughness_tex, str): roughness_tex = [roughness_tex,]
    if isinstance(metalness_tex, str): metalness_tex = [metalness_tex,]
    if isinstance(normal_tex, str): normal_tex = [normal_tex,]

    m_params = {
        "RadianceProgram": radianceProgram,
        "OcclusionProgram": "materials7.ptx::__closesthit__occlusion",
        "VarUInt": {
            "flags": flags
            },
        "VarFloat": {
            "base_roughness": float(roughness),
            "reflectivity_index": float(metalness),
            "reflectivity_range": float(specular),
            "radiation_length": float(radiation_length),
            "light_emission": float(light_emission),
            "normaltilt_iar": float(normaltilt_iar)
            },
        "VarFloat3": {
            "surface_albedo": a,
            "subsurface_color": s,
            "refraction_index": r
            },
        "VarFloat4": {
            "base_color": c
            },
        "ColorTextures": color_tex,
        "RoughnessTextures": roughness_tex,
        "MetalnessTextures": metalness_tex,
        "NormalTextures": normal_tex
    }

    return m_params


m_flat = {
      "RadianceProgram": "materials7.ptx::__closesthit__radiance__flat",
      "OcclusionProgram": "materials7.ptx::__closesthit__occlusion",
    }
"""
Super-fast material, color is just flat. Use color components range ``<0; 1>``.
"""

m_eye_normal_cos = {
      "RadianceProgram": "materials7.ptx::__closesthit__radiance__cos",
      "OcclusionProgram": "materials7.ptx::__closesthit__occlusion",
    }
"""
Fast material, color is shaded by the cos(eye-hit-normal). Use color components range
``<0; 1>``.
"""

m_diffuse = {
      "RadianceProgram": "materials7.ptx::__closesthit__radiance__diffuse",
      "OcclusionProgram": "materials7.ptx::__closesthit__occlusion",
      "VarUInt": { "flags": 2 }
    }
"""
Lambertian diffuse material. Note it is available by default under the name "diffuse".
Use color components range ``<0; 1>``.
"""

m_matt_diffuse = {
      "RadianceProgram": "materials7.ptx::__closesthit__radiance__diffuse",
      "OcclusionProgram": "materials7.ptx::__closesthit__occlusion",
      "VarUInt": { "flags": 2 },
      "VarFloat": { "base_roughness": 1 }
    }
"""
Oren-Nayar diffuse material. Surface roughness range is ``<0; inf)``, 0 is equivalent to
the Lambertian "diffuse" material.
"""

m_transparent_diffuse = {
      "RadianceProgram": "materials7.ptx::__closesthit__radiance__diffuse_masked",
      "OcclusionProgram": "materials7.ptx::__closesthit__occlusion_transparency",
      "VarUInt": { "flags": 2 },
      "VarFloat": { "base_roughness": 0 }
    }
"""
Diffuse material with transparency set according to alpha channel of ``ColorTextures``.
Roughness can be set to Lambertian or Oren-Nayar with ``base_roughness`` parameter.
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
range ``<0; 1>``), which results with colorized reflections.
"""

m_metallic = {
      "RadianceProgram": "materials7.ptx::__closesthit__radiance__reflective",
      "OcclusionProgram": "materials7.ptx::__closesthit__occlusion",
      "VarUInt": { "flags": 6 },
      "VarFloat": { "base_roughness": 0.002 },
}
"""
Strongly reflective, metallic material. Note, this material has default values: ``reflectivity_index = 1``
and ``reflectivity_range = 1``. In this configuration the shading algorithm overrides ``surface_albedo``
with the color assigned to each primitive (RGB range <0; 1>), which results with colorized reflections.
Roughness of the surface should be usually small.
"""

m_transparent_metallic = {
      "RadianceProgram": "materials7.ptx::__closesthit__radiance__reflective_masked",
      "OcclusionProgram": "materials7.ptx::__closesthit__occlusion_transparency",
      "VarUInt": { "flags": 6 },
      "VarFloat": { "base_roughness": 0.002 },
}
"""
Strongly reflective, metallic material with transparency set according to alpha channel of
``ColorTextures``. See also :attr:`plotoptix.materials.m_metallic`.
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

m_transparent_plastic = {
      "RadianceProgram": "materials7.ptx::__closesthit__radiance__reflective_masked",
      "OcclusionProgram": "materials7.ptx::__closesthit__occlusion_transparency",
      "VarUInt": { "flags": 6 },
      "VarFloat": {
        "reflectivity_index": 0.0,
        "reflectivity_range": 0.5,
        "base_roughness": 0
      },
      "VarFloat3": {
        "refraction_index": [ 2.0, 2.0, 2.0 ],
        "surface_albedo": [ 1.0, 1.0, 1.0 ]
      }
}
"""
Combined reflective and diffuse surface with transparency set according to alpha channel of
``ColorTextures``. See :attr:`plotoptix.materials.m_plastic` and :attr:`plotoptix.materials.m_matt_plastic`
for details.
"""

m_clear_glass = {
      "RadianceProgram": "materials7.ptx::__closesthit__radiance__glass",
      "OcclusionProgram": "materials7.ptx::__closesthit__occlusion",
      "VarUInt": { "flags": 12 },
      "VarFloat": {
        "radiation_length": 0.0,
        "light_emission": 0.0
      },
      "VarFloat3": {
        "refraction_index": [ 1.4, 1.4, 1.4 ],
        "surface_albedo": [ 1.0, 1.0, 1.0 ],
        "subsurface_color": [ 1.0, 1.0, 1.0 ]
      }
    }
"""
Glass, with reflection and refraction simulated. Color components meaning is "attenuation length"
and the color range is <0; inf>. Set ``radiation_length > 0`` to enable sub-surface scattering. It
is supported in background modes :attr:`plotoptix.enums.MissProgram.AmbientAndVolume`,
:attr:`plotoptix.enums.MissProgram.TextureFixed` and :attr:`plotoptix.enums.MissProgram.TextureEnvironment`,
see also :meth:`plotoptix.NpOptiX.set_background_mode`. Use ``subsurface_color`` to set diffuse color of
scattering (RGB components range should be ``<0; 1>``). Volumes can emit light in  ``subsurface_color``
if ``light_emission > 0``.
"""

m_matt_glass = {
      "RadianceProgram": "materials7.ptx::__closesthit__radiance__glass",
      "OcclusionProgram": "materials7.ptx::__closesthit__occlusion",
      "VarUInt": { "flags": 12 },
      "VarFloat": {
        "radiation_length": 0.0,
        "light_emission": 0.0,
        "base_roughness": 0.2
      },
      "VarFloat3": {
        "refraction_index": [ 1.4, 1.4, 1.4 ],
        "surface_albedo": [ 1.0, 1.0, 1.0 ],
        "subsurface_color": [ 1.0, 1.0, 1.0 ]
      }
    }
"""
Glass with surface roughness configured to obtain matt appearance. Color components meaning is
"attenuation length" and the color range is <0; inf>. Set ``radiation_length > 0`` to enable
sub-surface scattering. It is supported in background modes :attr:`plotoptix.enums.MissProgram.AmbientAndVolume`,
:attr:`plotoptix.enums.MissProgram.TextureFixed` and :attr:`plotoptix.enums.MissProgram.TextureEnvironment`,
see also :meth:`plotoptix.NpOptiX.set_background_mode`. Use ``subsurface_color`` to set diffuse color of
scattering (RGB components range should be ``<0; 1>``). Volumes can emit light in  ``subsurface_color``
if ``light_emission > 0``.
"""

m_dispersive_glass = {
      "RadianceProgram": "materials7.ptx::__closesthit__radiance__glass",
      "OcclusionProgram": "materials7.ptx::__closesthit__occlusion",
      "VarUInt": { "flags": 12 },
      "VarFloat": {
        "radiation_length": 0.0,
        "light_emission": 0.0
      },
      "VarFloat3": {
        "refraction_index": [ 1.4, 1.42, 1.45 ],
        "surface_albedo": [ 1.0, 1.0, 1.0 ],
        "subsurface_color": [ 1.0, 1.0, 1.0 ]
      }
    }
"""
Clear glass, with reflection and refraction simulated. Refraction index is varying with the wavelength,
resulting with the light dispersion. Color components meaning is "attenuation length"
and the range is <0; inf>. Set ``radiation_length > 0`` to enable sub-surface scattering. It
is supported in background modes :attr:`plotoptix.enums.MissProgram.AmbientAndVolume`,
:attr:`plotoptix.enums.MissProgram.TextureFixed` and :attr:`plotoptix.enums.MissProgram.TextureEnvironment`,
see also :meth:`plotoptix.NpOptiX.set_background_mode`. Use ``subsurface_color`` to set diffuse color of
scattering (RGB components range should be ``<0; 1>``). Volumes can emit light in  ``subsurface_color``
if ``light_emission > 0``.
"""

m_thin_walled = {
      "RadianceProgram": "materials7.ptx::__closesthit__radiance__glass",
      "OcclusionProgram": "materials7.ptx::__closesthit__occlusion",
      "VarUInt": { "flags": 44 },
      "VarFloat": {
        "radiation_length": 0.0,
        "light_emission": 0.0
      },
      "VarFloat3": {
        "refraction_index": [ 1.9, 1.9, 1.9 ],
      }
    }
"""
Ideal for the soap-like bubbles. Reflection amount depends on the refraction index, however, there is
no refraction on crossing the surface. Reflections can be textured or colorized with the primitive
colors, and the color values range is ``<0; inf)``.
"""

m_shadow_catcher = {
      "RadianceProgram": "materials7.ptx::__closesthit__radiance__shadow_catcher",
      "OcclusionProgram": "materials7.ptx::__closesthit__occlusion",
      "VarUInt": { "flags": 2 },
      "VarFloat": { "base_roughness": 0 }
    }
"""
Diffuse material, transparent except shadowed regions. Colors, textures, roughness can be set
as for other diffuse materials. Useful for preparation of packshot style images. 
"""
