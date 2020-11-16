"""Enumerations and flags used by PlotOptiX raytracer.

https://github.com/rnd-team-dev/plotoptix/blob/master/LICENSE.txt

Have a look at examples on GitHub: https://github.com/rnd-team-dev/plotoptix.
"""

from enum import Enum, IntFlag

class Coordinates(Enum):
    """Coordinate system styles.

    Note, implementation of the coordinate system geometries is ongoing. Now
    only a simple box containing the data points is ready to use.
    """

    Hidden = 0
    """No coordinate system in the image.
    """

    Box = 1
    """Box containing all the data points.
    """

    #Axes = 2

class MissProgram(Enum):
    """Miss program.

    Miss program is executed when ray segment is not intersecting any object
    at a defined maximum distance. Radiance assigned to the ray by the miss
    program appears as a color of the background (i.e. on primary segments)
    or environmental light color (segments scattered off a diffuse surfaces)
    or sub-surface color (segments scattered inside volumes).

    See Also
    --------
    :meth:`plotoptix.NpOptiX.set_background_mode`
    """

    Default = 0
    """Constant background color is used.

    See Also
    --------
    :meth:`plotoptix.NpOptiX.set_background`
    """

    AmbientLight = 1
    """Background color is used if the ray is not scattering of any surface;
    ambient light color is used otherwise (e.g. background can be black while
    the scene is illuminated with any color of light).

    See Also
    --------
    :meth:`plotoptix.NpOptiX.set_ambient`, :meth:`plotoptix.NpOptiX.set_background`
    """

    AmbientAndVolume = 2
    """Same as :attr:`plotoptix.enums.MissProgram.AmbientLight` but supports
    volumetric scattering (just a tiny fraction slower).
    """

    TextureFixed = 3
    """Texture color is used if the ray is not scattering of any surface;
    ambient light color is used otherwise. Texture in the background is not
    reacting to the camera changes and is not affacting the scene illumination.
    This mode supports volumetric scattering.

    See Also
    --------
    :meth:`plotoptix.NpOptiX.set_ambient`, :meth:`plotoptix.NpOptiX.set_background`
    """

    TextureEnvironment = 4
    """Texture color is used for both, the background and the scene illumination.
    Texture pixel is selected by the ray direction, so effectively the texture
    is mapped on the sphere with infinite radius: use 360 deg environment maps.
    This mode supports volumetric scattering.

    See Also
    --------
    :meth:`plotoptix.NpOptiX.set_background`
    """

    #Custom = 5
    #"""Reserved for the future use.
    #"""

class GeomAttributeProgram(Enum):
    """Geometry attributes program.
    
    Executed to find the closest ray-object intersection. Calculates the
    geometry normal and texture coordinates.
    """

    Default = 0
    """
    """

    DisplacedSurface = 2
    """Displaced surface position.
    
    Modifies surface position according to the provided texture. Available with
    textured particles.
    """

    #Custom = 5
    #"""Reserved for the future use.
    #"""

class DisplacementMapping(Enum):
    """Surface displacement mapping mode.
    """

    NormalTilt = 1
    """Only the shading normal is affected.
    """

    DisplacedSurface = 2
    """Surface is actually displaced, shading normal is tilted accordingly.
    """

class TextureMapping(Enum):
    """Texture projection mode.
    """

    Flat = 1
    """Orthogonal projection on a flat surface.
    """

    Spherical = 2
    """Projection on a spherical surface.
    """

class TextureAddressMode(Enum):
    """Texture addressing mode on edge crossing.
    """

    Wrap = 0

    Clamp = 1

    Mirror = 2

    Border = 3

class MaterialType(Enum):
    """Type of the material shader.

    Used to select the shader program when creating
    a dictionary for the new material parameters.
    """

    Flat = 0
    """Simple and fast flat color shading.
    """

    Cosine = 1
    """Shaded transparency with cos(eye-hit-normal), fast.
    """

    Diffuse = 2
    """Lambertian and Oren-Nayar diffuse materials, no specular reflections, no
    transmission, no transparency, therefore a bit faster than other, more
    complex materials.
    """

    TransparentDiffuse = 3
    """Diffuse material, similar to :attr:`plotoptix.enums.MaterialType.Diffuse`,
    but with the transparency support.
    """

    Reflective = 4
    """Supports all surfaces with specular reflections (also mixed with diffuse
    behavior), including metallic surface, with no transmission and no transparency support.
    """

    TransparentReflective = 5
    """Reflective material, similar to :attr:`plotoptix.enums.MaterialType.Reflective`,
    but with the transparency support.
    """

    Transmissive = 6
    """Glass-like, transmissive material with the light refraction and
    dispersion support; includes also volumetric scattering and emission.
    """

    ThinWalled = 7
    """Bubble-like, transmissive material with no refraction on the walls;
    includes volumetric scattering and emission.
    """

    ShadowCatcher = 8
    """Shadow catcher, a diffuse material, transparent except shadowed regions.
    Useful for preparation of packshot style images. 
    """

class Geometry(Enum):
    """Geometry shapes.
    """
    Unknown = 0
    """Value reserved for errors.
    """

    ParticleSet = 1
    """Spherical particle for each data point.

    Each point can have individual radius and color.
    """

    ParticleSetTextured = 4
    """Spherical particle, 3D oriented, for each data point.

    Each point can have individual radius and color. U and V vectors
    allow for 3D orientation of each point so it can works best with
    textures. Texture overrides individual flat colors of particles.

    U vector points to the **north** of the particle, V vector sets
    the zero **longitude** direction. V vector is orthogonalized to U.
    """

    #ParticleNetConstL = 2
    #ParticleSetVarL = 3
    #BezierCurves = 5

    BezierChain = 6
    """Bezier line interpolating data points.

    Line thickness and color can be provided for each data point (curve node).

    Curve is smoothed, use :attr:`plotoptix.enums.Geometry.SegmentChain`
    for a piecewise linear plot.
    """

    SegmentChain = 11
    """Linear segments connecting data points.

    Line thickness and color can be provided for each data point (node).
    """

    BSplineQuad = 12
    """Quadratic b-spline with nodes at data points.

    Line thickness and color can be provided for each data point (curve node).
    
    Note: b-spline is not interpolating data points; see examples
    how to pin start/end to a fixed position. Use
    :attr:`plotoptix.enums.Geometry.BezierChain` for a smooth curve
    interpolating all data points.
    """

    BSplineCubic = 13
    """Cubic b-spline with nodes at data points.

    Line thickness and color can be provided for each data point (curve node).
    
    Note: b-spline is not interpolating data points; see examples
    how to pin start/end to a fixed position. Use
    :attr:`plotoptix.enums.Geometry.BezierChain` for a smooth curve
    interpolating all data points.
    """

    Parallelograms = 7
    """Flat parallelograms.

    Color and parallelogram U / V vectors can be provided for each
    data point. 
    """

    Parallelepipeds = 8
    """Parallelepipeds.

    Color and U / V / W vectors can be provided for each data point.
    """

    Tetrahedrons = 9
    """Tetrahedrons.

    Color and U / V / W vectors can be provided for each data point.
    """

    Mesh = 10
    """Mesh.

    Color and normal vectors can be provided for each data point (mesh vertex).
    """

    Graph = 14
    """Graph or a wireframed mesh.

    Color and radius can be provided for each data point (graph vertex).
    """

class GeomBuffer(IntFlag):
    """Geometry buffer flags.
    
    Flags are used for:

    - selection of buffers which should be updated on GPU side with
      :meth:`plotoptix.NpOptiX.update_geom_buffers` method after geometry
      modifications made with move_geometry()/move_primitive() and similar
      functions;
    - selection of internal memory buffer for direct data access with
      :class:`plotoptix.geometry.PinnedBuffer`.
    """

    Positions = 1
    """Data points positions.
    """
    Velocities = 2
    """Velocities of data points / curve nodes / mesh vertices.

    Used only on the host side, for the simulation support. Allocated on
    the first access.
    """

    Colors0 = 4
    """Bezier segments starting color.
    """

    Colors1 = 8
    """Bezier segments end color.
    """

    Colors = Colors0 | Colors1
    """Any geometry color, including start/end of bezier and line
    segments.

    Allocated on the first access (otherwise constant color is used).
    """

    Radii = 16
    """Particles radii, bezier and line segment thickness.
    """

    U = 32
    """U vector of parallelograms, parallelepipeds and tetrahedrons.
    """

    V = 64
    """V vector of parallelograms, parallelepipeds and tetrahedrons.
    """

    W = 128
    """W vector of parallelograms, parallelepipeds and tetrahedrons.
    """

    Vectors = U | V | W
    """All vectors of parallelograms, parallelepipeds and tetrahedrons; normal vectors of meshes.
    """

    V0 = 256
    """Start node of bezier segments.
    """

    V1 = 512
    """1st mid node of bezier segments.
    """

    V2 = 1024
    """2nd mid node of bezier segments.
    """

    V3 = 2048
    """End node of bezier segments.
    """

    VNodes = V0 | V1 | V2 | V3
    """All nodes of bezier and line segments.
    """

    VertexIdx = 4096
    """Triplets of mesh face indexes (to be replaced with the ``FaceIdx`` name).
    """
    FaceIdx = 4096
    """Triplets of mesh face indexes.
    """

    NormalIdx = 8192
    """Mesh normal indexes.
    """

    TextureMap = 16384
    """Tecture UV coordinates.
    """

    EdgeIdx = 32768
    """Doublets of mesh or graph edge indexes.

    Allocated on the first access in meshes.
    """

    All = 0xFFFFFFFF
    """All buffers.
    """

class Camera(Enum):
    """Cameras (ray generation programs).
    """

    Pinhole = 0
    """Perspective camera.
    """

    DoF = 1
    """Thin lens perspective camera with depth of field simulation.

    This camera produces ideal, straight perspective lines.
    """

    ThinLens = 1
    """Alias for DoF.
    
    Actually a better name that should replace DoF at some point.
    """

    Panoramic = 2
    """360 deg panoramic (equirectangular) camera.
    """

    Ortho = 3
    """Orthogonal projection camera.
    """

    Fisheye = 4
    """Fisheye (equisolid) camera with depth of field simulation.

    This camera renders a fisheye lens distortion of perspective lines. It
    also focuses on a sphere rather than on a plane like thin lens camera.
    """

    ThinLensChroma = 5
    """Thin lens perspective camera with depth of field and chromatic abberation simulation.

    This camera produces ideal, straight perspective lines.
    """

    FisheyeChroma = 6
    """Fisheye (equisolid) camera with depth of field and chromatic abberation simulation.

    This camera renders a fisheye lens distortion of perspective lines. It
    also focuses on a sphere rather than on a plane like thin lens camera.
    """

    CustomProj = 98
    """Custom projection camera.

    Ray angles are defined with a 2D texture ``[height, width, 2]`` composed of angles
    ``[horizontal, vertical]`` w.r.t. the camera axis. Angles should be provided in radians,
    and normalized so the value ``1.0`` is corresponding to ``pi``. Negative angles are to
    the left (horizontal) and down (vertical) w.r.t. the camera axis.

    Exact values from the texture are used if the render size and texture size are the
    same. Otherwise ray angles are interpolated.
    """

    TexTest = 99
    """Test texture mapping in the camera shader.
    """

class LightShading(Enum):
    """Light shading program.
    """
    Soft = 0
    """Soft light shading.

    Raytracing in this mode converges quickly and lighting is well
    balanced for most scenes and light sources.
    """

    Hard = 1
    """Hard light shading.

    Slower convergence (more frames required to eliminate noise),
    but much better suited for scenes with caustics.
    """

class Light(Enum):
    """Light sources.
    """

    Parallelogram = 0
    """Flat parallelogram, light on front face only.
    """

    Spherical = 1
    """Spherical light, shining in all directions.
    """

class Channel(Enum):
    """Image color channel.
    """

    Gray = 0
    """Brightness.
    """

    R = 1
    """Red channel.
    """

    G = 2
    """Green channel.
    """

    B = 3
    """Blue channel.
    """

    A = 4
    """Alpha channel.
    """

class ChannelOrder(Enum):
    """Color channel ordering.
    """

    RGB = 1
    """
    """

    BGR = 2
    """
    """

    RGBA = 3
    """
    """

    BGRA = 4
    """
    """

class ChannelDepth(Enum):
    """Color channel depth: 8, 16 or 32 bits per sample.
    """

    Bps8 = 1
    """
    """

    Bps16 = 2
    """
    """

    Bps32 = 3
    """
    """

class Postprocessing(Enum):
    """2D postprocessing stages.

    Postprocessing stages can be added with :meth:`plotoptix.NpOptiX.add_postproc`
    to correct ray traced 2D image. Each algorithm has its own variables that should
    be configured before adding the postprocessing stage.

    See Also
    --------
    :meth:`plotoptix.NpOptiX.add_postproc`
    """

    Levels = 1
    """Image levels correction.

    Variables to configure:
    - levels_low_range, float3, RGB values
    - levels_high_range, float3, RGB values.

    Examples
    --------
    >>> optix = TkOptiX()
    >>> optix.set_float("levels_low_range", 0.1, 0.15, 0.2)
    >>> optix.set_float("levels_high_range", 0.9, 0.8, 0.7)
    >>> optix.add_postproc("Levels")
    """

    Gamma = 2
    """Image gamma correction.

    Variables to configure:
    - tonemap_exposure, float, exposure value
    - tonemap_gamma, float, gamma value.

    Examples
    --------
    >>> optix = TkOptiX()
    >>> optix.set_float("tonemap_exposure", 0.6)
    >>> optix.set_float("tonemap_gamma", 1/2.2)
    >>> optix.add_postproc("Gamma")
    """

    GrayCurve = 3
    """Brightness correction curve.

    Variables to configure:
    
    - tonemap_exposure, float, exposure value
    - tone_curve_gray, texture 1D, correction curve; can be configured by
      passing the values directly (:meth:`plotoptix.NpOptiX.set_texture_1d`) or
      with :meth:`plotoptix.NpOptiX.set_correction_curve`

    Examples
    --------
    >>> optix = TkOptiX()
    >>> optix.set_float("tonemap_exposure", 0.6)
    >>> optix.set_texture_1d("tone_curve_gray", [0, 0.33, 0.75, 1])
    >>> optix.add_postproc("GrayCurve")
    """

    RgbCurve = 4
    """RGB correction curve.

    Variables to configure:
    
    - tonemap_exposure, float, exposure value
    - tone_curve_r, texture 1D, red channel correction curve
    - tone_curve_g, texture 1D, green channel correction curve
    - tone_curve_b, texture 1D, blue channel correction curve
    
    Correction curves can be configured by passing the values directly
    (using :meth:`plotoptix.NpOptiX.set_texture_1d`) or with
    :meth:`plotoptix.NpOptiX.set_correction_curve`

    Examples
    --------
    >>> optix = TkOptiX()
    >>> optix.set_float("tonemap_exposure", 0.6)
    >>> optix.set_texture_1d("tone_curve_r", [0, 0.31, 0.75, 1])
    >>> optix.set_texture_1d("tone_curve_g", [0, 0.33, 0.78, 1])
    >>> optix.set_texture_1d("tone_curve_b", [0, 0.35, 0.81, 1])
    >>> optix.add_postproc("RgbCurve")
    """

    Mask = 5
    """2D mask multiplied by the image.

    Variables to configure:
    
    - frame_mask, grayscale texture 2D, mask to apply

    Examples
    --------
    >>> optix = TkOptiX()
    >>>
    >>> x = np.linspace(-1, 1, 20)
    >>> z = np.linspace(-1, 1, 20)
    >>> Mx, Mz = np.meshgrid(x, z)
    >>> M = np.abs(Mx) ** 3 + np.abs(Mz) ** 3
    >>> M = 1 - (0.6 / np.max(M)) * M
    >>>
    >>> optix.set_texture_2d("frame_mask", M)
    >>> optix.add_postproc("Mask")
    """

    Overlay = 6
    """2D overlay, added to the image according to alpha channel.

    Variables to configure:
    
    - frame_overlay, RGBA texture 2D, an overlay to add

    Examples
    --------
    >>> optix = TkOptiX()
    >>>
    >>> # ...read or create overlay image
    >>>
    >>> optix.set_texture_2d("frame_overlay", M)
    >>> optix.add_postproc("Overlay")
    """

    Denoiser = 7
    """AI denoiser.

    Variables to configure:
    
    - denoiser_blend, float, amount of original image mixed with denoiser output
      range: 0 (only denoiser output) to 1 (only original raytracing output)

    - denoiser_kind, int value of :class:`plotoptix.enums.DenoiserKind`, decides
      which buffers are used as denoiser inputs

    Examples
    --------
    >>> optix = TkOptiX()
    >>>
    >>> optix.set_float("denoiser_blend", 0.5)
    >>> optix.set_int("denoiser_kind", DenoiserKind.Rgb.value)
    >>> optix.add_postproc("Denoiser")
    """

class DenoiserKind(Enum):
    """Inputs provided to the denoiser.
    """

    Rgb = 0x2301
    """Only raw RGB values are used. Use this mode to save memory or
    if denoising in other modes is not satisfactory.
    """

    RgbAlbedo = 0x2302
    """Default mode. Raw RGB values and surface albedo are used.
    """

    RgbAlbedoNormal = 0x2303
    """Raw RGB values, surface albedo and normals are used. Use this
    mode for scenes with fine details of surface structures.
    """

class RtResult(Enum):
    """Raytracing result codes.
    """

    Success = 0
    """Frame raytracing completed with no errors.
    """

    AccumDone = 1
    """Last accumulation frame completed.

    Image is ready, raytracng stops until changes in the scene
    are made.
    """

    NoUpdates = 2
    """There is no change in the output image.
    """

    ComputeTimeout = 0x100
    """Compute/upload task timed out.
    """

    LaunchTimeout = 0x101
    """Ray-tracing task timed out.
    """

class RtFormat(Enum):
    """OptiX buffer formats.
    """

    Unknown = 0x100
    """Reserved.
    """

    Float = 0x101
    """32 bit single precision scalars.
    """

    Float2 = 0x102
    """32 bit single precision 2D vectors.
    """

    Float3 = 0x103
    """32 bit single precision 3D vectors.
    """

    Float4 = 0x104
    """32 bit single precision (x, y, z, w) vectors.
    """

class NvEncProfile(Enum):
    """H.264 encoding profile.

    Beware that some combinations are not supported by all players
    (e.g. lossless encoding is not playable in Windows Media Player).

    See Also
    --------
    :meth:`plotoptix.NpOptiX.encoder_create`
    """

    Default = 0
    """
    """

    Baseline = 1
    """
    """

    Main = 2
    """
    """

    High = 3
    """
    """

    High444 = 4
    """
    """

class NvEncPreset(Enum):
    """H.264 encoding preset.

    Beware that some combinations may not be supported by all players
    (e.g. lossless encoding is not playable in Windows Media Player).

    See Also
    --------
    :meth:`plotoptix.NpOptiX.encoder_create`
    """

    Default = 0
    """
    """

    HP = 1
    """
    """

    HQ = 2
    """
    """

    BD = 3
    """
    """

    LL = 4
    """
    """

    LL_HP = 5
    """
    """

    LL_HQ = 6
    """
    """

    Lossless = 7
    """
    """

    Lossless_HP = 8
    """
    """

class GpuArchitecture(Enum):
    """SM architecture.

    See Also
    --------
    :meth:`plotoptix.utils.get_gpu_architecture`, :meth:`plotoptix.utils.set_gpu_architecture`,
    :meth:`plotoptix.NpOptiX.get_gpu_architecture`
    """

    Auto = 0
    """Select highest SM architecture matching available GPU's.
    """

    Compute_50 = 500
    """Maxwell.
    """

    Compute_52 = 520
    """Maxwell.
    """

    #Compute_53 = 530
    #"""Maxwell.
    #"""

    Compute_60 = 600
    """Pascal.
    """

    Compute_61 = 610
    """Pascal.
    """

    #Compute_62 = 620
    #"""Pascal.
    #"""

    Compute_70 = 700
    """Volta.
    """

    #Compute_72 = 720
    #"""Volta.
    #"""

    Compute_75 = 750
    """Turing.
    """

    Compute_86 = 860
    """Ampere.
    """