"""Enumerations and flags used by PlotOptiX raytracer.
"""

from enum import Enum, IntFlag

class Coordinates(Enum):
    """Coordinate system styles.

    Note, implementation of the coordinate system geometries is ongoing. Now
    only a simple box containing the data points is visible in plots.
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
    program appears as a color of the background (e.g. on primary segments)
    or environmental light color (segments scattered off a diffuse surfaces).

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

    #AmbientAndVolume = 2
    #"""***Not yet used.*** Same as AmbientLight but supports volumetric
    #scattering.
    #"""

    TextureFixed = 3
    """Texture color is used if the ray is not scattering of any surface;
    ambient light color is used otherwise. Texture in the background is not
    reacting to the camera changes and is not affacting the scene illumination.

    See Also
    --------
    :meth:`plotoptix.NpOptiX.set_ambient`, :meth:`plotoptix.NpOptiX.set_background`
    """

    TextureEnvironment = 4
    """Texture color is used for both, the background and the scene illumination.
    Texture pixel is selected by the ray direction, so effectively the texture
    is mapped on the sphere with infinite radius: use 360 deg environment maps.

    See Also
    --------
    :meth:`plotoptix.NpOptiX.set_background`
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

    #ParticleNetConstL = 2
    #ParticleSetVarL = 3
    #ParticleSetTextured = 4
    #BezierCurves = 5
    
    BezierChain = 6
    """Bezier line interpolating data points.

    Line thickness and color can be provided for each data point.
    Line is smoothed, use SegmentChain for a piecewise linear plot.
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

    Color and normal vectors can be provided for each data point.
    """

class GeomBuffer(IntFlag):
    """Geometry buffer flags.
    
    Flags are used for selection of buffers which should be updated
    in GPU (update geometry() method) after geometry modifications
    made with move_geometry()/move_primitive() and similar functions.
    """

    Positions = 1
    """Data point positions.
    """

    Colors0 = 4
    """Bezier and line segments starting color.
    """

    Colors1 = 8
    """Bezier and line segments end color.
    """

    Colors = Colors0 | Colors1
    """Any geometry color, including start/end of bezier and line
    segments.
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
    """All vectors of parallelograms, parallelepipeds and tetrahedrons.
    """

    V0 = 256
    """Start node of bezier and line segments.
    """

    V1 = 512
    """1st mid node of bezier segments.
    """

    V2 = 1024
    """2nd mid node of bezier segments.
    """

    V3 = 2048
    """End node of bezier and line segments.
    """

    VNodes = V0 | V1 | V2 | V3
    """All nodes of bezier and line segments.
    """

    VertexIdx = 4096
    """Mesh vertex indexes.
    """

    NormalIdx = 8192
    """Mesh normal indexes.
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
    """Perspective camera with depth of field simulation.
    """

    #Ortho = 2

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
    - tonemap_igamma, float, 1/gamma value.

    Examples
    --------
    >>> optix = TkOptiX()
    >>> optix.set_float("tonemap_exposure", 0.6)
    >>> optix.set_float("tonemap_igamma", 1/2.2)
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
    
    - frame_mask, texture 2D, mask to apply

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
