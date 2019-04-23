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
    #SegmentChain = 10
    
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

    Color and U/V/W vectors can be provided for each data point.
    """

    Tetrahedrons = 9
    """Tetrahedrons.

    Color and U/V/W vectors can be provided for each data point.
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
