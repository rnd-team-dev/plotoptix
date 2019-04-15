"""
Enums for PlotOptiX raytracer.

Copyright (C) 2019 R&D Team. All Rights Reserved.

Have a look at examples on GitHub: https://github.com/rnd-team-dev/plotoptix.
"""

from enum import Enum, IntFlag

class Coordinates(Enum):
    """
    Coordinate system styles.
    """
    Hidden = 0
    Box = 1
    #Axes = 2

class Geometry(Enum):
    """
    Geometry shapes.
    """
    Unknown = 0
    ParticleSet = 1
    #ParticleNetConstL = 2
    #ParticleSetVarL = 3
    #ParticleSetTextured = 4
    #BezierCurves = 5
    BezierChain = 6
    Parallelograms = 7
    Parallelepipeds = 8

class GeomBuffer(IntFlag):
    """
    Geometry buffer flags. Used for selection of buffers which should
    be updated in GPU (update geometry() method) after geometry modifications
    made with move_geometry()/move_primitive() and similar functions.
    """
    Positions = 1
    Colors0 = 4
    Colors1 = 8
    Colors = Colors0 | Colors1
    Radii = 16
    U = 32
    V = 64
    W = 128
    Vectors = U | V | W
    V0 = 256
    V1 = 512
    V2 = 1024
    V3 = 2048
    VNodes = V0 | V1 | V2 | V3
    All = 0xFFFFFFFF

class Camera(Enum):
    """
    Cameras (ray generation programs).
    """
    Pinhole = 0
    DoF = 1
    #Ortho = 2

class LightShading(Enum):
    """
    Light shading program. Soft converges quickly and
    is well balanced for most scenes and light sources.
    Hard is better suited for scenes with caustics.
    """
    Soft = 0
    Hard = 1

class Light(Enum):
    """
    Light sources.
    """
    Parallelogram = 0
    Spherical = 1

class RtResult(Enum):
    """
    Raytracing result codes.
    """
    Success = 0
    AccumDone = 1
    NoUpdates = 2

class RtFormat(Enum):
    """
    OptiX buffer formats.
    """
    Unknown = 0x100
    Float = 0x101
    Float2 = 0x102
    Float3 = 0x103
    Float4 = 0x104
