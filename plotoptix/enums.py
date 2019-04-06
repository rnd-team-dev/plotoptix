"""
Enums for PlotOptiX raytracer.

Copyright (C) 2019 R&D Team. All Rights Reserved.

Have a look at examples on GitHub: https://github.com/rnd-team-dev/plotoptix.
"""

from enum import Enum

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

class Camera(Enum):
    """
    Cameras (ray generation programs).
    """
    Pinhole = 0
    DoF = 1
    #Ortho = 2

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

