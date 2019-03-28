"""
Enums for PlotOptiX raytracer.

Copyright (C) 2019 R&D Team. All Rights Reserved.

Have a look at examples on GitHub: https://github.com/rnd_team/plotoptix.
"""

from enum import Enum

class Coordinates(Enum):
    Hidden = 0
    Box = 1
    #Axes = 2

class Geometry(Enum):
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
    Pinhole = 0
    DoF = 1
    #Ortho = 2

class Light(Enum):
    Parallelogram = 0
    Spherical = 1

class RtResult(Enum):
    Success = 0
    AccumDone = 1
    NoUpdates = 2

