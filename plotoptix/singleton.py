"""
Singleton metaclass for PlotOptiX raytracer.

Copyright (C) 2019 R&D Team. All Rights Reserved.

Have a look at examples on GitHub: https://github.com/rnd-team-dev/plotoptix.
"""

class Singleton(type):
    """
    Singleton metaclass for NpOptiX and derived UI's.

    OptiX does not support multiple coxtexts. It works other way around: leverages multiple GPU
    in a single context. => Only single raytracing UI window or headless raytracer per process
    is possible with PlotOptiX.
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
