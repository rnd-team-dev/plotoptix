"""
Import RnD.SharpOptiX library.

Copyright (C) 2019 R&D Team. All Rights Reserved.

Have a look at examples on GitHub: https://github.com/rnd-team-dev/plotoptix.
"""

import os, platform, sys

from ctypes import cdll, CFUNCTYPE, POINTER, byref, c_float, c_uint, c_int, c_long, c_bool, c_char_p, c_wchar_p, c_void_p

BIN_PATH = "bin"

PLATFORM = platform.system()
if PLATFORM == "Linux":
    import clr
    from System import IntPtr

PARAM_NONE_CALLBACK = CFUNCTYPE(None)
PARAM_INT_CALLBACK = CFUNCTYPE(None, c_int)

sharp_optix = None


def _load_optix_win():
    """
    Load RnD.SharpOptiX library with ctypes, setup arguments and return types.
    """
    dll_name = os.path.join(os.path.dirname(__file__), BIN_PATH, "RnD.SharpOptiX.dll")

    optix = cdll.LoadLibrary(dll_name)

    optix.create_empty_scene.argtypes = [c_int, c_int, c_void_p, c_int]
    optix.create_empty_scene.restype = c_bool

    optix.create_scene_from_json.argtypes = [c_wchar_p, c_int, c_int, c_void_p, c_int]
    optix.create_scene_from_json.restype = c_bool

    optix.load_scene_from_json.argtypes = [c_wchar_p]
    optix.load_scene_from_json.restype = c_bool

    optix.load_scene_from_file.argtypes = [c_wchar_p]
    optix.load_scene_from_file.restype = c_bool

    optix.save_scene_to_file.argtypes = [c_wchar_p]
    optix.save_scene_to_file.restype = c_bool

    optix.save_image_to_file.argtypes = [c_wchar_p]
    optix.save_image_to_file.restype = c_bool

    optix.start_rt.restype = c_bool
    optix.stop_rt.restype = c_bool

    optix.set_compute_paused.argtypes = [c_bool]
    optix.set_compute_paused.restype = c_bool

    optix.get_int.argtypes = [c_wchar_p, POINTER(c_int)]
    optix.get_int.restype = c_bool

    optix.set_int.argtypes = [c_wchar_p, c_int, c_bool]
    optix.set_int.restype = c_bool

    optix.get_uint.argtypes = [c_wchar_p, POINTER(c_uint)]
    optix.get_uint.restype = c_bool

    optix.set_uint.argtypes = [c_wchar_p, c_uint, c_bool]
    optix.set_uint.restype = c_bool

    optix.get_uint2.argtypes = [c_wchar_p, POINTER(c_uint), POINTER(c_uint)]
    optix.get_uint2.restype = c_bool

    optix.set_uint2.argtypes = [c_wchar_p, c_uint, c_uint, c_bool]
    optix.set_uint2.restype = c_bool

    optix.get_float.argtypes = [c_wchar_p, POINTER(c_float)]
    optix.get_float.restype = c_bool

    optix.set_float.argtypes = [c_wchar_p, c_float, c_bool]
    optix.set_float.restype = c_bool

    optix.get_float2.argtypes = [c_wchar_p, POINTER(c_float), POINTER(c_float)]
    optix.get_float2.restype = c_bool

    optix.set_float2.argtypes = [c_wchar_p, c_float, c_float, c_bool]
    optix.set_float2.restype = c_bool

    optix.get_float3.argtypes = [c_wchar_p, POINTER(c_float), POINTER(c_float), POINTER(c_float)]
    optix.get_float3.restype = c_bool

    optix.set_float3.argtypes = [c_wchar_p, c_float, c_float, c_float, c_bool]
    optix.set_float3.restype = c_bool

    optix.set_texture_1d.argtypes = [c_wchar_p, c_void_p, c_int, c_uint, c_bool]
    optix.set_texture_1d.restype = c_bool

    optix.set_texture_2d.argtypes = [c_wchar_p, c_void_p, c_int, c_int, c_uint, c_bool]
    optix.set_texture_2d.restype = c_bool

    optix.resize_scene.argtypes = [c_int, c_int, c_void_p, c_int]
    optix.resize_scene.restype = c_bool

    optix.get_material.argtypes = [c_wchar_p]
    optix.get_material.restype = c_wchar_p

    optix.setup_material.argtypes = [c_wchar_p, c_wchar_p]
    optix.setup_material.restype = c_bool

    optix.set_correction_curve.argtypes = [c_void_p, c_int, c_int, c_int, c_float, c_bool]
    optix.set_correction_curve.restype = c_bool

    optix.add_postproc.argtypes = [c_int, c_bool]
    optix.add_postproc.restype = c_bool

    optix.setup_geometry.argtypes = [c_int, c_wchar_p, c_wchar_p, c_bool, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    optix.setup_geometry.restype = c_uint

    optix.update_geometry.argtypes = [c_wchar_p, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    optix.update_geometry.restype = c_uint

    optix.get_surface_size.argtypes = [c_wchar_p, POINTER(c_uint), POINTER(c_uint)]
    optix.get_surface_size.restype = c_bool

    optix.update_surface.argtypes = [c_wchar_p, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_float, c_float, c_float, c_float, c_float]
    optix.update_surface.restype = c_uint

    optix.setup_surface.argtypes = [c_wchar_p, c_wchar_p, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_float, c_float, c_float, c_float, c_float, c_bool]
    optix.setup_surface.restype = c_uint

    optix.setup_mesh.argtypes = [c_wchar_p, c_wchar_p, c_int, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    optix.setup_mesh.restype = c_uint

    optix.update_mesh.argtypes = [c_wchar_p, c_int, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    optix.update_mesh.restype = c_uint

    optix.load_mesh_obj.argtypes = [c_wchar_p, c_wchar_p, c_wchar_p, c_void_p, c_bool]
    optix.load_mesh_obj.restype = c_uint

    optix.move_geometry.argtypes = [c_wchar_p, c_float, c_float, c_float, c_bool]
    optix.move_geometry.restype = c_bool

    optix.move_primitive.argtypes = [c_wchar_p, c_long, c_float, c_float, c_float, c_bool]
    optix.move_primitive.restype = c_bool

    optix.rotate_geometry.argtypes = [c_wchar_p, c_float, c_float, c_float, c_bool]
    optix.rotate_geometry.restype = c_bool

    optix.rotate_primitive.argtypes = [c_wchar_p, c_long, c_float, c_float, c_float, c_bool]
    optix.rotate_primitive.restype = c_bool

    optix.rotate_geometry_about.argtypes = [c_wchar_p, c_float, c_float, c_float, c_float, c_float, c_float, c_bool]
    optix.rotate_geometry_about.restype = c_bool

    optix.rotate_primitive_about.argtypes = [c_wchar_p, c_long, c_float, c_float, c_float, c_float, c_float, c_float, c_bool]
    optix.rotate_primitive_about.restype = c_bool

    optix.scale_geometry.argtypes = [c_wchar_p, c_float, c_bool]
    optix.scale_geometry.restype = c_bool

    optix.scale_primitive.argtypes = [c_wchar_p, c_long, c_float, c_bool]
    optix.scale_primitive.restype = c_bool

    optix.update_geom_buffers.argtypes = [c_wchar_p, c_uint]
    optix.update_geom_buffers.restype = c_bool

    optix.set_coordinates_geom.argtypes = [c_int, c_float]
    optix.set_coordinates_geom.restype = c_bool

    optix.setup_camera.argtypes = [c_int, c_void_p, c_void_p, c_void_p, c_float, c_float, c_float, c_float, c_float, c_bool]
    optix.setup_camera.restype = c_int

    optix.update_camera.argtypes = [c_uint, c_void_p, c_void_p, c_void_p, c_float, c_float, c_float]
    optix.update_camera.restype = c_bool

    optix.fit_camera.argtypes = [c_uint, c_wchar_p, c_float]
    optix.fit_camera.restype = c_bool

    optix.get_current_camera.restype = c_uint

    optix.set_current_camera.argtypes = [c_uint]
    optix.set_current_camera.restype = c_bool

    optix.rotate_camera_eye.argtypes = [c_int, c_int, c_int, c_int]
    optix.rotate_camera_eye.restype = c_bool

    optix.rotate_camera_tgt.argtypes = [c_int, c_int, c_int, c_int]
    optix.rotate_camera_tgt.restype = c_bool

    optix.get_camera_focal_scale.argtypes = [c_uint]
    optix.get_camera_focal_scale.restype = c_float

    optix.set_camera_focal_scale.argtypes = [c_float]
    optix.set_camera_focal_scale.restype = c_bool

    optix.set_camera_focal_length.argtypes = [c_float]
    optix.set_camera_focal_length.restype = c_bool

    optix.get_camera_fov.argtypes = [c_uint]
    optix.get_camera_fov.restype = c_float

    optix.set_camera_fov.argtypes = [c_float]
    optix.set_camera_fov.restype = c_bool

    optix.get_camera_aperture.argtypes = [c_uint]
    optix.get_camera_aperture.restype = c_float

    optix.set_camera_aperture.argtypes = [c_float]
    optix.set_camera_aperture.restype = c_bool

    optix.get_camera_eye.argtypes = [c_uint, c_void_p]
    optix.get_camera_eye.restype = c_bool

    optix.set_camera_eye.argtypes = [c_void_p]
    optix.set_camera_eye.restype = c_bool

    optix.get_camera_target.argtypes = [c_uint, c_void_p]
    optix.get_camera_target.restype = c_bool

    optix.set_camera_target.argtypes = [c_void_p]
    optix.set_camera_target.restype = c_bool

    optix.get_camera.argtypes = [c_uint]
    optix.get_camera.restype = c_wchar_p

    optix.get_light_shading.restype = c_int

    optix.get_light_pos.argtypes = [c_int, c_void_p]
    optix.get_light_pos.restype = c_bool

    optix.get_light_color.argtypes = [c_int, c_void_p]
    optix.get_light_color.restype = c_bool

    optix.get_light_u.argtypes = [c_int, c_void_p]
    optix.get_light_u.restype = c_bool

    optix.get_light_v.argtypes = [c_int, c_void_p]
    optix.get_light_v.restype = c_bool

    optix.get_light_r.argtypes = [c_int]
    optix.get_light_r.restype = c_float

    optix.set_light_shading.argtypes = [c_int]
    optix.set_light_shading.restype = c_bool

    optix.setup_spherical_light.argtypes = [c_void_p, c_void_p, c_float, c_bool]
    optix.setup_spherical_light.restype = c_int

    optix.setup_parallelogram_light.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_bool]
    optix.setup_parallelogram_light.restype = c_int

    optix.update_light.argtypes = [c_int, c_void_p, c_void_p, c_float, c_void_p, c_void_p]
    optix.update_light.restype = c_bool

    optix.fit_light.argtypes = [c_int, c_uint, c_float, c_float, c_float]
    optix.fit_light.restype = c_bool

    optix.get_object_at.argtypes = [c_int, c_int, POINTER(c_uint), POINTER(c_uint)]
    optix.get_object_at.restype = c_bool

    optix.get_hit_at.argtypes = [c_int, c_int, POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float)]
    optix.get_hit_at.restype = c_bool

    optix.register_launch_finished_callback.argtypes = [PARAM_INT_CALLBACK]
    optix.register_launch_finished_callback.restype = c_bool

    optix.register_accum_done_callback.argtypes = [PARAM_NONE_CALLBACK]
    optix.register_accum_done_callback.restype = c_bool

    optix.register_scene_rt_starting_callback.argtypes = [PARAM_NONE_CALLBACK]
    optix.register_scene_rt_starting_callback.restype = c_bool

    optix.register_start_scene_compute_callback.argtypes = [PARAM_INT_CALLBACK]
    optix.register_start_scene_compute_callback.restype = c_bool

    optix.register_scene_rt_completed_callback.argtypes = [PARAM_INT_CALLBACK]
    optix.register_scene_rt_completed_callback.restype = c_bool

    optix.get_min_accumulation_step.restype = c_int

    optix.set_min_accumulation_step.argtypes = [c_int]
    optix.set_min_accumulation_step.restype = c_bool

    optix.get_max_accumulation_frames.restype = c_int

    optix.set_max_accumulation_frames.argtypes = [c_int]
    optix.set_max_accumulation_frames.restype = c_bool

    optix.encoder_create.argtypes = [c_int, c_int, c_int, c_int, c_int]
    optix.encoder_create.restype = c_bool

    optix.encoder_start.argtypes = [c_wchar_p, c_uint]
    optix.encoder_start.restype = c_bool

    optix.encoder_stop.restype = c_bool

    optix.encoder_is_open.restype = c_bool

    optix.encoded_frames.restype = c_int
    optix.encoding_frames.restype = c_int

    optix.open_simplex_2d.argtypes = [c_void_p, c_void_p, c_int]
    optix.open_simplex_3d.argtypes = [c_void_p, c_void_p, c_int]
    optix.open_simplex_4d.argtypes = [c_void_p, c_void_p, c_int]

    optix.set_gpu_architecture.argtypes = [c_int]

    optix.set_library_dir.argtypes = [c_wchar_p]

    optix.set_include_dir.argtypes = [c_wchar_p]

    optix.get_display_scaling.restype = c_float

    optix.test_library.argtypes = [c_int]
    optix.test_library.restype = c_bool

    return optix

class _ClrOptiX:
    """
    Pythonnet wrapper for RnD.SharpOptiX library.
    """

    def __init__(self):

        #c_encoder = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), BIN_PATH, "librndSharpEncoder.so"))
        c_optix = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), BIN_PATH, "liboptix.so.6.0.0"))
        c_optixu = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), BIN_PATH, "liboptixu.so.6.0.0"))
        c_rnd = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), BIN_PATH, "librndSharpOptiX.so"))

        json_name = os.path.join(os.path.dirname(__file__), BIN_PATH, "Newtonsoft.Json.dll")
        tiff_name = os.path.join(os.path.dirname(__file__), BIN_PATH, "BitMiracle.LibTiff.NET.dll")
        rnd_name = os.path.join(os.path.dirname(__file__), BIN_PATH, "RnD.SharpOptiX.dll")

        head, tail = os.path.split(rnd_name)
        sys.path.append(head)

        json_assembly = clr.System.Reflection.Assembly.LoadFile(json_name)
        tiff_assembly = clr.System.Reflection.Assembly.LoadFile(tiff_name)
        rnd_assembly = clr.System.Reflection.Assembly.LoadFile(rnd_name)

        clr.AddReference(os.path.splitext(tail)[0])

        self._optix = rnd_assembly.CreateInstance("RnD.SharpOptiX.Py.PyOptiX")

    def create_empty_scene(self, width, height, buf_ptr, buf_size):
        return self._optix.create_empty_scene_ptr(width, height, IntPtr.__overloads__[int](buf_ptr), buf_size)

    def resize_scene(self, width, height, buf_ptr, buf_size):
        return self._optix.resize_scene_ptr(width, height, IntPtr.__overloads__[int](buf_ptr), buf_size)

    def open_simplex_2d(self, noise_ptr, inputs_ptr, length):
        return self._optix.open_simplex_2d_ptr(IntPtr.__overloads__[int](noise_ptr), IntPtr.__overloads__[int](inputs_ptr), length)

    def open_simplex_3d(self, noise_ptr, inputs_ptr, length):
        return self._optix.open_simplex_3d_ptr(IntPtr.__overloads__[int](noise_ptr), IntPtr.__overloads__[int](inputs_ptr), length)

    def open_simplex_4d(self, noise_ptr, inputs_ptr, length):
        return self._optix.open_simplex_4d_ptr(IntPtr.__overloads__[int](noise_ptr), IntPtr.__overloads__[int](inputs_ptr), length)

    def set_gpu_architecture(self, arch): self._optix.set_gpu_architecture(arch)

    def set_library_dir(self, path): self._optix.set_library_dir(path)

    def set_include_dir(self, path): self._optix.set_include_dir(path)

    def get_display_scaling(self): return self._optix.get_display_scaling()

    def test_library(self, x): return self._optix.test_library(x)


def load_optix():
    """
    Load RnD.SharpOptiX library, setup CUDA lib and include folders.
    """
    global sharp_optix
    if sharp_optix is not None: return sharp_optix

    if PLATFORM == "Windows":
        optix = _load_optix_win()
    elif PLATFORM == "Linux":
        optix = _ClrOptiX()
    else:
        raise NotImplementedError

    package_dir = os.path.dirname(__file__)
    optix.set_library_dir(os.path.join(package_dir, BIN_PATH))
    optix.set_include_dir(os.path.join(package_dir, BIN_PATH, "cuda"))

    sharp_optix = optix

    return sharp_optix

