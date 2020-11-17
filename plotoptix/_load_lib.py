"""Import RnD.SharpOptiX library.

https://github.com/rnd-team-dev/plotoptix/blob/master/LICENSE.txt

Have a look at examples on GitHub: https://github.com/rnd-team-dev/plotoptix.
"""

import os, platform, sys

from ctypes import cdll, CFUNCTYPE, POINTER, byref, cast, c_float, c_uint, c_int, c_long, c_longlong, c_bool, c_char_p, c_wchar_p, c_void_p

BIN_PATH = "bin"

PLATFORM = platform.system()
if PLATFORM == "Linux":
    import clr
    from System import IntPtr, Int64

PARAM_NONE_CALLBACK = CFUNCTYPE(None)
PARAM_INT_CALLBACK = CFUNCTYPE(None, c_int)


sharp_optix = None
def _load_optix_win():
    """Load RnD.SharpOptiX library with ctypes, setup arguments and return types.
    """
    dll_name = os.path.join(os.path.dirname(__file__), BIN_PATH, "RnD.SharpOptiX.dll")

    optix = cdll.LoadLibrary(dll_name)

    optix.create_empty_scene.argtypes = [c_int, c_int, c_void_p, c_int, c_int]
    optix.create_empty_scene.restype = c_bool

    optix.get_miss_program.restype = c_int

    optix.set_miss_program.argtypes = [c_int, c_bool]
    optix.set_miss_program.restype = c_bool

    optix.create_scene_from_json.argtypes = [c_wchar_p, c_int, c_int, c_void_p, c_int]
    optix.create_scene_from_json.restype = c_bool

    optix.create_scene_from_file.argtypes = [c_wchar_p, c_int, c_int, c_void_p, c_int]
    optix.create_scene_from_file.restype = c_bool

    optix.load_scene_from_json.argtypes = [c_wchar_p]
    optix.load_scene_from_json.restype = c_bool

    optix.load_scene_from_file.argtypes = [c_wchar_p]
    optix.load_scene_from_file.restype = c_bool

    optix.get_scene_metadata.restype = c_wchar_p

    optix.save_scene_to_json.restype = c_wchar_p

    optix.save_scene_to_file.argtypes = [c_wchar_p]
    optix.save_scene_to_file.restype = c_bool

    optix.save_image_to_file.argtypes = [c_wchar_p]
    optix.save_image_to_file.restype = c_bool

    optix.save_image_to_file_16bps.argtypes = [c_wchar_p]
    optix.save_image_to_file_16bps.restype = c_bool

    optix.save_image_to_file_32bps.argtypes = [c_wchar_p]
    optix.save_image_to_file_32bps.restype = c_bool

    optix.get_output.argtypes = [c_void_p, c_int, c_int, c_int]
    optix.get_output.restype = c_bool

    optix.get_fps.restype = c_float

    optix.start_rt.restype = c_bool
    optix.stop_rt.restype = c_bool

    optix.break_launch.restype = c_bool

    optix.set_compute_paused.argtypes = [c_bool]
    optix.set_compute_paused.restype = c_bool

    optix.is_defined.argtypes = [c_wchar_p]
    optix.is_defined.restype = c_bool

    optix.has_texture.argtypes = [c_wchar_p]
    optix.has_texture.restype = c_bool

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

    optix.set_texture_1d.argtypes = [c_wchar_p, c_void_p, c_int, c_uint, c_int, c_bool, c_bool]
    optix.set_texture_1d.restype = c_bool

    optix.set_texture_2d.argtypes = [c_wchar_p, c_void_p, c_int, c_int, c_uint, c_int, c_bool, c_bool]
    optix.set_texture_2d.restype = c_bool

    optix.load_texture_2d.argtypes = [c_wchar_p, c_wchar_p, c_float, c_float, c_float, c_float, c_uint, c_int, c_bool]
    optix.load_texture_2d.restype = c_bool

    optix.set_displacement.argtypes = [c_wchar_p, c_void_p, c_int, c_int, c_int, c_bool, c_bool]
    optix.set_displacement.restype = c_bool

    optix.load_displacement.argtypes = [c_wchar_p, c_wchar_p, c_float, c_float, c_int, c_bool]
    optix.load_displacement.restype = c_bool

    optix.resize_scene.argtypes = [c_int, c_int, POINTER(c_longlong), POINTER(c_int), POINTER(c_longlong), POINTER(c_int), POINTER(c_longlong), POINTER(c_int), POINTER(c_longlong), POINTER(c_int), POINTER(c_longlong), POINTER(c_int), POINTER(c_longlong), POINTER(c_int)]
    optix.resize_scene.restype = c_bool

    optix.get_device_buffers.argtypes = [POINTER(c_longlong), POINTER(c_int), POINTER(c_longlong), POINTER(c_int), POINTER(c_longlong), POINTER(c_int), POINTER(c_longlong), POINTER(c_int), POINTER(c_longlong), POINTER(c_int), POINTER(c_longlong), POINTER(c_int)]
    optix.get_device_buffers.restype = c_bool

    optix.get_device_buf.argtypes = [POINTER(c_longlong), POINTER(c_int)]
    optix.get_device_buf.restype = c_bool

    optix.pin_geometry_buffer.argtypes = [c_wchar_p, c_int, POINTER(c_longlong), POINTER(c_longlong), POINTER(c_int), POINTER(c_uint)]
    optix.pin_geometry_buffer.restype = c_bool

    optix.unpin_geometry_buffer.argtypes = [c_wchar_p, c_int]
    optix.unpin_geometry_buffer.restype = c_bool

    optix.get_material.argtypes = [c_wchar_p]
    optix.get_material.restype = c_wchar_p

    optix.setup_material.argtypes = [c_wchar_p, c_wchar_p]
    optix.setup_material.restype = c_bool

    optix.set_normal_tilt.argtypes = [c_wchar_p, c_void_p, c_int, c_int, c_int, c_int, c_bool, c_bool]
    optix.set_normal_tilt.restype = c_bool

    optix.load_normal_tilt.argtypes = [c_wchar_p, c_wchar_p, c_int, c_int, c_float, c_float, c_bool]
    optix.load_normal_tilt.restype = c_bool

    optix.set_correction_curve.argtypes = [c_void_p, c_int, c_int, c_int, c_float, c_bool]
    optix.set_correction_curve.restype = c_bool

    optix.add_postproc.argtypes = [c_int, c_bool]
    optix.add_postproc.restype = c_bool

    optix.setup_geometry.argtypes = [c_int, c_int, c_wchar_p, c_wchar_p, c_bool, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    optix.setup_geometry.restype = c_uint

    optix.update_geometry.argtypes = [c_wchar_p, c_wchar_p, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    optix.update_geometry.restype = c_uint

    optix.get_geometry_size.argtypes = [c_wchar_p]
    optix.get_geometry_size.restype = c_int

    optix.get_faces_count.argtypes = [c_wchar_p]
    optix.get_faces_count.restype = c_int

    optix.get_surface_size.argtypes = [c_wchar_p, POINTER(c_uint), POINTER(c_uint)]
    optix.get_surface_size.restype = c_bool

    optix.update_surface.argtypes = [c_wchar_p, c_wchar_p, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_float, c_float, c_float, c_float, c_float]
    optix.update_surface.restype = c_uint

    optix.setup_surface.argtypes = [c_int, c_wchar_p, c_wchar_p, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_float, c_float, c_float, c_float, c_float, c_bool]
    optix.setup_surface.restype = c_uint

    optix.update_psurface.argtypes = [c_wchar_p, c_wchar_p, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    optix.update_psurface.restype = c_uint

    optix.setup_psurface.argtypes = [c_int, c_wchar_p, c_wchar_p, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_bool, c_bool, c_bool]
    optix.setup_psurface.restype = c_uint

    optix.setup_graph.argtypes = [c_wchar_p, c_wchar_p, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    optix.setup_graph.restype = c_uint

    optix.update_graph.argtypes = [c_wchar_p, c_wchar_p, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    optix.update_graph.restype = c_uint

    optix.setup_mesh.argtypes = [c_wchar_p, c_wchar_p, c_int, c_int, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_bool]
    optix.setup_mesh.restype = c_uint

    optix.update_mesh.argtypes = [c_wchar_p, c_wchar_p, c_int, c_int, c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
    optix.update_mesh.restype = c_uint

    optix.load_mesh_obj.argtypes = [c_wchar_p, c_wchar_p, c_wchar_p, c_wchar_p, c_void_p, c_bool]
    optix.load_mesh_obj.restype = c_wchar_p

    optix.load_multiple_mesh_obj.argtypes = [c_wchar_p, c_wchar_p, c_wchar_p, c_wchar_p]
    optix.load_multiple_mesh_obj.restype = c_wchar_p

    optix.load_merged_mesh_obj.argtypes = [c_wchar_p, c_wchar_p, c_wchar_p, c_void_p, c_bool]
    optix.load_merged_mesh_obj.restype = c_uint

    optix.move_geometry.argtypes = [c_wchar_p, c_float, c_float, c_float, c_bool]
    optix.move_geometry.restype = c_bool

    optix.move_primitive.argtypes = [c_wchar_p, c_long, c_float, c_float, c_float, c_bool]
    optix.move_primitive.restype = c_bool

    optix.move_geometry_in_view.argtypes = [c_wchar_p, c_float, c_float, c_float, c_bool]
    optix.move_geometry_in_view.restype = c_bool

    optix.move_primitive_in_view.argtypes = [c_wchar_p, c_long, c_float, c_float, c_float, c_bool]
    optix.move_primitive_in_view.restype = c_bool

    optix.rotate_geometry.argtypes = [c_wchar_p, c_float, c_float, c_float, c_bool]
    optix.rotate_geometry.restype = c_bool

    optix.rotate_primitive.argtypes = [c_wchar_p, c_long, c_float, c_float, c_float, c_bool]
    optix.rotate_primitive.restype = c_bool

    optix.rotate_geometry_in_view.argtypes = [c_wchar_p, c_float, c_float, c_float, c_bool]
    optix.rotate_geometry_in_view.restype = c_bool

    optix.rotate_primitive_in_view.argtypes = [c_wchar_p, c_long, c_float, c_float, c_float, c_bool]
    optix.rotate_primitive_in_view.restype = c_bool

    optix.rotate_geometry_about.argtypes = [c_wchar_p, c_float, c_float, c_float, c_float, c_float, c_float, c_bool]
    optix.rotate_geometry_about.restype = c_bool

    optix.rotate_primitive_about.argtypes = [c_wchar_p, c_long, c_float, c_float, c_float, c_float, c_float, c_float, c_bool]
    optix.rotate_primitive_about.restype = c_bool

    optix.scale_geometry.argtypes = [c_wchar_p, c_float, c_bool]
    optix.scale_geometry.restype = c_bool

    optix.scale_primitive.argtypes = [c_wchar_p, c_long, c_float, c_bool]
    optix.scale_primitive.restype = c_bool

    optix.scale_geometry_c.argtypes = [c_wchar_p, c_float, c_float, c_float, c_float, c_bool]
    optix.scale_geometry_c.restype = c_bool

    optix.scale_primitive_c.argtypes = [c_wchar_p, c_long, c_float, c_float, c_float, c_float, c_bool]
    optix.scale_primitive_c.restype = c_bool

    optix.scale_geometry_xyz.argtypes = [c_wchar_p, c_float, c_float, c_float, c_bool]
    optix.scale_geometry_xyz.restype = c_bool

    optix.scale_primitive_xyz.argtypes = [c_wchar_p, c_long, c_float, c_float, c_float, c_bool]
    optix.scale_primitive_xyz.restype = c_bool

    optix.scale_geometry_xyz_c.argtypes = [c_wchar_p, c_float, c_float, c_float, c_float, c_float, c_float, c_bool]
    optix.scale_geometry_xyz_c.restype = c_bool

    optix.scale_primitive_xyz_c.argtypes = [c_wchar_p, c_long, c_float, c_float, c_float, c_float, c_float, c_float, c_bool]
    optix.scale_primitive_xyz_c.restype = c_bool

    optix.update_geom_buffers.argtypes = [c_wchar_p, c_uint, c_bool]
    optix.update_geom_buffers.restype = c_bool

    optix.set_coordinates_geom.argtypes = [c_int, c_float]
    optix.set_coordinates_geom.restype = c_bool

    optix.setup_camera.argtypes = [c_wchar_p, c_int, c_void_p, c_void_p, c_void_p, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_bool, c_wchar_p, c_bool]
    optix.setup_camera.restype = c_int

    optix.update_camera.argtypes = [c_wchar_p, c_void_p, c_void_p, c_void_p, c_float, c_float, c_float]
    optix.update_camera.restype = c_bool

    optix.fit_camera.argtypes = [c_uint, c_wchar_p, c_float]
    optix.fit_camera.restype = c_bool

    optix.get_current_camera.restype = c_uint

    optix.set_current_camera.argtypes = [c_wchar_p]
    optix.set_current_camera.restype = c_bool

    optix.move_camera_by.argtypes = [c_float, c_float, c_float]
    optix.move_camera_by.restype = c_bool

    optix.move_camera_by_local.argtypes = [c_float, c_float, c_float]
    optix.move_camera_by_local.restype = c_bool

    optix.rotate_camera_by.argtypes = [c_float, c_float, c_float, c_float, c_float, c_float]
    optix.rotate_camera_by.restype = c_bool

    optix.rotate_camera_by_local.argtypes = [c_float, c_float, c_float, c_float, c_float, c_float]
    optix.rotate_camera_by_local.restype = c_bool

    optix.rotate_camera_eye.argtypes = [c_int, c_int, c_int, c_int]
    optix.rotate_camera_eye.restype = c_bool

    optix.rotate_camera_eye_by.argtypes = [c_float, c_float, c_float]
    optix.rotate_camera_eye_by.restype = c_bool

    optix.rotate_camera_eye_by_local.argtypes = [c_float, c_float, c_float]
    optix.rotate_camera_eye_by_local.restype = c_bool

    optix.rotate_camera_tgt.argtypes = [c_int, c_int, c_int, c_int]
    optix.rotate_camera_tgt.restype = c_bool

    optix.rotate_camera_tgt_by.argtypes = [c_float, c_float, c_float]
    optix.rotate_camera_tgt_by.restype = c_bool

    optix.rotate_camera_tgt_by_local.argtypes = [c_float, c_float, c_float]
    optix.rotate_camera_tgt_by_local.restype = c_bool

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

    optix.get_camera_glock.argtypes = [c_uint]
    optix.get_camera_glock.restype = c_bool

    optix.set_camera_glock.argtypes = [c_bool]
    optix.set_camera_glock.restype = c_bool

    optix.get_camera.argtypes = [c_uint]
    optix.get_camera.restype = c_wchar_p

    optix.get_light_shading.restype = c_int

    optix.get_light_handle.argtypes = [c_int, c_int]
    optix.get_light_handle.restype = c_int

    optix.get_light_pos.argtypes = [c_wchar_p, c_void_p]
    optix.get_light_pos.restype = c_bool

    optix.get_light_color.argtypes = [c_wchar_p, c_void_p]
    optix.get_light_color.restype = c_bool

    optix.get_light_u.argtypes = [c_wchar_p, c_void_p]
    optix.get_light_u.restype = c_bool

    optix.get_light_v.argtypes = [c_wchar_p, c_void_p]
    optix.get_light_v.restype = c_bool

    optix.get_light_r.argtypes = [c_wchar_p]
    optix.get_light_r.restype = c_float

    optix.get_light.argtypes = [c_wchar_p]
    optix.get_light.restype = c_wchar_p

    optix.set_light_shading.argtypes = [c_int]
    optix.set_light_shading.restype = c_bool

    optix.setup_spherical_light.argtypes = [c_wchar_p, c_void_p, c_void_p, c_float, c_bool]
    optix.setup_spherical_light.restype = c_int

    optix.setup_parallelogram_light.argtypes = [c_wchar_p, c_void_p, c_void_p, c_void_p, c_void_p, c_bool]
    optix.setup_parallelogram_light.restype = c_int

    optix.update_light.argtypes = [c_wchar_p, c_void_p, c_void_p, c_float, c_void_p, c_void_p]
    optix.update_light.restype = c_bool

    optix.fit_light.argtypes = [c_wchar_p, c_uint, c_float, c_float, c_float]
    optix.fit_light.restype = c_bool

    optix.dolly_light.argtypes = [c_wchar_p, c_float]
    optix.dolly_light.restype = c_bool

    optix.move_light_in_view.argtypes = [c_wchar_p, c_float, c_float, c_float]
    optix.move_light_in_view.restype = c_bool

    optix.rotate_light_in_view.argtypes = [c_wchar_p, c_float, c_float, c_float]
    optix.rotate_light_in_view.restype = c_bool

    optix.scale_light.argtypes = [c_wchar_p, c_float]
    optix.scale_light.restype = c_bool


    optix.get_object_at.argtypes = [c_int, c_int, POINTER(c_uint), POINTER(c_uint), POINTER(c_uint)]
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

    optix.get_compute_timeout.restype = c_int

    optix.set_compute_timeout.argtypes = [c_int]
    optix.set_compute_timeout.restype = c_bool

    optix.get_rt_timeout.restype = c_int

    optix.set_rt_timeout.argtypes = [c_int]
    optix.set_rt_timeout.restype = c_bool

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

    optix.get_image_meta.argtypes = [c_wchar_p, POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int)]
    optix.get_image_meta.restype = c_bool

    optix.read_image.argtypes = [c_wchar_p, c_void_p, c_int, c_int, c_int, c_int]
    optix.read_image.restype = c_bool

    optix.read_image_normalized.argtypes = [c_wchar_p, c_void_p, c_int, c_int, c_int]
    optix.read_image_normalized.restype = c_bool

    optix.get_gpu_architecture.restype = c_int
    optix.set_gpu_architecture.argtypes = [c_int]

    optix.get_n_gpu_architecture.argtypes = [c_uint]
    optix.get_n_gpu_architecture.restype = c_int

    optix.set_library_dir.argtypes = [c_wchar_p]

    optix.set_include_dir.argtypes = [c_wchar_p]

    optix.get_display_scaling.restype = c_float

    optix.test_library.argtypes = [c_int]
    optix.test_library.restype = c_bool

    return optix

class _ClrOptiX:
    """Pythonnet wrapper for RnD.SharpOptiX library; provides identical interface
    as RnD.SharpOptiX library loaded in Windows with ctypes.
    """

    def __init__(self):

        try:
            c_encoder = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), BIN_PATH, "librndSharpEncoder.so"))
            self._encoder_available = True
        except:
            print(82 * "*"); print(82 * "*")
            print("Video encoding library initialization failed, encoding features are not available.")
            print(82 * "*"); print(82 * "*")
            self._encoder_available = False

        try:
            c_rnd = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), BIN_PATH, "librndSharpOptiX7.so"))
        except:
            print(80 * "*"); print(80 * "*")
            print("Low level ray tracing libraries initialization failed, cannot continue.")
            print(80 * "*"); print(80 * "*")
            raise ImportError

        json_name = os.path.join(os.path.dirname(__file__), BIN_PATH, "Newtonsoft.Json.dll")
        tiff_name = os.path.join(os.path.dirname(__file__), BIN_PATH, "BitMiracle.LibTiff.NET.dll")
        rnd_name = os.path.join(os.path.dirname(__file__), BIN_PATH, "RnD.SharpOptiX.dll")

        head, tail = os.path.split(rnd_name)
        sys.path.append(head)

        try:
            json_assembly = clr.System.Reflection.Assembly.LoadFile(json_name)
            tiff_assembly = clr.System.Reflection.Assembly.LoadFile(tiff_name)
            rnd_assembly = clr.System.Reflection.Assembly.LoadFile(rnd_name)
        except:
            print(80 * "*"); print(80 * "*")
            print(".NET ray tracing libraries initialization failed, cannot continue.")
            print(80 * "*"); print(80 * "*")
            raise ImportError

        clr.AddReference(os.path.splitext(tail)[0])

        self._optix = rnd_assembly.CreateInstance("RnD.SharpOptiX.Py.PyOptiX7")

    def refresh_scene(self): self._optix.refresh_scene()

    def destroy_scene(self): self._optix.destroy_scene()

    def create_empty_scene(self, width, height, dev_ptr, dev_size, log_level):
        return self._optix.create_empty_scene_ptr(width, height,
                                                  IntPtr.__overloads__[Int64](dev_ptr), dev_size,
                                                  log_level)

    def get_miss_program(self): return self._optix.get_miss_program()
    def set_miss_program(self, algorithm, refresh): return self._optix.set_miss_program(algorithm, refresh)

    def create_scene_from_json(self, jstr, width, height, dev_ptr, dev_size):
        return self._optix.create_scene_from_json_ptr(jstr, width, height,
                                                      IntPtr.__overloads__[Int64](dev_ptr), dev_size)

    def create_scene_from_file(self, jstr, width, height, dev_ptr, dev_size):
        return self._optix.create_scene_from_file_ptr(jstr, width, height,
                                                      IntPtr.__overloads__[Int64](dev_ptr), dev_size)

    def load_scene_from_json(self, jstr): return self._optix.load_scene_from_json(jstr)

    def load_scene_from_file(self, fname): return self._optix.load_scene_from_file(fname)

    def get_scene_metadata(self): return self._optix.get_scene_metadata()

    def save_scene_to_json(self): return self._optix.save_scene_to_json()

    def save_scene_to_file(self, fname): return self._optix.save_scene_to_file(fname)

    def save_image_to_file(self, fname): return self._optix.save_image_to_file(fname)

    def save_image_to_file_16bps(self, fname): return self._optix.save_image_to_file_16bps(fname)

    def save_image_to_file_32bps(self, fname): return self._optix.save_image_to_file_32bps(fname)

    def get_output(self, buf_ptr, buf_size, depth, channels):
        return self._optix.get_output_ptr(IntPtr.__overloads__[Int64](buf_ptr), buf_size, depth, channels)


    def get_fps(self): return self._optix.get_fps()

    def start_rt(self): return self._optix.start_rt()
    def stop_rt(self): return self._optix.stop_rt()

    def break_launch(self): return self._optix.break_launch()

    def set_compute_paused(self, state): return self._optix.set_compute_paused(state)

    def is_defined(self, name): return self._optix.is_defined(name)

    def has_texture(self, name): return self._optix.has_texture(name)

    def get_int(self, name, x_ref):
        return self._optix.get_int_ptr(name,
                                       IntPtr.__overloads__[Int64](cast(x_ref, c_void_p).value))

    def set_int(self, name, x, refresh): return self._optix.set_int(name, x, refresh)

    def get_uint(self, name, x_ref):
        return self._optix.get_uint_ptr(name,
                                        IntPtr.__overloads__[Int64](cast(x_ref, c_void_p).value))

    def set_uint(self, name, x, refresh): return self._optix.set_uint(name, x, refresh)

    def get_uint2(self, name, x_ref, y_ref):
        return self._optix.get_uint2_ptr(name,
                                         IntPtr.__overloads__[Int64](cast(x_ref, c_void_p).value),
                                         IntPtr.__overloads__[Int64](cast(y_ref, c_void_p).value))

    def set_uint2(self, name, x, y, refresh): return self._optix.set_uint2(name, x, y, refresh)

    def get_float(self, name, x_ref):
        return self._optix.get_float_ptr(name,
                                         IntPtr.__overloads__[Int64](cast(x_ref, c_void_p).value))

    def set_float(self, name, x, refresh): return self._optix.set_float(name, x, refresh)

    def get_float2(self, name, x_ref, y_ref):
        return self._optix.get_float2_ptr(name,
                                          IntPtr.__overloads__[Int64](cast(x_ref, c_void_p).value),
                                          IntPtr.__overloads__[Int64](cast(y_ref, c_void_p).value))

    def set_float2(self, name, x, y, refresh): return self._optix.set_float2(name, x, y, refresh)

    def get_float3(self, name, x_ref, y_ref, z_ref):
        return self._optix.get_float3_ptr(name,
                                          IntPtr.__overloads__[Int64](cast(x_ref, c_void_p).value),
                                          IntPtr.__overloads__[Int64](cast(y_ref, c_void_p).value),
                                          IntPtr.__overloads__[Int64](cast(z_ref, c_void_p).value))

    def set_float3(self, name, x, y, z, refresh): return self._optix.set_float3(name, x, y, z, refresh)

    def set_texture_1d(self, name, data_ptr, length, tformat, addr_mode, keep_on_host, refresh):
        return self._optix.set_texture_1d_ptr(name,
                                              IntPtr.__overloads__[Int64](data_ptr),
                                              length, tformat, addr_mode, keep_on_host, refresh)

    def set_texture_2d(self, name, data_ptr, width, height, tformat, addr_mode, keep_on_host, refresh):
        return self._optix.set_texture_2d_ptr(name,
                                              IntPtr.__overloads__[Int64](data_ptr),
                                              width, height, tformat, addr_mode, keep_on_host, refresh)

    def load_texture_2d(self, tex_name, file_name, prescale, baseline, exposure, gamma, tformat, addr_mode, refresh):
        return self._optix.load_texture_2d(tex_name, file_name, prescale, baseline, exposure, gamma, tformat, addr_mode, refresh)

    def set_displacement(self, obj_name, data_ptr, width, height, addr_mode, keep_on_host, refresh):
        return self._optix.set_displacement_ptr(obj_name,
                                                IntPtr.__overloads__[Int64](data_ptr),
                                                width, height, addr_mode, keep_on_host, refresh)

    def load_displacement(self, obj_name, file_name, prescale, baseline, addr_mode, refresh):
        return self._optix.load_displacement(obj_name, file_name, prescale, baseline, addr_mode, refresh)


    def resize_scene(self, width, height,
                     img_ptr_ref, img_size_ref,
                     raw_ptr_ref, raw_size_ref,
                     hit_ptr_ref, hit_size_ref,
                     geo_ptr_ref, geo_size_ref,
                     albedo_ptr_ref, albedo_size_ref,
                     normal_ptr_ref, normal_size_ref):
        return self._optix.resize_scene_ptr(width, height,
                                            IntPtr.__overloads__[Int64](cast(img_ptr_ref, c_void_p).value),
                                            IntPtr.__overloads__[Int64](cast(img_size_ref, c_void_p).value),
                                            IntPtr.__overloads__[Int64](cast(raw_ptr_ref, c_void_p).value),
                                            IntPtr.__overloads__[Int64](cast(raw_size_ref, c_void_p).value),
                                            IntPtr.__overloads__[Int64](cast(hit_ptr_ref, c_void_p).value),
                                            IntPtr.__overloads__[Int64](cast(hit_size_ref, c_void_p).value),
                                            IntPtr.__overloads__[Int64](cast(geo_ptr_ref, c_void_p).value),
                                            IntPtr.__overloads__[Int64](cast(geo_size_ref, c_void_p).value),
                                            IntPtr.__overloads__[Int64](cast(albedo_ptr_ref, c_void_p).value),
                                            IntPtr.__overloads__[Int64](cast(albedo_size_ref, c_void_p).value),
                                            IntPtr.__overloads__[Int64](cast(normal_ptr_ref, c_void_p).value),
                                            IntPtr.__overloads__[Int64](cast(normal_size_ref, c_void_p).value))

    def get_device_buffers(self, width, height,
                     img_ptr_ref, img_size_ref,
                     raw_ptr_ref, raw_size_ref,
                     hit_ptr_ref, hit_size_ref,
                     geo_ptr_ref, geo_size_ref,
                     albedo_ptr_ref, albedo_size_ref,
                     normal_ptr_ref, normal_size_ref):
        return self._optix.get_device_buffers_ptr(
                                            IntPtr.__overloads__[Int64](cast(img_ptr_ref, c_void_p).value),
                                            IntPtr.__overloads__[Int64](cast(img_size_ref, c_void_p).value),
                                            IntPtr.__overloads__[Int64](cast(raw_ptr_ref, c_void_p).value),
                                            IntPtr.__overloads__[Int64](cast(raw_size_ref, c_void_p).value),
                                            IntPtr.__overloads__[Int64](cast(hit_ptr_ref, c_void_p).value),
                                            IntPtr.__overloads__[Int64](cast(hit_size_ref, c_void_p).value),
                                            IntPtr.__overloads__[Int64](cast(geo_ptr_ref, c_void_p).value),
                                            IntPtr.__overloads__[Int64](cast(geo_size_ref, c_void_p).value),
                                            IntPtr.__overloads__[Int64](cast(albedo_ptr_ref, c_void_p).value),
                                            IntPtr.__overloads__[Int64](cast(albedo_size_ref, c_void_p).value),
                                            IntPtr.__overloads__[Int64](cast(normal_ptr_ref, c_void_p).value),
                                            IntPtr.__overloads__[Int64](cast(normal_size_ref, c_void_p).value))

    def get_device_buf(self, buf_ptr_ref, gpu_size_ref):
        return self._optix.get_device_buf_ptr(IntPtr.__overloads__[Int64](cast(buf_ptr_ref, c_void_p).value),
                                              IntPtr.__overloads__[Int64](cast(gpu_size_ref, c_void_p).value))

    def pin_geometry_buffer(self, name, kind, buf_ptr_ref, shape_ptr_ref, shape_size_ref, data_type_ref):
        return self._optix.pin_geometry_buffer_ptr(name, kind,
                                                   IntPtr.__overloads__[Int64](cast(buf_ptr_ref, c_void_p).value),
                                                   IntPtr.__overloads__[Int64](cast(shape_ptr_ref, c_void_p).value),
                                                   IntPtr.__overloads__[Int64](cast(shape_size_ref, c_void_p).value),
                                                   IntPtr.__overloads__[Int64](cast(data_type_ref, c_void_p).value))

    def unpin_geometry_buffer(self, name, kind): return self._optix.unpin_geometry_buffer(name, kind)

    def get_material(self, name): return self._optix.get_material(name)

    def setup_material(self, name, jstr): return self._optix.setup_material(name, jstr)

    def set_normal_tilt(self, mat_name, data_ptr, width, height, mapping, addr_mode, keep_on_host, refresh):
        return self._optix.set_normal_tilt_ptr(mat_name,
                                               IntPtr.__overloads__[Int64](data_ptr),
                                               width, height, mapping, addr_mode, keep_on_host, refresh)

    def load_normal_tilt(self, mat_name, file_name, mapping, addr_mode, prescale, baseline, refresh):
        return self._optix.load_normal_tilt(mat_name, file_name, mapping, addr_mode, prescale, baseline, refresh)


    def set_correction_curve(self, data_ptr, n_ctrl_points, n_curve_points, channel, vrange, refresh):
        return self._optix.set_correction_curve_ptr(IntPtr.__overloads__[Int64](data_ptr),
                                                    n_ctrl_points, n_curve_points, channel, vrange, refresh)

    def add_postproc(self, algorithm, refresh): return self._optix.add_postproc(algorithm, refresh)

    def setup_geometry(self, geomType, attrProg, name, material, rnd_missing, n_primitives, pos, cconst, c, r, u, v, w):
        return self._optix.setup_geometry_ptr(geomType, attrProg, name, material, rnd_missing, n_primitives,
                                              IntPtr.__overloads__[Int64](pos),
                                              IntPtr.__overloads__[Int64](cconst),
                                              IntPtr.__overloads__[Int64](c),
                                              IntPtr.__overloads__[Int64](r),
                                              IntPtr.__overloads__[Int64](u),
                                              IntPtr.__overloads__[Int64](v),
                                              IntPtr.__overloads__[Int64](w))

    def update_geometry(self, name, material, n_primitives, pos, cconst, c, r, u, v, w):
        return self._optix.update_geometry_ptr(name, material, n_primitives,
                                               IntPtr.__overloads__[Int64](pos),
                                               IntPtr.__overloads__[Int64](cconst),
                                               IntPtr.__overloads__[Int64](c),
                                               IntPtr.__overloads__[Int64](r),
                                               IntPtr.__overloads__[Int64](u),
                                               IntPtr.__overloads__[Int64](v),
                                               IntPtr.__overloads__[Int64](w))

    def get_geometry_size(self, name): return self._optix.get_geometry_size(name)

    def get_faces_count(self, name): return self._optix.get_faces_count(name)

    def get_surface_size(self, name, x_ref, z_ref):
        return self._optix.get_surface_size_ptr(name,
                                                IntPtr.__overloads__[Int64](cast(x_ref, c_void_p).value),
                                                IntPtr.__overloads__[Int64](cast(z_ref, c_void_p).value))

    def update_surface(self, name, material, x_size, z_size, pos, radii, norm, csurf, cfloor, x_min, x_max, z_min, z_max, floor_level):
        return self._optix.update_surface_ptr(name, material, x_size, z_size,
                                              IntPtr.__overloads__[Int64](pos),
                                              IntPtr.__overloads__[Int64](radii),
                                              IntPtr.__overloads__[Int64](norm),
                                              IntPtr.__overloads__[Int64](csurf),
                                              IntPtr.__overloads__[Int64](cfloor),
                                              x_min, x_max, z_min, z_max, floor_level)

    def setup_surface(self, geomType, name, material, x_size, z_size, pos, radii, norm, csurf, cfloor, x_min, x_max, z_min, z_max, floor_level, make_normals):
        return self._optix.setup_surface_ptr(geomType, name, material, x_size, z_size,
                                             IntPtr.__overloads__[Int64](pos),
                                             IntPtr.__overloads__[Int64](radii),
                                             IntPtr.__overloads__[Int64](norm),
                                             IntPtr.__overloads__[Int64](csurf),
                                             IntPtr.__overloads__[Int64](cfloor),
                                             x_min, x_max, z_min, z_max, floor_level, make_normals)

    def update_psurface(self, name, material, u_size, v_size, pos, radii, norm, cconst, c):
        return self._optix.update_psurface_ptr(name, material, u_size, v_size,
                                               IntPtr.__overloads__[Int64](pos),
                                               IntPtr.__overloads__[Int64](radii),
                                               IntPtr.__overloads__[Int64](norm),
                                               IntPtr.__overloads__[Int64](cconst),
                                               IntPtr.__overloads__[Int64](c))

    def setup_psurface(self, geomType, name, material, u_size, v_size, pos, radii, norm, cconst, c, wrap_u, wrap_v, make_normals):
        return self._optix.setup_psurface_ptr(geomType, name, material, u_size, v_size,
                                              IntPtr.__overloads__[Int64](pos),
                                              IntPtr.__overloads__[Int64](radii),
                                              IntPtr.__overloads__[Int64](norm),
                                              IntPtr.__overloads__[Int64](cconst),
                                              IntPtr.__overloads__[Int64](c),
                                              wrap_u, wrap_v, make_normals)

    def setup_graph(self, name, material, n_vtx, n_edges, pos, radii, edges, cconst, c):
        return self._optix.setup_graph_ptr(name, material, n_vtx, n_edges,
                                           IntPtr.__overloads__[Int64](pos),
                                           IntPtr.__overloads__[Int64](radii),
                                           IntPtr.__overloads__[Int64](edges),
                                           IntPtr.__overloads__[Int64](cconst),
                                           IntPtr.__overloads__[Int64](c))

    def update_graph(self, name, material, n_vtx, n_edges, pos, radii, edges, cconst, c):
        return self._optix.update_graph_ptr(name, material, n_vtx, n_tri, n_norm, n_uv,
                                           IntPtr.__overloads__[Int64](pos),
                                           IntPtr.__overloads__[Int64](radii),
                                           IntPtr.__overloads__[Int64](edges),
                                           IntPtr.__overloads__[Int64](cconst),
                                           IntPtr.__overloads__[Int64](c))

    def setup_mesh(self, name, material, n_vtx, n_tri, n_norm, n_uv, pos, cconst, c, vidx, norm, nidx, uvmap, uvidx, make_normals):
        return self._optix.setup_mesh_ptr(name, material, n_vtx, n_tri, n_norm, n_uv,
                                          IntPtr.__overloads__[Int64](pos),
                                          IntPtr.__overloads__[Int64](cconst),
                                          IntPtr.__overloads__[Int64](c),
                                          IntPtr.__overloads__[Int64](vidx),
                                          IntPtr.__overloads__[Int64](norm),
                                          IntPtr.__overloads__[Int64](nidx),
                                          IntPtr.__overloads__[Int64](uvmap),
                                          IntPtr.__overloads__[Int64](uvidx),
                                          make_normals)

    def update_mesh(self, name, material, n_vtx, n_tri, n_norm, n_uv, pos, vidx, cconst, c, norm, nidx, uvmap, uvidx):
        return self._optix.update_mesh_ptr(name, material, n_vtx, n_tri, n_norm, n_uv,
                                           IntPtr.__overloads__[Int64](pos),
                                           IntPtr.__overloads__[Int64](vidx),
                                           IntPtr.__overloads__[Int64](cconst),
                                           IntPtr.__overloads__[Int64](c),
                                           IntPtr.__overloads__[Int64](norm),
                                           IntPtr.__overloads__[Int64](nidx),
                                           IntPtr.__overloads__[Int64](uvmap),
                                           IntPtr.__overloads__[Int64](uvidx))

    def load_mesh_obj(self, file_name, mesh_name, material, parent, color, make_normals):
        return self._optix.load_mesh_obj_ptr(file_name, mesh_name, material, parent,
                                             IntPtr.__overloads__[Int64](color),
                                             make_normals)

    def load_multiple_mesh_obj(self, file_name, materials, default_mat, parent):
        return self._optix.load_multiple_mesh_obj_ptr(file_name, materials, default_mat, parent)

    def load_merged_mesh_obj(self, file_name, mesh_name, material, color, make_normals):
        return self._optix.load_merged_mesh_obj_ptr(file_name, mesh_name, material,
                                                    IntPtr.__overloads__[Int64](color),
                                                    make_normals)

    def move_geometry(self, name, x, y, z, update): return self._optix.move_geometry(name, x, y, z, update)

    def move_primitive(self, name, idx, x, y, z, update): return self._optix.move_primitive(name, idx, x, y, z, update)

    def move_geometry_in_view(self, name, x, y, z, update): return self._optix.move_geometry_in_view(name, x, y, z, update)

    def move_primitive_in_view(self, name, idx, x, y, z, update): return self._optix.move_primitive_in_view(name, idx, x, y, z, update)

    def rotate_geometry(self, name, x, y, z, update): return self._optix.rotate_geometry(name, x, y, z, update)

    def rotate_primitive(self, name, idx, x, y, z, update): return self._optix.rotate_primitive(name, idx, x, y, z, update)

    def rotate_geometry_in_view(self, name, x, y, z, update): return self._optix.rotate_geometry_in_view(name, x, y, z, update)

    def rotate_primitive_in_view(self, name, idx, x, y, z, update): return self._optix.rotate_primitive_in_view(name, idx, x, y, z, update)

    def rotate_geometry_about(self, name, x, y, z, cx, cy, cz, update):
        return self._optix.rotate_geometry_about(name, x, y, z, cx, cy, cz, update)

    def rotate_primitive_about(self, name, idx, x, y, z, cx, cy, cz, update):
        return self._optix.rotate_primitive_about(name, idx, x, y, z, cx, cy, cz, update)

    def scale_geometry(self, name, s, update): return self._optix.scale_geometry(name, s, update)

    def scale_primitive(self, name, idx, s, update): return self._optix.scale_primitive(name, idx, s, update)

    def scale_geometry_c(self, name, s, cx, cy, cz, update): return self._optix.scale_geometry_c(name, s, cx, cy, cz, update)

    def scale_primitive_c(self, name, idx, s, cx, cy, cz, update): return self._optix.scale_primitive_c(name, idx, s, cx, cy, cz, update)

    def scale_geometry_xyz(self, name, x, y, z, update): return self._optix.scale_geometry_xyz(name, x, y, z, update)

    def scale_primitive_xyz(self, name, idx, x, y, z, update): return self._optix.scale_primitive_xyz(name, idx, x, y, z, update)

    def scale_geometry_xyz_c(self, name, x, y, z, cx, cy, cz, update): return self._optix.scale_geometry_xyz_c(name, x, y, z, cx, cy, cz, update)

    def scale_primitive_xyz_c(self, name, idx, x, y, z, cx, cy, cz, update): return self._optix.scale_primitive_xyz_c(name, idx, x, y, z, cx, cy, cz, update)

    def update_geom_buffers(self, name, mask, forced): return self._optix.update_geom_buffers(name, mask, forced)

    def set_coordinates_geom(self, mode, thickness): return self._optix.set_coordinates_geom(mode, thickness)

    def setup_camera(self, name, camera_type, eye, target, up, aperture_r, aperture_fract, focal_scale, chroma_l, chroma_t, fov, blur, glock, textures, make_current):
        return self._optix.setup_camera_ptr(name, camera_type,
                                            IntPtr.__overloads__[Int64](eye),
                                            IntPtr.__overloads__[Int64](target),
                                            IntPtr.__overloads__[Int64](up),
                                            aperture_r, aperture_fract, focal_scale, chroma_l, chroma_t, fov, blur, glock, textures, make_current)

    def update_camera(self, name, eye, target, up, aperture_r, focal_scale, fov):
        return self._optix.update_camera_ptr(name,
                                             IntPtr.__overloads__[Int64](eye),
                                             IntPtr.__overloads__[Int64](target),
                                             IntPtr.__overloads__[Int64](up),
                                             aperture_r, focal_scale, fov)

    def fit_camera(self, handle, geo_name, scale): return self._optix.fit_camera(handle, geo_name, scale)

    def get_current_camera(self): return self._optix.get_current_camera()

    def set_current_camera(self, name): return self._optix.set_current_camera(name)

    def move_camera_by(self, dx, dy, dz): return self._optix.move_camera_by(dx, dy, dz)

    def move_camera_by_local(self, dx, dy, dz): return self._optix.move_camera_by_local(dx, dy, dz)

    def rotate_camera_by(self, rx, ry, rz, cx, cy, cz): return self._optix.rotate_camera_by(rx, ry, rz, cx, cy, cz)

    def rotate_camera_by_local(self, rx, ry, rz, cx, cy, cz): return self._optix.rotate_camera_by_local(rx, ry, rz, cx, cy, cz)

    def rotate_camera_eye(self, from_x, from_y, to_x, to_y): return self._optix.rotate_camera_eye(from_x, from_y, to_x, to_y)

    def rotate_camera_eye_by(self, rx, ry, rz): return self._optix.rotate_camera_eye_by(rx, ry, rz)

    def rotate_camera_eye_by_local(self, rx, ry, rz): return self._optix.rotate_camera_eye_by_local(rx, ry, rz)

    def rotate_camera_tgt(self, from_x, from_y, to_x, to_y): return self._optix.rotate_camera_tgt(from_x, from_y, to_x, to_y)

    def rotate_camera_tgt_by(self, rx, ry, rz): return self._optix.rotate_camera_tgt_by(rx, ry, rz)

    def rotate_camera_tgt_by_local(self, rx, ry, rz): return self._optix.rotate_camera_tgt_by_local(rx, ry, rz)

    def get_camera_focal_scale(self, handle): return self._optix.get_camera_focal_scale(handle)

    def set_camera_focal_scale(self, dist): return self._optix.set_camera_focal_scale(dist)

    def set_camera_focal_length(self, dist): return self._optix.set_camera_focal_length(dist)

    def get_camera_fov(self, handle): return self._optix.get_camera_fov(handle)

    def set_camera_fov(self, fov): return self._optix.set_camera_fov(fov)

    def get_camera_aperture(self, handle): return self._optix.get_camera_aperture(handle)

    def set_camera_aperture(self, radius): return self._optix.set_camera_aperture(radius)

    def get_camera_eye(self, handle, eye):
        return self._optix.get_camera_eye_ptr(handle, IntPtr.__overloads__[Int64](eye))

    def set_camera_eye(self, eye):
        return self._optix.set_camera_eye_ptr(IntPtr.__overloads__[Int64](eye))

    def get_camera_target(self, handle, target):
        return self._optix.get_camera_target_ptr(handle, IntPtr.__overloads__[Int64](target))

    def set_camera_target(self, target):
        return self._optix.set_camera_target_ptr(IntPtr.__overloads__[Int64](target))

    def get_camera_glock(self, handle): return self._optix.get_camera_glock(handle)

    def set_camera_glock(self, state): return self._optix.set_camera_glock(state)


    def get_camera(self, handle): return self._optix.get_camera(handle)

    def get_light_shading(self): return self._optix.get_light_shading()

    def set_light_shading(self, mode): return self._optix.set_light_shading(mode)

    def get_light_handle(self, handle, primitive): return self._optix.get_light_handle(handle, primitive)

    def get_light_pos(self, name, pos):
        return self._optix.get_light_pos_ptr(name, IntPtr.__overloads__[Int64](pos))

    def get_light_color(self, name, color):
        return self._optix.get_light_color_ptr(name, IntPtr.__overloads__[Int64](color))

    def get_light_u(self, name, u):
        return self._optix.get_light_u_ptr(name, IntPtr.__overloads__[Int64](u))

    def get_light_v(self, name, v):
        return self._optix.get_light_v_ptr(name, IntPtr.__overloads__[Int64](v))

    def get_light_r(self, name): return self._optix.get_light_r(name)

    def get_light(self, name): return self._optix.get_light(name)

    def setup_spherical_light(self, name, pos, color, r, in_geometry):
        return self._optix.setup_spherical_light_ptr(name,
                                                     IntPtr.__overloads__[Int64](pos),
                                                     IntPtr.__overloads__[Int64](color),
                                                     r, in_geometry)

    def setup_parallelogram_light(self, name, pos, color, u, v, in_geometry):
        return self._optix.setup_parallelogram_light_ptr(name,
                                                         IntPtr.__overloads__[Int64](pos),
                                                         IntPtr.__overloads__[Int64](color),
                                                         IntPtr.__overloads__[Int64](u),
                                                         IntPtr.__overloads__[Int64](v),
                                                         in_geometry)

    def update_light(self, name, pos, color, r, u, v):
        return self._optix.update_light_ptr(name,
                                            IntPtr.__overloads__[Int64](pos),
                                            IntPtr.__overloads__[Int64](color),
                                            r,
                                            IntPtr.__overloads__[Int64](u),
                                            IntPtr.__overloads__[Int64](v))

    def fit_light(self, name, cam_handle, horizontal_rot, vertical_rot, dist_scale):
        return self._optix.fit_light(name, cam_handle, horizontal_rot, vertical_rot, dist_scale)

    def dolly_light(self, name, x): return self._optix.dolly_light(name, x)

    def move_light_in_view(self, name, x, y, z): return self._optix.move_light_in_view(name, x, y, z)

    def rotate_light_in_view(self, name, x, y, z): return self._optix.rotate_light_in_view(name, x, y, z)

    def scale_light(self, name, s): return self._optix.scale_light(name, s)

    def get_object_at(self, x, y, h_ref, idx_ref, face_ref):
        return self._optix.get_object_at_ptr(x, y,
                                             IntPtr.__overloads__[Int64](cast(h_ref, c_void_p).value),
                                             IntPtr.__overloads__[Int64](cast(idx_ref, c_void_p).value),
                                             IntPtr.__overloads__[Int64](cast(face_ref, c_void_p).value))

    def get_hit_at(self, x, y, px_ref, py_ref, pz_ref, d_ref):
        return self._optix.get_hit_at_ptr(x, y,
                                          IntPtr.__overloads__[Int64](cast(px_ref, c_void_p).value),
                                          IntPtr.__overloads__[Int64](cast(py_ref, c_void_p).value),
                                          IntPtr.__overloads__[Int64](cast(pz_ref, c_void_p).value),
                                          IntPtr.__overloads__[Int64](cast(d_ref, c_void_p).value))

    def register_launch_finished_callback(self, ptr):
        return self._optix.register_launch_finished_callback_ptr(IntPtr.__overloads__[Int64](cast(ptr, c_void_p).value))

    def register_accum_done_callback(self, ptr):
        return self._optix.register_accum_done_callback_ptr(IntPtr.__overloads__[Int64](cast(ptr, c_void_p).value))

    def register_scene_rt_starting_callback(self, ptr):
        return self._optix.register_scene_rt_starting_callback_ptr(IntPtr.__overloads__[Int64](cast(ptr, c_void_p).value))

    def register_start_scene_compute_callback(self, ptr):
        return self._optix.register_start_scene_compute_callback_ptr(IntPtr.__overloads__[Int64](cast(ptr, c_void_p).value))

    def register_scene_rt_completed_callback(self, ptr):
        return self._optix.register_scene_rt_completed_callback_ptr(IntPtr.__overloads__[Int64](cast(ptr, c_void_p).value))

    def get_compute_timeout(self): return self._optix.get_compute_timeout()

    def set_compute_timeout(self, n): return self._optix.set_compute_timeout(n)

    def get_rt_timeout(self): return self._optix.get_rt_timeout()

    def set_rt_timeout(self, n): return self._optix.set_rt_timeout(n)

    def get_min_accumulation_step(self): return self._optix.get_min_accumulation_step()

    def set_min_accumulation_step(self, n): return self._optix.set_min_accumulation_step(n)

    def get_max_accumulation_frames(self): return self._optix.get_max_accumulation_frames()

    def set_max_accumulation_frames(self, n): return self._optix.set_max_accumulation_frames(n)

    def encoder_create(self, fps, bit_rate, idr_rate, profile, preset):
        if self._encoder_available:
            return self._optix.encoder_create(fps, bit_rate, idr_rate, profile, preset)
        else: return False

    def encoder_start(self, output_name, n_frames): return self._optix.encoder_start(output_name, n_frames)

    def encoder_stop(self): return self._optix.encoder_stop()

    def encoder_is_open(self): return self._optix.encoder_is_open()

    def encoded_frames(self): return self._optix.encoded_frames()

    def encoding_frames(self): return self._optix.encoding_frames()

    def open_simplex_2d(self, noise_ptr, inputs_ptr, length):
        return self._optix.open_simplex_2d_ptr(IntPtr.__overloads__[Int64](noise_ptr), IntPtr.__overloads__[Int64](inputs_ptr), length)

    def open_simplex_3d(self, noise_ptr, inputs_ptr, length):
        return self._optix.open_simplex_3d_ptr(IntPtr.__overloads__[Int64](noise_ptr), IntPtr.__overloads__[Int64](inputs_ptr), length)

    def open_simplex_4d(self, noise_ptr, inputs_ptr, length):
        return self._optix.open_simplex_4d_ptr(IntPtr.__overloads__[Int64](noise_ptr), IntPtr.__overloads__[Int64](inputs_ptr), length)

    def get_image_meta(self, name, width_ref, height_ref, spp_ref, bps_ref):
        return self._optix.get_image_meta_ptr(name,
                                      IntPtr.__overloads__[Int64](cast(width_ref, c_void_p).value),
                                      IntPtr.__overloads__[Int64](cast(height_ref, c_void_p).value),
                                      IntPtr.__overloads__[Int64](cast(spp_ref, c_void_p).value),
                                      IntPtr.__overloads__[Int64](cast(bps_ref, c_void_p).value))

    def read_image(self, name, data_ptr, width, height, spp, bps):
        return self._optix.read_image_ptr(name,
                                      IntPtr.__overloads__[Int64](cast(data_ptr, c_void_p).value),
                                      width, height, spp, bps)

    def read_image_normalized(self, name, data_ptr, width, height, spp):
        return self._optix.read_image_normalized_ptr(name,
                                      IntPtr.__overloads__[Int64](cast(data_ptr, c_void_p).value),
                                      width, height, spp)


    def get_gpu_architecture(self): return self._optix.get_gpu_architecture()
    def set_gpu_architecture(self, arch): self._optix.set_gpu_architecture(arch)

    def get_n_gpu_architecture(self, ordinal): return self._optix.get_n_gpu_architecture(ordinal)

    def set_library_dir(self, path): self._optix.set_library_dir(path)

    def set_include_dir(self, path): self._optix.set_include_dir(path)

    def get_display_scaling(self): return self._optix.get_display_scaling()

    def test_library(self, x): return self._optix.test_library(x)


def load_optix():
    """Load RnD.SharpOptiX library, setup CUDA lib and include folders.
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

