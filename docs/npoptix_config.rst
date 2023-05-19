Start, configure, get output
============================

.. automethod:: plotoptix.NpOptiX.start
.. automethod:: plotoptix.NpOptiX.close
.. automethod:: plotoptix.NpOptiX.get_scene
.. automethod:: plotoptix.NpOptiX.set_scene
.. automethod:: plotoptix.NpOptiX.load_scene
.. automethod:: plotoptix.NpOptiX.save_scene
.. automethod:: plotoptix.NpOptiX.save_image
.. automethod:: plotoptix.NpOptiX.get_rt_output
.. autoattribute:: plotoptix.NpOptiX._img_rgba
.. autoattribute:: plotoptix.NpOptiX._raw_rgba
.. autoattribute:: plotoptix.NpOptiX._hit_pos
.. autoattribute:: plotoptix.NpOptiX._geo_id
.. autoattribute:: plotoptix.NpOptiX._albedo
.. autoattribute:: plotoptix.NpOptiX._normal
.. automethod:: plotoptix.NpOptiX._run_event_loop
.. automethod:: plotoptix.NpOptiX.run
.. automethod:: plotoptix.NpOptiX.get_gpu_architecture
.. automethod:: plotoptix.NpOptiX.enable_cupy
.. automethod:: plotoptix.NpOptiX.enable_torch

Scene configuration
-------------------

.. automethod:: plotoptix.NpOptiX.set_ambient
.. automethod:: plotoptix.NpOptiX.get_ambient
.. automethod:: plotoptix.NpOptiX.set_background
.. automethod:: plotoptix.NpOptiX.get_background
.. automethod:: plotoptix.NpOptiX.get_background_mode
.. automethod:: plotoptix.NpOptiX.set_background_mode
.. automethod:: plotoptix.NpOptiX.refresh_scene
.. automethod:: plotoptix.NpOptiX.resize

Raytracer configuration
-----------------------

Ray tracing algorithms are controlled via variables allocated in
the host memory and on the GPU. Host variables can be modified with
:meth:`plotoptix.NpOptiX.set_param` method and are described there.
Values of host variables are available through the :meth:`plotoptix.NpOptiX.get_param`
method. Most of GPU variables controll postprocessing algorithms (see 
description in :py:mod:`plotoptix.enums.Postprocessing`). These variables
are accessed via :meth:`plotoptix.NpOptiX.set_int` / :meth:`plotoptix.NpOptiX.get_int`
and similar methods.

GPU variables related to the raytracer general configuraton are documented below:

- **Number of the ray segments**
  
  Name: *path_seg_range*

  Type: ``uint2``

  Default value: ``[2, 6]``

  Description: ``[min, max]`` range of the ray segments; if the ray is scattered, reflected
  or refracted more than ``min`` times it may be terminated with the Russian Roulette algorithm
  but the number of segments never exceeds ``max``. Use higher values in scenes with multiple
  transparent and/or reflective objects and if you need the high quality of diffuse lights.
  Use lower values to increase performance.

  Example:

  .. code-block:: python

     rt = TkOptiX()
     rt.set_uint("path_seg_range", 4, 16)

- **Ray tracing precision**

  Name: *scene_epsilon*

  Type: ``float``

  Default value: ``0.002``

  Description: epsilon value used whenever a geometrical computation threshold is needed, e.g.
  as a minimum distance between hits or displacement of the ray next segment in the normal
  direction. You may need to set a lower value if your scene dimensions are tiny, or increase
  the value to avoid artifacts (e.g. in scenes with huge amounts of very small primitives).

  Example:

  .. code-block:: python

     rt = TkOptiX()
     rt.set_float("scene_epsilon", 1.0e-04)

- **Denoiser start frame**

  Name: *denoiser_start*

  Type: ``uint``

  Default value: ``4``

  AI denoiser is applied to output image after accumulating ``denoiser_start`` frames. Use default
  value for interactive work. Use higher values for final rendering, when noisy intermediate results
  are acceptable. In such cases the optimal configuration is to set ``denoiser_start`` value equal
  to ``max_accumulation_frames`` (see :meth:`plotoptix.NpOptiX.set_param`), then denoiser is applied
  only once, at the end of ray tracing.

  Example:

  .. code-block:: python

     rt = TkOptiX()
     rt.set_param(min_accumulation_step=8,     # update image every 8 frames
                  max_accumulation_frames=128, # accumulate 128 frames in total
                 )
     rt.set_uint("denoiser_start", 128)        # denoise when the accumulation is finished

     rt.set_float("tonemap_exposure", 0.9)
     rt.set_float("tonemap_gamma", 2.2)
     rt.add_postproc("Denoiser")               # setup denoiser postprocessing


.. automethod:: plotoptix.NpOptiX.set_param
.. automethod:: plotoptix.NpOptiX.get_param
.. automethod:: plotoptix.NpOptiX.set_int
.. automethod:: plotoptix.NpOptiX.get_int
.. automethod:: plotoptix.NpOptiX.set_uint
.. automethod:: plotoptix.NpOptiX.get_uint
.. automethod:: plotoptix.NpOptiX.get_uint2
.. automethod:: plotoptix.NpOptiX.set_float
.. automethod:: plotoptix.NpOptiX.get_float
.. automethod:: plotoptix.NpOptiX.get_float2
.. automethod:: plotoptix.NpOptiX.get_float3
.. automethod:: plotoptix.NpOptiX.set_texture_1d
.. automethod:: plotoptix.NpOptiX.set_texture_2d
.. automethod:: plotoptix.NpOptiX.load_texture

Postprocessing 2D
-----------------
.. automethod:: plotoptix.NpOptiX.add_postproc
.. automethod:: plotoptix.NpOptiX.set_correction_curve

Encoder configuration
---------------------
.. automethod:: plotoptix.NpOptiX.encoder_create
.. automethod:: plotoptix.NpOptiX.encoder_start
.. automethod:: plotoptix.NpOptiX.encoder_stop
.. automethod:: plotoptix.NpOptiX.encoder_is_open
.. automethod:: plotoptix.NpOptiX.encoded_frames
.. automethod:: plotoptix.NpOptiX.encoding_frames

.. toctree::
   :caption: API Reference
   :maxdepth: 2
