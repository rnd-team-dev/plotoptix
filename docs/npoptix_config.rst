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
.. automethod:: plotoptix.NpOptiX._run_event_loop
.. automethod:: plotoptix.NpOptiX.run
.. automethod:: plotoptix.NpOptiX.get_gpu_architecture

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
