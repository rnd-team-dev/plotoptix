.. _callbacks:

Callbacks
=========

Callback methods
----------------

Methods launching callables provided to the NpOptiX constructor and
corresponding setters to configure callbacks after initialization.

.. automethod:: plotoptix.NpOptiX.set_launch_finished_cb
.. automethod:: plotoptix.NpOptiX._launch_finished_callback
.. automethod:: plotoptix.NpOptiX._scene_rt_starting_callback
.. automethod:: plotoptix.NpOptiX.set_accum_done_cb
.. automethod:: plotoptix.NpOptiX._accum_done_callback
.. automethod:: plotoptix.NpOptiX.set_scene_compute_cb
.. automethod:: plotoptix.NpOptiX._start_scene_compute_callback
.. automethod:: plotoptix.NpOptiX.set_rt_completed_cb
.. automethod:: plotoptix.NpOptiX._scene_rt_completed_callback

Pause and resume compute
------------------------

Scene computation callbacks can be paused/resumed, without stopping
the raytracing loop, using following two methods:

.. automethod:: plotoptix.NpOptiX.pause_compute
.. automethod:: plotoptix.NpOptiX.resume_compute

.. toctree::
   :caption: API Reference
   :maxdepth: 2
