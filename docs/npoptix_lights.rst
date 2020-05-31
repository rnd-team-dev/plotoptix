Lights
======

Light shading mode
------------------

Two shading modes are available: :attr:`plotoptix.enums.LightShading.Hard` for
best caustics or :attr:`plotoptix.enums.LightShading.Soft` for fast convergence.
Use :meth:`plotoptix.NpOptiX.get_param` to setup the mode. Note, it shuld be set
before lights are added to the scene.

Deprecated methods to configure shading (we'll remove them at some point):

.. automethod:: plotoptix.NpOptiX.set_light_shading
.. automethod:: plotoptix.NpOptiX.get_light_shading

Setup and update lighting
-------------------------

.. automethod:: plotoptix.NpOptiX.get_light_names
.. automethod:: plotoptix.NpOptiX.setup_light
.. automethod:: plotoptix.NpOptiX.setup_spherical_light
.. automethod:: plotoptix.NpOptiX.setup_parallelogram_light
.. automethod:: plotoptix.NpOptiX.setup_area_light
.. automethod:: plotoptix.NpOptiX.update_light
.. automethod:: plotoptix.NpOptiX.light_fit

Read back light properties
--------------------------

.. automethod:: plotoptix.NpOptiX.get_light
.. automethod:: plotoptix.NpOptiX.get_light_color
.. automethod:: plotoptix.NpOptiX.get_light_pos
.. automethod:: plotoptix.NpOptiX.get_light_r
.. automethod:: plotoptix.NpOptiX.get_light_u
.. automethod:: plotoptix.NpOptiX.get_light_v

.. toctree::
   :caption: API Reference
   :maxdepth: 2
