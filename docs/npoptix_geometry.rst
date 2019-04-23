Plot geometry
=============

Create and update plot
----------------------

.. automethod:: plotoptix.NpOptiX.set_data
.. automethod:: plotoptix.NpOptiX.update_data

Direct modifications of data
----------------------------

These methods allow making changes to properties of data points
stored internally in the raytracer, without re-sending whole data
arrays from Python code.

.. automethod:: plotoptix.NpOptiX.move_geometry
.. automethod:: plotoptix.NpOptiX.move_primitive
.. automethod:: plotoptix.NpOptiX.rotate_geometry
.. automethod:: plotoptix.NpOptiX.rotate_primitive
.. automethod:: plotoptix.NpOptiX.scale_geometry
.. automethod:: plotoptix.NpOptiX.scale_primitive
.. automethod:: plotoptix.NpOptiX.update_geom_buffers

Coordinate system
-----------------

**Note:** coordinate system layouts are now under development,
only a simple box containing all data is available now.

.. automethod:: plotoptix.NpOptiX.set_coordinates

.. toctree::
   :caption: API Reference
   :maxdepth: 2
