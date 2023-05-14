Geometry
========

Create, load, update, and remove a plot
---------------------------------------

.. automethod:: plotoptix.NpOptiX.get_geometry_names
.. automethod:: plotoptix.NpOptiX.set_data
.. automethod:: plotoptix.NpOptiX.update_data
.. automethod:: plotoptix.NpOptiX.set_data_2d
.. automethod:: plotoptix.NpOptiX.update_data_2d
.. automethod:: plotoptix.NpOptiX.set_surface
.. automethod:: plotoptix.NpOptiX.update_surface
.. automethod:: plotoptix.NpOptiX.set_mesh
.. automethod:: plotoptix.NpOptiX.update_mesh
.. automethod:: plotoptix.NpOptiX.load_mesh_obj
.. automethod:: plotoptix.NpOptiX.load_merged_mesh_obj
.. automethod:: plotoptix.NpOptiX.set_displacement
.. automethod:: plotoptix.NpOptiX.load_displacement
.. automethod:: plotoptix.NpOptiX.delete_geometry

Direct modifications of data
----------------------------

These methods allow making changes to properties of data points
stored internally in the raytracer, without re-sending whole data
arrays from the Python code.

:class:`plotoptix.geometry.PinnedBuffer` functionality should be
considered *experimental*, although it seems to be extremely useful
in simulation use cases.

.. autoclass:: plotoptix.geometry.PinnedBuffer

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
