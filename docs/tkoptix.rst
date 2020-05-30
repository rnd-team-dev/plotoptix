TkOptiX - Tkinter based GUI
===========================

Mouse and keys actions
----------------------

You can manipulate camera, objects and lights from Tkinter GUI:

- camera is selected by default, double-click an object/light to select it, double click again to select a primitive within the object, double-click empty area to or double-right-click select camera

With camera selected:

- rotate eye around the target: hold and drag left mouse button
- rotate target around the eye (pan/tilt): hold and drag right mouse button
- zoom out/in (change field of view): hold shift + left mouse button and drag up/down
- move eye backward/forward (dolly): hold shift + right mouse button and drag up/down
- change focus distance in *depth of field* cameras: hold ctrl + left mouse button and drag up/down
- change aperture radius in *depth of field* cameras: hold ctrl + right mouse button and drag up/down
- focus at an object: hold ctrl + double-click left mouse button
- select an object: double-click left mouse button

With a light or an object / primitive selected:

- rotate around camera XY (right, up) coordinates: hold and drag left mouse button
- rotate around camera XZ (right, forward) coordinates: hold ctrl and drag left mouse button
- move in camera XY (right, up) coordinates: hold shift and drag left mouse button
- move in camera XZ (right, forward) coordinates: hold and drag right mouse button
- move in the normal direction (parallelogram light only): shift + right mouse button and drag up/down
- scale up/down: hold ctrl + shift + left mouse button and drag up/down
- select camera: double-click left mouse button in empty area or double-right-click anywhere

API reference
-------------

.. automethod:: plotoptix.TkOptiX.get_rt_size
.. automethod:: plotoptix.TkOptiX.set_rt_size
.. automethod:: plotoptix.TkOptiX.select
.. automethod:: plotoptix.TkOptiX.show
.. automethod:: plotoptix.TkOptiX.close

.. toctree::
   :caption: API Reference
   :maxdepth: 2