## PlotOptiX examples

Note, some fresh examples may use features available in the repository head, but not yet in the release submitted to PyPI. Examples compatible with the PyPI release can be downloaded with::

	python -m plotoptix.install examples

### 1. Basics

Collection of short and simple scripts to get started.

Many scripts are using Tkinter GUI, so here is a summary of mouse and keys actions:

- camera is selected by default, double-click an object/light to select it, double click again to select a primitive within the object, double-click in empty area or double-right-click to select camera
- use ``rt.select(name)`` method if an object or light is not in the view or it is hard to be selected
- select parent mesh to apply transformations to all children as well

With camera selected:

- rotate eye around the target: hold and drag left mouse button
- rotate target around the eye (pan/tilt): hold and drag right mouse button
- zoom out/in (change field of view): hold shift + left mouse button and drag up/down
- move eye backward/forward (dolly): hold shift + right mouse button and drag up/down
- change focus distance in "depth of field" cameras: hold ctrl + left mouse button and drag up/down
- change aperture radius in "depth of field" cameras: hold ctrl + right mouse button and drag up/down
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


You'll find here super-basic examples of displaying data, like scatter plots or line plots below:

![screenshots1](https://plotoptix.rnd.team/images/basic_scripts_screens.jpg "PlotOptiX output screenshots")

...and a bit more on tuning the available options, like the material and light shading modes:

![screenshots2](https://plotoptix.rnd.team/images/light_shading_modes.jpg "PlotOptiX light shading")

...or usage of material properties such as refraction index and textures:

![screenshots3](https://plotoptix.rnd.team/images/refractions_dispersion_textures.jpg "PlotOptiX light dispersion and textures")

...or 2D postprocessing algorithms:

![screenshots4](https://plotoptix.rnd.team/images/postprocessing.jpg "PlotOptiX 2D postprocessing")

...or normal shading with displacement maps:

![screenshots5](https://plotoptix.rnd.team/images/normal_shading_with_textures.jpg "PlotOptiX 2D postprocessing")

...or subsurface scattering parameters:

![screenshots5](https://plotoptix.rnd.team/images/subsurface.jpg "PlotOptiX scattering in volumes")

### 2. Animations and callbacks

Callbacks in PlotOptiX are widely available throughout the raytracing process. You can provide functions to execute on each frame raytracing start, completion, etc., allowing for progressive image updates, saving output to file or making animated plots. Callbacks designed for heavy compute are executed in parallel to the raytracing, and those intended for accessing image data are synchronized with GPU transfers. That is a really powerfull pattern!

GitHub can render notebooks content, so it is best to look at descriptions inlined in the code there.

![screenshots5](https://plotoptix.rnd.team/images/notebook_screens.jpg "PlotOptiX in notebook screenshots")
![screenshots6](https://plotoptix.rnd.team/images/notebook_screens_2.jpg "PlotOptiX in notebook screenshots")

### 3. Projects

Collection of tutorials and artistic projects. Expect this section to grow slowly. Projects are large and need lots of work to prepare. We started with the *"Making of the Moon"*, a really big one!

![moon ray_traced](https://plotoptix.rnd.team/images/moon_2res_banner1.jpg "The Moon ray-traced with PlotOptiX")

Now we're collecting scripts for the next project, *"Simplex noise trajectories"*:

![noise trajectories](https://plotoptix.rnd.team/images/opensimplex_banner.jpg "Noise compositions ray-traced with PlotOptiX")
