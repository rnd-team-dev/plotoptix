## PlotOptiX examples

Note, some fresh examples may use features available in the repository head, but not yet in the release submitted to PyPI. Examples compatible with the PyPI release can be downloaded with::

	python -m plotoptix.install examples

### 1. Basic plot making

Collection of short and simple scripts to get started.

Many scripts are using Tkinter UI, so here is a summary of mouse and keys actions:
- rotate camera eye around the target: hold and drag left mouse button
- rotate camera target around the eye: hold and drag right mouse button
- zoom out/in (change camera field of view): hold shift + left mouse button and drag up/down
- move camera eye backward/forward: hold shift + right mouse button and drag up/down
- change focus distance in "depth of field" cameras: hold ctrl + left mouse button and drag up/down
- change aperture radius in "depth of field" cameras: hold ctrl + right mouse button and drag up/down
- focus at an object: hold ctrl + double-click left mouse button
- select an object: double-click left mouse button (info on terminal output)

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

### 2. Animations and callbacks

Callbacks in PlotOptiX are widely available throughout the raytracing process. You can provide functions to execute on each frame raytracing start, completion, etc., allowing for progressive image updates, saving output to file or making animated plots. Callbacks designed for heavy compute are executed in parallel to the raytracing, and those intended for accessing image data are synchronized with GPU transfers. That is a really powerfull pattern!

GitHub can render notebooks content, so it is best to look at descriptions inlined in the code there.

![screenshots5](https://plotoptix.rnd.team/images/notebook_screens.jpg "PlotOptiX in notebook screenshots")
![screenshots6](https://plotoptix.rnd.team/images/notebook_screens_2.jpg "PlotOptiX in notebook screenshots")

### 3. Projects

Collection of tutorials and artistic projects. Expect this section to grow slowly. Projects are large and need lots of work to prepare. Let's start with the *"Making of the Moon"*, code and graphics are ready, now working on the tutorial text.

![moon ray_traced](https://plotoptix.rnd.team/images/moon_2res_banner.jpg "The Moon ray-traced with PlotOptiX")
