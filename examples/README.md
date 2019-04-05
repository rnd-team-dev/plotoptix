## PlotOptiX examples

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

### 2. Animations and callbacks

Callbacks in PlotOptiX are widely available throughout the raytracing process. You can provide functions to execute on each frame raytracing start, completion, etc., allowing for progressive image updates, saving output to file or making animated plots. Callbacks designed for heavy compute are executed in parallel to the raytracing, and those intended for accessing image data are synchronized with GPU transfers. That is a really powerfull pattern!

GitHub can render notebooks content, so it is best to look at descriptions inlined in the code there.

![screenshots](https://github.com/rnd-team-dev/plotoptix/blob/master/examples/notebook_screens.jpg "Notebook screenshots")

*Progressing with more examples...*
