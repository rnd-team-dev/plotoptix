Making of the Moon
==================

![moon ray_traced](https://plotoptix.rnd.team/images/moon_2res_banner.jpg "The Moon ray-traced with PlotOptiX")

Data-viz project for ray-tracing the Moon model with the highest possible details. Data is downloaded from NASA resources. Read the [tutorial on Medium](https://medium.com/@sulej.robert/the-moon-made-twice-at-home-a2cb73b3f1e8), and have a look at [images on Behance](https://www.behance.net/gallery/84326717/Making-of-the-Moon).

There are few options to run this project:

- *mesh* notebooks are using triangular mesh - fast, but eats up lots of memory,
- *displacement* notebooks are displacing a sphere surface - slower, but can make much more detailed Moon,
- *remote.ipynb* notebooks can run on remote servers, raytracing output is inlined between cells there.

Read [setup instructions](https://github.com/rnd-team-dev/plotoptix/blob/master/examples/3_projects/moon/setup_gcp_for_python_notebooks.txt) to configure the Google Cloud Platform Compute Engine instance.
