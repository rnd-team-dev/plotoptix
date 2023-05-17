"""
Pytorch texture source - game of life.
"""

import torch
import torch.nn.functional as F

from plotoptix import TkOptiX
from plotoptix.materials import m_flat


class params():
    if torch.cuda.is_available:
        device = torch.device('cuda')
        dtype = torch.float16
    else:
        device = torch.device('cpu')
        dtype = torch.float32
    w = torch.tensor(
        [[[[1.0,1.0,1.0], [1.0,0.0,1.0], [1.0,1.0,1.0]]]],
        dtype=dtype, device=device, requires_grad=False
    )
    cells = torch.rand((1,1,128,128), dtype=dtype, device=device, requires_grad=False)
    cells[cells > 0.995] = 1.0
    cells[cells < 1.0] = 0.0
    tex2D = torch.unsqueeze(cells[0, 0].type(torch.float32), -1).expand(-1, -1, 4).contiguous()


# Update texture data with a simple "game of life" rules.
def compute(rt, delta):
    params.cells = F.conv2d(params.cells, weight=params.w, stride=1, padding=1)
    params.cells[params.cells < 2.0] = 0.0
    params.cells[params.cells > 3.0] = 0.0
    params.cells[params.cells != 0.0] = 1.0

    # Conversion to float32 and to contiguous memoty layout is ensured by plotoptix,
    # though you may wamt to make it explicit like here, eg for performance reasons.
    params.tex2D = torch.unsqueeze(params.cells[0, 0].type(torch.float32), -1).expand(-1, -1, 4).contiguous()


# Copy texture data to plotoptix scene.
def update_data(rt):
    rt.set_torch_texture_2d("tex2d", params.tex2D, refresh=True)


def main():
    rt = TkOptiX(
        on_scene_compute=compute,
        on_rt_completed=update_data
    )
    rt.set_param(min_accumulation_step=1)
    rt.set_background(0)
    rt.set_ambient(0)

    # NOTE: pytorch features are not enabled by default. You need
    # to call this method before using anything related to pytorch.
    rt.enable_torch()

    rt.set_torch_texture_2d("tex2d", params.tex2D, addr_mode="Clamp", filter_mode="Nearest")
    m_flat["ColorTextures"] = [ "tex2d" ]
    rt.setup_material("flat", m_flat)

    rt.setup_camera("cam1", eye=[0, 0, 3], target=[0, 0, 0], fov=35, glock=True)

    rt.set_data("plane", geom="Parallelograms", mat="flat",
                pos=[-1, -1, 0], u=[2, 0, 0], v=[0, 2, 0], c=0.9
    )

    rt.start()

if __name__ == '__main__':
    main()

