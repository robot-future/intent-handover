# Third-party attribution

`src/intent_handover/_vendor/text2hoi/` contains `texthom.py`, `pointnet.py`,
`cvae.py`, and `diffusion.py` from [JunukCha/Text2HOI](https://github.com/JunukCha/Text2HOI),
local upstream commit `111f5301692374e4be18084c8284d6a2e4336a34`.
Copyright (c) 2024 Junuk Cha; MIT license copied alongside the code.

Changes: the CVAE PointNet import is package-relative; PointNet's identity tensor
follows the input CUDA device instead of forcing the default CUDA device.
Line endings and trailing whitespace are normalized. The architecture and checkpoint parameter names are retained. New adapter,
selection, proxy geometry and demo code are separately authored for this release.

The avoidance cost was extracted from DUM-E's modified
`Text2HOI/lib/utils/demo_utils.py::find_angle`. Usage filtering, clean interfaces,
input validation, standalone reports and release tests are new implementations.

The procedural example assets are newly generated boxes, not OakInk/ShapeNet,
MANO or third-party grasp annotations. Pretrained Text2HOI and CLIP models are
obtained separately from their original providers. Their code licenses do not
automatically determine the terms for those model/data files.
