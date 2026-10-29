"""Inference-only PyTorch equivalent of the official adaptive trilinear lookup."""
import torch
import torch.nn.functional as F


def ailut_transform(image, lut, vertices):
    # Find each RGB value between its two neighbouring adaptive LUT vertices.
    coordinates = []
    for channel in range(3):
        values = image[:, channel].flatten(1).contiguous()
        axis = vertices[:, channel].contiguous()
        lower = (torch.searchsorted(axis, values) - 1).clamp(0, axis.shape[1] - 2)
        left = axis.gather(1, lower)
        right = axis.gather(1, lower + 1)
        fraction = (values - left) / (right - left).clamp_min(1e-8)
        coordinates.append((lower + fraction) / (axis.shape[1] - 1) * 2 - 1)
    # Official LUT storage is [output channel, blue, green, red].
    grid = torch.stack(coordinates, -1).view(image.shape[0], 1, *image.shape[2:], 3)
    return F.grid_sample(lut, grid, mode='bilinear', padding_mode='border',
                         align_corners=True).squeeze(2)
