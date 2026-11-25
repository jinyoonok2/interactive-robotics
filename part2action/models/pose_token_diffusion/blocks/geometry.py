"""Differentiable pose geometry and the actual PartGym agent-view projection."""
import math
import torch
from torch.nn import functional as F


def quaternion_matrix(quaternion):
    q = quaternion.float()
    if not torch.isfinite(q).all() or (q.norm(dim=-1) < 1e-6).any():
        raise ValueError('Measured TCP needs finite, nonzero xyzw quaternions')
    x, y, z, w = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1).reshape(*q.shape[:-1], 3, 3)


def rotation_matrix(six):
    columns = six.float().reshape(-1, 3, 2)
    first = columns[..., 0]
    fallback = torch.zeros_like(first)
    fallback[:, 0] = 1
    first = F.normalize(torch.where(first.norm(dim=-1, keepdim=True) > 1e-6, first, fallback), dim=-1)
    second = columns[..., 1] - (first*columns[..., 1]).sum(-1, keepdim=True)*first
    axis = F.one_hot(first.abs().argmin(-1), 3).to(first.dtype)
    fallback = axis-(axis*first).sum(-1, keepdim=True)*first
    second = F.normalize(torch.where(second.norm(dim=-1, keepdim=True) > 1e-6, second, fallback), dim=-1)
    return torch.stack([first, second, torch.cross(first, second, dim=-1)], -1)


def relative_pose(tcp_pose, target):
    """9D actual pose error, zeroed safely for invalid reference targets."""
    with torch.autocast(device_type=tcp_pose.device.type, enabled=False):
        current_rotation = quaternion_matrix(tcp_pose[:, :4])
        position = torch.where(target.valid[:, None], target.position_world.float(), tcp_pose[:, 4:7].float())
        rotation = torch.where(target.valid[:, None, None], target.rotation_world.float(), current_rotation)
        local_position = (current_rotation.transpose(-1, -2) @ (position-tcp_pose[:, 4:7].float())[..., None]).squeeze(-1)
        local_rotation = current_rotation.transpose(-1, -2) @ rotation
        identity = local_position.new_tensor([1., 0., 0., 1., 0., 0.])
        error = torch.cat([local_position, local_rotation[:, :, :2].reshape(-1, 6)-identity], -1)
        return torch.where(target.valid[:, None], error, torch.zeros_like(error))


def camera_grid(points, height, width, fov_degrees=60.):
    """Match bullet_sim._build_intrinsic_from_fov and Open3D camera XYZ.

    Point clouds use the intrinsic matrix built by the environment, not nominal
    config.yaml focal values. Pixel centers are mapped through image resize and
    patch sampling with align_corners=False. Current PartGym frames are square.
    """
    points = points.float()
    valid = torch.isfinite(points).all(-1) & (points[..., 2] > 1e-6)
    clean = torch.where(valid[..., None], points, torch.zeros_like(points))
    focal = width/(2*math.tan(math.radians(fov_degrees)/2))
    z = clean[..., 2].clamp(min=1e-6)
    u = clean[..., 0]/z*focal+width/2
    v = clean[..., 1]/z*focal+height/2
    valid = valid & (u >= -.5) & (u < width-.5) & (v >= -.5) & (v < height-.5)
    grid = torch.stack([2*(u+.5)/width-1, 2*(v+.5)/height-1], -1)
    return torch.where(valid[..., None], grid, torch.full_like(grid, 2.)), valid


def sample_image_features(patch_map, camera_points, image_height, image_width, fov_degrees=60.):
    grid, valid = camera_grid(camera_points, image_height, image_width, fov_degrees)
    # grid_sample is run in float32 for stable CPU/AMP support.
    with torch.autocast(device_type=patch_map.device.type, enabled=False):
        values = F.grid_sample(patch_map.float(), grid[:, :, None], align_corners=False,
                               padding_mode='zeros').squeeze(-1).transpose(1, 2)
    return values.to(patch_map.dtype)*valid[..., None], valid
