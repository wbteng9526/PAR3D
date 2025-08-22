import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


_EPS = np.finfo(float).eps * 4.0

def generate_rays(h, w, extrinsics, intrinsics):
    pixel_center = 0.5
    image_coords = torch.meshgrid(
        torch.arange(h),
        torch.arange(w),
        indexing="ij"
    )

    image_coords = torch.stack(image_coords, dim=-1) + pixel_center
    image_coords = image_coords.to(extrinsics.device)
    y = image_coords[..., 0]
    x = image_coords[..., 1]

    cx, cy = intrinsics[0, 0, 2], intrinsics[0, 1, 2]
    fx, fy = intrinsics[0, 0, 0], intrinsics[0, 1, 1]
    pixels = torch.stack([(x - cx) / fx, (y - cy) / fy], -1)
    pixels[..., 1] *= -1
    camera_dirs = torch.stack([pixels[..., 0], pixels[..., 1], -torch.ones_like(pixels[..., 0])], dim=-1).to(extrinsics.device) # [h, w, 3]

    # pixels = torch.stack([x, -y, -torch.ones_like(x)], dim=-1).to(extrinsics.device)
    # inverse_intrinsics = intrinsics.inverse()
    # camera_dirs = (inverse_intrinsics[:, None, None, :] @ pixels[Ellipsis, None])[Ellipsis, 0]

    rotations = extrinsics[..., :3, :3] # [N, 3, 3]
    directions = torch.sum(camera_dirs[None, ..., None, :] * rotations[:, None, None, ...], dim=-1)
    directions_norm = torch.maximum(torch.linalg.vector_norm(directions, dim=-1, keepdims=True), torch.tensor([_EPS]).to(directions.device))
    rays_d = directions / directions_norm
    rays_d = rays_d.to(dtype=directions.dtype)

    # directions = (extrinsics[:, None, None, :3, :3] @ camera_dirs[None, Ellipsis, None])[Ellipsis, 0]
    # rays_d = F.normalize(directions, dim=-1)

    rays_o = extrinsics[:, :3, -1]
    rays_o = rays_o[:, None, None, ...].expand_as(rays_d)
    # inds = torch.arange(0, h*w).expand(extrinsics.shape[0], h*w).to(extrinsics.device)

    # i = inds % w + 0.5
    # j = torch.div(inds, w, rounding_mode='floor') + 0.5

    # zs = - torch.ones_like(i)
    # xs = - (i - intrinsics[0, 0, 2]) / (intrinsics[0, 0, 0]) * zs
    # ys =   (j - intrinsics[0, 1, 2]) / (intrinsics[0, 1, 1]) * zs

    # directions = torch.stack((xs, ys, zs), dim=-1)
    # rays_d = F.normalize(directions @ extrinsics[:, :3, :3].transpose(-1, -2), dim=-1)
    # rays_o = extrinsics[..., :3, 3]
    # rays_o = rays_o[..., None, :].expand_as(rays_d)


    return rays_o, rays_d

def interpolate_between_transform(c2w1, c2w2, num_views=8):
    # convert from nerfstudio to opencv
    # c2w1[2, :] *= -1
    # c2w1 = c2w1[np.array([1, 0, 2, 3]), :]
    # c2w1[0:3, 1:3] *= -1

    # c2w2[2, :] *= -1
    # c2w2 = c2w2[np.array([1, 0, 2, 3]), :]
    # c2w2[0:3, 1:3] *= -1

    o1 = c2w1[:3, -1]
    o2 = c2w2[:3, -1]

    r1 = np.linalg.inv(c2w1[:3, :3])
    r2 = np.linalg.inv(c2w2[:3, :3])

    q1 = R.from_matrix(r1).as_quat()
    q2 = R.from_matrix(r2).as_quat()

    t_value = np.linspace(0, 1, num_views)
    interp_c2ws = []
    for t in t_value:
        slerp = Slerp(np.array([0, 1]), R.from_quat([q1, q2]))
        interp_r = slerp([t])
        interp_o = (1 - t) * o1 + t * o2

        interp_c2w = np.eye(4)
        interp_c2w[:3, :3] = np.linalg.inv(interp_r.as_matrix())
        interp_c2w[:3, -1] = interp_o
        interp_c2ws.append(interp_c2w)
    
    interp_c2ws = np.stack(interp_c2ws)
    return interp_c2ws

def get_relative_extrinsic(extrinsics: torch.Tensor, first_context_view_index: int = 0):
    num_views = len(extrinsics)
    anc_cam_pos = extrinsics[first_context_view_index][:3, -1]
    anc_cam_rot = extrinsics[first_context_view_index][:3, :3]

    rel_cam_pos = extrinsics[:, :3, -1] - anc_cam_pos[None]
    rel_cam_rot = extrinsics[:, :3, :3] @ torch.linalg.inv(anc_cam_rot)[None]
    rel_extrinsics = torch.eye(4).unsqueeze(0).repeat(num_views, 1, 1).to(extrinsics.device)
    rel_extrinsics[:, :3, :3] = rel_cam_rot
    rel_extrinsics[:, :3, -1] = rel_cam_pos

    return rel_extrinsics

def orient_and_center_pose(poses):
    origins = poses[..., :3, 3]

    translation = torch.mean(origins, dim=0)
    up = torch.mean(poses[:, :3, 1], dim=0)
    up = up / torch.linalg.norm(up)

    rotation = rotation_matrix_between(up, torch.Tensor([0, 0, 1]).to(poses.device))
    transform = torch.cat([rotation, rotation @ -translation[..., None]], dim=-1)
    oriented_poses = transform @ poses

    return oriented_poses

def rotation_matrix_between(a, b):
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.linalg.cross(a, b)  # Axis of rotation.

    # Handle cases where `a` and `b` are parallel.
    eps = 1e-6
    if torch.sum(torch.abs(v)) < eps:
        x = torch.tensor([1.0, 0, 0]).to(a.device) if abs(a[0]) < eps else torch.tensor([0, 1.0, 0]).to(a.device)
        v = torch.linalg.cross(a, x)

    v = v / torch.linalg.norm(v)
    skew_sym_mat = torch.Tensor(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    ).to(a.device)
    theta = torch.acos(torch.clip(torch.dot(a, b), -1, 1))

    # Rodrigues rotation formula. https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    return torch.eye(3).to(a.device) + torch.sin(theta) * skew_sym_mat + (1 - torch.cos(theta)) * (skew_sym_mat @ skew_sym_mat)



# The following functions are copied from https://github.com/Stability-AI/stable-virtual-camera/blob/a4b72c6b3c08ca0da928e38b2bafbea9c237f3b9/seva/geometry.py

def to_hom(X):
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)
    return X_hom

def to_hom_pose(pose):
    # get homogeneous coordinates of the input pose
    if pose.shape[-2:] == (3, 4):
        pose_hom = torch.eye(4, device=pose.device)[None].repeat(pose.shape[0], 1, 1)
        pose_hom[:, :3, :] = pose
        return pose_hom
    return pose

def get_image_grid(img_h, img_w):
    # add 0.5 is VERY important especially when your img_h and img_w
    # is not very large (e.g., 72)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    y_range = torch.arange(img_h, dtype=torch.float32).add_(0.5)
    x_range = torch.arange(img_w, dtype=torch.float32).add_(0.5)
    Y, X = torch.meshgrid(y_range, x_range, indexing="ij")  # [H,W]
    xy_grid = torch.stack([X, Y], dim=-1).view(-1, 2)  # [HW,2]
    return to_hom(xy_grid)  # [HW,3]


def img2cam(X, cam_intr):
    return X @ cam_intr.inverse().transpose(-1, -2)


def cam2world(X, pose):
    X_hom = to_hom(X)
    pose_inv = torch.linalg.inv(to_hom_pose(pose))[..., :3, :4]
    return X_hom @ pose_inv.transpose(-1, -2)

def get_center_and_ray(img_h, img_w, pose, intr):  # [HW,2]
    # given the intrinsic/extrinsic matrices, get the camera center and ray directions]
    # assert(opt.camera.model=="perspective")

    # compute center and ray
    grid_img = get_image_grid(img_h, img_w)  # [HW,3]
    grid_3D_cam = img2cam(grid_img.to(intr.device), intr.float())  # [B,HW,3]
    center_3D_cam = torch.zeros_like(grid_3D_cam)  # [B,HW,3]

    # transform from camera to world coordinates
    grid_3D = cam2world(grid_3D_cam, pose)  # [B,HW,3]
    center_3D = cam2world(center_3D_cam, pose)  # [B,HW,3]
    ray = grid_3D - center_3D  # [B,HW,3]

    return center_3D, ray, grid_3D_cam


def get_plucker_coordinates(
    extrinsics_src,
    extrinsics,
    intrinsics,
    target_size=[72, 72],
):
    if not (
        torch.all(intrinsics[:, :2, -1] >= 0)
        and torch.all(intrinsics[:, :2, -1] <= 1)
    ):
        intrinsics[:, :2] /= intrinsics.new_tensor(target_size).view(1, -1, 1) * 8
    # you should ensure the intrisics are expressed in
    # resolution-independent normalized image coordinates just performing a
    # very simple verification here checking if principal points are
    # between 0 and 1
    assert (
        torch.all(intrinsics[:, :2, -1] >= 0)
        and torch.all(intrinsics[:, :2, -1] <= 1)
    ), "Intrinsics should be expressed in resolution-independent normalized image coordinates."

    c2w_src = torch.linalg.inv(extrinsics_src)
    # transform coordinates from the source camera's coordinate system to the coordinate system of the respective camera
    extrinsics_rel = torch.einsum(
        "vnm,vmp->vnp", extrinsics, c2w_src[None].repeat(extrinsics.shape[0], 1, 1)
    )

    intrinsics[:, :2] *= extrinsics.new_tensor(
        [
            target_size[1],  # w
            target_size[0],  # h
        ]
    ).view(1, -1, 1)
    centers, rays, grid_cam = get_center_and_ray(
        img_h=target_size[0],
        img_w=target_size[1],
        pose=extrinsics_rel[:, :3, :],
        intr=intrinsics,
    )

    rays = torch.nn.functional.normalize(rays, dim=-1)
    plucker = torch.cat((rays, torch.cross(centers, rays, dim=-1)), dim=-1)
    plucker = plucker.permute(2, 0, 1).reshape(-1, plucker.shape[0], *target_size)
    return plucker


def so3_log(R):
    # R: [...,3,3]
    cos_theta = (R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0
    cos_theta = cos_theta.clamp(-1+1e-7, 1-1e-7)
    theta = torch.acos(cos_theta)
    W = (R - R.transpose(-1, -2)) / 2.0
    w = torch.stack([W[...,2,1], W[...,0,2], W[...,1,0]], dim=-1)  # [...,3]
    s = torch.sin(theta)[..., None]
    small = theta.abs() < 1e-5
    coef = torch.where(small, 0.5 - (theta**2)[...,None]/12.0, theta[...,None]/(2.0*s + 1e-12))
    return coef * w

def so3_exp(w):
    # w: [...,3]
    theta = torch.linalg.norm(w, dim=-1, keepdim=True)
    small = theta < 1e-5
    k = torch.where(small, 1.0 + theta**2/6.0, torch.sin(theta)/ (theta+1e-12))
    kk = torch.where(small, 0.5 - theta**2/24.0, (1 - torch.cos(theta))/ (theta**2+1e-12))
    wx, wy, wz = w[...,0], w[...,1], w[...,2]
    O = torch.zeros_like(wx)
    W = torch.stack([
        torch.stack([O,   -wz,  wy], dim=-1),
        torch.stack([wz,   O,  -wx], dim=-1),
        torch.stack([-wy, wx,   O], dim=-1),
    ], dim=-2)
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand(W.shape)
    R = I + k[...,None]*W + kk[...,None]*W@W
    return R

def se3_log(T):
    # T: [...,4,4]
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    w = so3_log(R)                     # [...,3]
    theta = torch.linalg.norm(w, dim=-1, keepdim=True)
    small = theta < 1e-5
    wx, wy, wz = w[...,0], w[...,1], w[...,2]
    O = torch.zeros_like(wx)
    W = torch.stack([
        torch.stack([O,   -wz,  wy], dim=-1),
        torch.stack([wz,   O,  -wx], dim=-1),
        torch.stack([-wy, wx,   O], dim=-1),
    ], dim=-2)                          # [...,3,3]
    I = torch.eye(3, device=T.device, dtype=T.dtype).expand(W.shape)
    A = torch.where(small, 1.0 - theta**2/6.0, (1 - torch.cos(theta)) / (theta**2 + 1e-12)).unsqueeze(-1)
    B = torch.where(small, 0.5 - theta**2/24.0, (theta - torch.sin(theta)) / (theta**3 + 1e-12)).unsqueeze(-1)
    V = I + A*W + B*(W@W)               # [...,3,3]
    v = torch.linalg.solve(V, t.unsqueeze(-1)).squeeze(-1)  # [...,3]
    xi = torch.cat([v, w], dim=-1)      # [...,6]
    return xi

def se3_exp(xi):
    # xi: [...,6] -> T: [...,4,4]
    v, w = xi[..., :3], xi[..., 3:]
    R = so3_exp(w)
    theta = torch.linalg.norm(w, dim=-1, keepdim=True)
    small = theta < 1e-5
    wx, wy, wz = w[...,0], w[...,1], w[...,2]
    O = torch.zeros_like(wx)
    W = torch.stack([
        torch.stack([O,   -wz,  wy], dim=-1),
        torch.stack([wz,   O,  -wx], dim=-1),
        torch.stack([-wy, wx,   O], dim=-1),
    ], dim=-2)
    I = torch.eye(3, device=xi.device, dtype=xi.dtype).expand(W.shape)
    A = torch.where(small, 1.0 - theta**2/6.0, (1 - torch.cos(theta)) / (theta**2 + 1e-12)).unsqueeze(-1)
    B = torch.where(small, 0.5 - theta**2/24.0, (theta - torch.sin(theta)) / (theta**3 + 1e-12)).unsqueeze(-1)
    V = I + A*W + B*(W@W)               # [...,3,3]
    t = (V @ v.unsqueeze(-1)).squeeze(-1)
    T = torch.zeros((*R.shape[:-2], 4, 4), device=xi.device, dtype=xi.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T

def se3_inv(T):
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Ti = torch.zeros_like(T)
    Ti[..., :3, :3] = R.transpose(-1, -2)
    Ti[..., :3, 3]  = -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)
    Ti[..., 3, 3]   = 1.0
    return Ti

def karcher_mean_se3(Ts, iters=10):
    # Ts: [N,4,4]
    Tbar = Ts[Ts.shape[0]//2].clone()  # 以中位初始化更稳
    for _ in range(iters):
        xi = se3_log(se3_inv(Tbar) @ Ts)  # [N,6]
        delta = xi.mean(dim=0)            # [6]
        Tbar = Tbar @ se3_exp(delta)
        if torch.linalg.norm(delta) < 1e-10:
            break
    return Tbar

def compress_16_to_4_karcher(Ts):
    # Ts: [16,4,4]
    outs = []
    for k in range(4):
        seg = Ts[k*4:(k+1)*4]
        outs.append(karcher_mean_se3(seg))
    return torch.stack(outs, dim=0)  # [4,4,4]

def se3_geodesic_interpolate(Ti, Tj, alpha):
    # alpha ∈ [0,1]
    return Ti @ se3_exp(alpha * se3_log(se3_inv(Ti) @ Tj))

def compress_16_to_4_geodesic(Ts):
    # 取段中心时间 1.5, 5.5, 9.5, 13.5
    times = torch.tensor([1.5, 5.5, 9.5, 13.5], device=Ts.device, dtype=Ts.dtype)
    outs = []
    for s in times:
        k = int(torch.floor(s).item())
        a = (s - k).item()
        outs.append(se3_geodesic_interpolate(Ts[k], Ts[min(k+1, 15)], a))
    return torch.stack(outs, dim=0)  # [4,4,4]