"""
Synthetic Scene Generator
==========================
Generates synthetic training data entirely in Python — no downloads required.

Creates a simple 3D scene (colored sphere + optional cube) and renders
ground-truth images from multiple camera viewpoints using classical
ray-sphere/ray-box intersection (not neural rendering).

This lets you train and test PI-NeRF immediately on a CPU without
needing the full NeRF Blender dataset.

Scene:
  - Unit sphere at origin with Lambertian + specular shading
  - Background: white
  - Camera: orbiting sphere at configurable distance and elevation
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
import os


# ── Ray-Sphere Intersection ────────────────────────────────────────────────────

def ray_sphere_intersect(
    rays_o: np.ndarray,  # [N, 3]
    rays_d: np.ndarray,  # [N, 3]
    center: np.ndarray = np.array([0., 0., 0.]),
    radius: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ray-sphere intersection analytically.

    Ray: p(t) = o + t·d
    Sphere: ||p - c||² = r²

    Substituting: ||o + td - c||² = r²
    (d·d)t² + 2(o-c)·d·t + (o-c)·(o-c) - r² = 0

    Returns:
        hit:    [N] bool mask
        t:      [N] intersection depth
        normals:[N, 3] outward surface normals
    """
    N = rays_o.shape[0]
    oc = rays_o - center[None, :]  # [N, 3]

    a = np.sum(rays_d * rays_d, axis=-1)         # [N]
    b = 2.0 * np.sum(oc * rays_d, axis=-1)       # [N]
    c = np.sum(oc * oc, axis=-1) - radius ** 2   # [N]

    discriminant = b * b - 4 * a * c             # [N]
    hit = discriminant >= 0                       # [N]

    t = np.full(N, np.inf)
    t_hit = (-b[hit] - np.sqrt(np.maximum(discriminant[hit], 0))) / (2 * a[hit])
    t2 = (-b[hit] + np.sqrt(np.maximum(discriminant[hit], 0))) / (2 * a[hit])
    # Choose smallest positive t
    mask_neg = t_hit < 0
    t_hit[mask_neg] = t2[mask_neg]
    valid = t_hit > 0
    hit_indices = np.where(hit)[0][valid]
    hit = np.zeros(N, dtype=bool)
    hit[hit_indices] = True
    t[hit_indices] = t_hit[valid]

    # Normals at intersection points
    pts = rays_o + t[:, None] * rays_d  # [N, 3]
    normals = (pts - center[None, :]) / radius  # [N, 3] (outward unit normals)

    return hit, t, normals


def shade_sphere(
    hit: np.ndarray,
    normals: np.ndarray,
    rays_d: np.ndarray,
    sphere_color: np.ndarray = np.array([0.8, 0.3, 0.2]),
    light_dir: np.ndarray = np.array([1., 1., 1.]),
) -> np.ndarray:
    """
    Lambertian + specular shading.

    L = k_d · (n · l) + k_s · (r · v)^α + ambient
    """
    pixels = np.ones((hit.shape[0], 3))  # White background

    if hit.sum() == 0:
        return pixels

    n = normals[hit]  # [M, 3]
    v = -rays_d[hit]  # View direction
    l = light_dir / (np.linalg.norm(light_dir) + 1e-8)  # Unit light dir

    # Diffuse (Lambertian)
    n_norm = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8)
    diffuse = np.maximum(np.dot(n_norm, l), 0.0)  # [M]

    # Specular (Blinn-Phong)
    h = (l + v) / (np.linalg.norm(l + v, axis=-1, keepdims=True) + 1e-8)
    specular = np.maximum(np.sum(n_norm * h, axis=-1), 0.0) ** 32  # [M]

    # Ambient
    ambient = 0.15

    shade = ambient + 0.7 * diffuse[:, None] * sphere_color[None, :] \
           + 0.3 * specular[:, None]
    pixels[hit] = np.clip(shade, 0, 1)

    return pixels


# ── Camera Utilities ───────────────────────────────────────────────────────────

def spherical_pose(theta: float, phi: float, radius: float = 3.0) -> np.ndarray:
    """
    Camera pose (c2w matrix) on a sphere around the origin.

    Args:
        theta: Azimuth angle (radians)
        phi:   Elevation angle (radians)
        radius: Distance from origin

    Returns:
        c2w: [4, 4] camera-to-world matrix
    """
    # Camera position
    x = radius * np.cos(phi) * np.cos(theta)
    y = radius * np.cos(phi) * np.sin(theta)
    z = radius * np.sin(phi)
    pos = np.array([x, y, z])

    # Look-at: camera looks toward origin
    forward = -pos / (np.linalg.norm(pos) + 1e-8)

    # Up vector (world up, handle gimbal lock)
    world_up = np.array([0., 0., 1.])
    if abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([0., 1., 0.])

    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)

    # Build c2w = [right | up | -forward | pos]
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward  # Camera looks along -z
    c2w[:3, 3] = pos

    return c2w


def render_synthetic_image(
    H: int,
    W: int,
    focal: float,
    c2w: np.ndarray,
    sphere_color: np.ndarray = np.array([0.8, 0.3, 0.2]),
) -> np.ndarray:
    """
    Render a ground-truth image of the synthetic sphere scene.

    Args:
        H, W:    Image dimensions
        focal:   Focal length
        c2w:     [4, 4] camera-to-world matrix
        sphere_color: RGB color of sphere

    Returns:
        image: [H, W, 3] float32 in [0, 1]
    """
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")

    # Camera-space ray directions
    dirs = np.stack([
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,
        -np.ones_like(i),
    ], axis=-1)  # [H, W, 3]

    # World-space directions
    R = c2w[:3, :3]
    dirs_world = (dirs[..., None, :] * R).sum(-1)  # [H, W, 3]
    dirs_world /= np.linalg.norm(dirs_world, axis=-1, keepdims=True) + 1e-8

    rays_o = np.tile(c2w[:3, 3][None, None, :], (H, W, 1))  # [H, W, 3]

    rays_o_flat = rays_o.reshape(-1, 3)
    rays_d_flat = dirs_world.reshape(-1, 3)

    hit, t, normals = ray_sphere_intersect(rays_o_flat, rays_d_flat)
    pixels = shade_sphere(hit, normals, rays_d_flat, sphere_color)

    return pixels.reshape(H, W, 3).astype(np.float32)


# ── Dataset Class ──────────────────────────────────────────────────────────────

class SyntheticSphereDataset(torch.utils.data.Dataset):
    """
    On-the-fly synthetic NeRF dataset.

    Generates camera poses on a hemisphere, renders ground-truth images
    using classical ray tracing, and provides rays + pixel colors for training.
    """

    def __init__(
        self,
        n_views: int = 50,
        H: int = 64,
        W: int = 64,
        focal_scale: float = 1.0,
        split: str = "train",
        seed: int = 42,
    ):
        super().__init__()
        self.H = H
        self.W = W
        self.focal = W * focal_scale
        self.split = split

        rng = np.random.RandomState(seed if split == "train" else seed + 999)

        # Generate camera poses
        thetas = rng.uniform(0, 2 * np.pi, n_views)
        phis = rng.uniform(np.pi / 8, np.pi / 3, n_views)

        self.poses = []
        self.images = []

        print(f"Generating {n_views} synthetic {split} views ({H}×{W})...")
        for i in range(n_views):
            c2w = spherical_pose(thetas[i], phis[i], radius=3.0)
            img = render_synthetic_image(H, W, self.focal, c2w)
            self.poses.append(torch.FloatTensor(c2w))
            self.images.append(torch.FloatTensor(img))

        self.poses = torch.stack(self.poses)    # [N, 4, 4]
        self.images = torch.stack(self.images)  # [N, H, W, 3]

        print(f"  Done. Images: {self.images.shape}, Poses: {self.poses.shape}")

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],   # [H, W, 3]
            "pose": self.poses[idx],     # [4, 4]
            "focal": torch.tensor(self.focal),
        }

    def get_all_rays(self):
        """Return all rays and pixels flattened (for full-image evaluation)."""
        from renderer.ray_utils import get_rays
        all_o, all_d, all_rgb = [], [], []
        for i in range(len(self)):
            o, d = get_rays(self.H, self.W, self.focal, self.poses[i])
            all_o.append(o.reshape(-1, 3))
            all_d.append(d.reshape(-1, 3))
            all_rgb.append(self.images[i].reshape(-1, 3))
        return (
            torch.cat(all_o),
            torch.cat(all_d),
            torch.cat(all_rgb),
        )
