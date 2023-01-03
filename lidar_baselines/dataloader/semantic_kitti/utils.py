import os
from typing import Optional

import numpy as np
import open3d as o3d

import wandb


def visualize_point_cloud_with_intensity(point_cloud, intensity):
    point_cloud = point_cloud.reshape(-1, 3)
    intensity = intensity.reshape(-1, 1)
    normalized_intensity = 255.0 * (
        np.ones_like(intensity) - intensity / (np.max(intensity))
    )
    gray_point_cloud_color = (
        0.30 * normalized_intensity
        + 0.59 * normalized_intensity
        + 0.11 * normalized_intensity
    )
    gray_point_cloud_color = np.concatenate(
        [gray_point_cloud_color.reshape(-1, 1)] * 3, axis=-1
    )
    return wandb.Object3D(
        np.concatenate(
            [point_cloud, gray_point_cloud_color],
            axis=-1,
        )
    )


def visualize_point_cloud_with_labels(
    point_cloud,
    point_cloud_label_colors,
    voxel_size: Optional[float] = None,
    voxel_size_factor: Optional[float] = 5e-3,
    voxel_size_precision: Optional[int] = 4,
):
    """
    Reference:

    (1) https://docs.wandb.ai/guides/track/log/media#3d-visualizations
    (2) https://towardsdatascience.com/how-to-automate-voxel-modelling-of-3d-point-cloud-with-python-459f4d43a227
    """
    point_cloud = point_cloud.reshape(-1, 3)
    if voxel_size is None:
        return wandb.Object3D(
            np.concatenate(
                [point_cloud, point_cloud_label_colors],
                axis=-1,
            )
        )
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            np.vstack(
                (point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
            ).transpose()
        )
        pcd.colors = o3d.utility.Vector3dVector(
            np.vstack(
                (
                    point_cloud_label_colors[:, 0],
                    point_cloud_label_colors[:, 1],
                    point_cloud_label_colors[:, 2],
                )
            ).transpose()
            / 255
        )
        v_size = round(
            max(pcd.get_max_bound() - pcd.get_min_bound()) * voxel_size_factor,
            voxel_size_precision,
        )
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd, voxel_size=v_size
        )
        voxels = voxel_grid.get_voxels()
        vox_mesh = o3d.geometry.TriangleMesh()
        for v in voxels:
            cube = o3d.geometry.TriangleMesh.create_box(
                width=voxel_size, height=voxel_size, depth=voxel_size
            )
            cube.paint_uniform_color(v.color)
            cube.translate(v.grid_index, relative=False)
            vox_mesh += cube
        vox_mesh.translate([0.5, 0.5, 0.5], relative=True)
        vox_mesh.scale(v_size, [0, 0, 0])
        vox_mesh.translate(voxel_grid.origin, relative=True)
        o3d.io.write_triangle_mesh("voxel_mesh.glb", vox_mesh)
        wandb_3d_object = wandb.Object3D(open("voxel_mesh.glb"))
        os.remove("voxel_mesh.glb")
        return wandb_3d_object
