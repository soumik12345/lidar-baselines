import os
from glob import glob
from typing import Optional

import numpy as np
from tqdm.auto import tqdm

import wandb

from .laserscan import SemLaserScan
from .maps import get_color_map, get_label_map, get_label_to_name
from .utils import (
    visualize_point_cloud_with_intensity,
    visualize_point_cloud_with_labels,
    compute_class_frequency,
    plot_frequency_dict,
)


class SemanticKITTIConverter:
    """
    Reference:

    (1) https://github.com/ika-rwth-aachen/PCLSegmentation/blob/main/dataset_convert/semantic_kitti_sequence.py
    """

    def __init__(self, artifact_address: str, sequence_id: str) -> None:
        self.artifact_address = artifact_address
        self.sequence_id = sequence_id
        self.resolution = [64, 1024]
        self.label_map = get_label_map()
        self.color_map = get_color_map()
        self.lidar_scans, self.lidar_labels = self.get_lidar_scan_paths()
        self.laser_scan = SemLaserScan(
            nclasses=len(self.color_map), sem_color_dict=self.color_map, project=True
        )
        self.vfunc_get_label = np.vectorize(self.label_map.get)
        columns = ["Sequence-ID", "Semantic-Labels", "Depth", "Intensity"]
        self.categories = [value for _, value in get_label_to_name().items()]
        columns += ["Frequency-" + category for category in self.categories]
        self.table = wandb.Table(columns=columns)
        self.global_frequency_dict = {category: 0 for category in self.categories}

    def __len__(self):
        return len(self.lidar_scans)

    def get_lidar_scan_paths(self):
        dataset_artifact = (
            wandb.Api().artifact(self.artifact_address, type="dataset")
            if wandb.run is None
            else wandb.use_artifact(self.artifact_address, type="dataset")
        )
        dataset_dir = dataset_artifact.download()

        lidar_scan_dir = os.path.join(
            dataset_dir, "sequences", self.sequence_id, "velodyne"
        )
        label_dir = os.path.join(dataset_dir, "sequences", self.sequence_id, "labels")
        assert os.path.isdir(
            lidar_scan_dir
        ), f"Lidar scan directory {lidar_scan_dir} doesn't exist"
        assert os.path.isdir(label_dir), f"Label directory {label_dir} doesn't exist"

        lidar_scans = sorted(glob(os.path.join(lidar_scan_dir, "*")))
        lidar_labels = sorted(glob(os.path.join(label_dir, "*")))
        assert len(lidar_scans) == len(
            lidar_labels
        ), f"Number of scans ({len(lidar_scans)}) is not equal to number of labels ({len(lidar_labels)})"

        return lidar_scans, lidar_labels

    def extract_tensor(
        self, lidar_scan, lidar_label, sequence_dir, index, log_visualizations
    ):
        self.laser_scan.open_scan(lidar_scan)
        self.laser_scan.open_label(lidar_label)

        # check if the projected depth is positive
        mask = self.laser_scan.proj_range > 0

        self.laser_scan.proj_range[~mask] = 0.0
        self.laser_scan.proj_xyz[~mask] = 0.0
        self.laser_scan.proj_remission[~mask] = 0.0

        # shape (64, 1024, 6)
        combined_tensor = np.concatenate(
            [
                self.laser_scan.proj_xyz,
                self.laser_scan.proj_remission.reshape((*self.resolution, 1)),
                self.laser_scan.proj_range.reshape((*self.resolution, 1)),
                # map class labels to values between 0 and 33
                self.vfunc_get_label(self.laser_scan.proj_sem_label).reshape(
                    (*self.resolution, 1)
                ),
            ],
            axis=2,
        )

        if wandb.run is not None and log_visualizations:
            point_cloud_label = np.array(
                [
                    np.array(self.color_map[label])
                    for label in self.laser_scan.proj_sem_label.flatten().tolist()
                ]
            )
            frequency_dict = compute_class_frequency(self.laser_scan.proj_sem_label)
            for key, value in frequency_dict.items():
                self.global_frequency_dict[key] += value

            self.table.add_data(
                self.sequence_id,
                visualize_point_cloud_with_labels(
                    self.laser_scan.proj_xyz, point_cloud_label
                ),
                visualize_point_cloud_with_intensity(
                    self.laser_scan.proj_xyz, self.laser_scan.proj_range
                ),
                visualize_point_cloud_with_intensity(
                    self.laser_scan.proj_xyz, self.laser_scan.proj_remission
                ),
                *[frequency_dict[category] for category in self.categories],
            )

        np.save(os.path.join(sequence_dir, f"{index}.npy"), combined_tensor)

    def save_numpy_dataset_as_artifact(
        self, output_dir, lower_bound_index, upper_bound_index, log_visualizations
    ):
        artifact = wandb.Artifact(
            f"semantic-kitti-numpy-{self.sequence_id}",
            type="numpy-dataset",
            metadata={
                "sequence_id": self.sequence_id,
                "lower_bound_index": lower_bound_index,
                "upper_bound_index": upper_bound_index,
            },
        )
        artifact.add_dir(output_dir)
        artifact_aliases = ["latest"]
        artifact_aliases = (
            artifact_aliases + [f"split-{lower_bound_index}-{upper_bound_index}"]
            if not log_visualizations
            else artifact_aliases
        )
        wandb.log_artifact(artifact, aliases=artifact_aliases)

    def save_data(
        self,
        output_dir: Optional[str] = None,
        lower_bound_index: Optional[int] = None,
        upper_bound_index: Optional[int] = None,
        log_visualizations: bool = False,
    ):
        if output_dir is not None:
            sequence_dir = os.path.join(output_dir, self.sequence_id)
            os.makedirs(sequence_dir, exist_ok=True)

        if lower_bound_index is None or upper_bound_index is None:
            progress_bar = tqdm(
                enumerate(zip(self.lidar_scans, self.lidar_labels)),
                total=len(self.lidar_scans),
            )
        elif upper_bound_index > len(self.lidar_scans):
            progress_bar = tqdm(
                enumerate(
                    zip(
                        self.lidar_scans[lower_bound_index:],
                        self.lidar_labels[lower_bound_index:],
                    )
                ),
                total=len(self.lidar_scans) - lower_bound_index,
            )
        elif lower_bound_index > len(self.lidar_scans):
            raise ValueError(
                f"Lower bound index {lower_bound_index} is greater than number of scans {len(self.lidar_scans)}"
            )
        else:
            progress_bar = tqdm(
                enumerate(
                    zip(
                        self.lidar_scans[lower_bound_index:upper_bound_index],
                        self.lidar_labels[lower_bound_index:upper_bound_index],
                    )
                ),
                total=upper_bound_index - lower_bound_index,
            )

        for index, (lidar_scan, lidar_label) in progress_bar:
            self.extract_tensor(
                lidar_scan, lidar_label, sequence_dir, index, log_visualizations
            )

        if log_visualizations:
            plot_frequency_dict(self.global_frequency_dict, self.sequence_id)
            wandb.log({"Semantic-KITTI": self.table})

        self.save_numpy_dataset_as_artifact(
            output_dir, lower_bound_index, upper_bound_index, log_visualizations
        )
