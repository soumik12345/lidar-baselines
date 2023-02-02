import os
from glob import glob
from typing import Optional, List

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

import wandb

from .laserscan import SemLaserScan
from .maps import get_color_map, get_label_map, get_label_to_name
from .utils import (
    compute_class_frequency,
    plot_frequency_dict,
    visualize_point_cloud_with_intensity,
    visualize_point_cloud_with_labels,
    create_tfrecord_feature,
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
            wandb.Api().artifact(self.artifact_address, type="raw_dataset")
            if wandb.run is None
            else wandb.use_artifact(self.artifact_address, type="raw_dataset")
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

    def extract_and_save_tensor(
        self,
        lidar_scan,
        lidar_label,
        index,
        log_visualizations,
        save_dir,
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

        np.save(
            os.path.join(save_dir, f"{index}.npy"),
            combined_tensor,
        )

    def save_numpy_dataset_as_artifact(
        self, output_dir, lower_bound_index, upper_bound_index
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
        artifact_aliases = ["latest", f"split-{lower_bound_index}-{upper_bound_index}"]
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

        progress_bar_description = f"Converting scans to numpy for Sequence-{self.sequence_id} Split-{lower_bound_index}-{upper_bound_index}"
        if lower_bound_index is None or upper_bound_index is None:
            progress_bar = tqdm(
                enumerate(zip(self.lidar_scans, self.lidar_labels)),
                total=len(self.lidar_scans),
                desc=progress_bar_description,
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
                desc=progress_bar_description,
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
                desc=progress_bar_description,
            )

        save_dir = os.path.join(
            sequence_dir, f"{lower_bound_index}-{upper_bound_index}"
        )
        os.makedirs(save_dir)
        for index, (lidar_scan, lidar_label) in progress_bar:
            self.extract_and_save_tensor(
                lidar_scan, lidar_label, index, log_visualizations, save_dir
            )

        if log_visualizations:
            plot_frequency_dict(self.global_frequency_dict, self.sequence_id)
            wandb.log({"Semantic-KITTI": self.table})

        self.save_numpy_dataset_as_artifact(
            save_dir, lower_bound_index, upper_bound_index
        )

    def save_sequence(
        self,
        split_size: int,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        output_dir: Optional[str] = None,
        log_visualizations: bool = False,
    ):
        os.makedirs(output_dir, exist_ok=True)
        for idx in range(0, len(self), split_size):
            upper_bound_index = (
                idx + split_size if idx + split_size < len(self) else len(self)
            )
            with wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=f"Semantic-KITTI/{self.sequence_id}/{idx}-{upper_bound_index}",
                job_type="numpy-conversion",
                tags=[
                    "semantic-kitti",
                    "numpy-conversion",
                    f"sequence-{self.sequence_id}",
                    f"split-{idx}-{upper_bound_index}",
                ],
            ):
                self.save_data(
                    output_dir=output_dir,
                    lower_bound_index=idx,
                    upper_bound_index=upper_bound_index,
                    log_visualizations=log_visualizations,
                )


class SemanticKITTITFRecordConverter:
    def __init__(
        self,
        numpy_dataset_artifact_address: str,
        input_mean: List,
        input_std: List,
        categories: List,
        class_weight: List,
        color_map: List[List],
    ):
        self.numpy_dataset_artifact_address = numpy_dataset_artifact_address
        self.input_mean = np.array([[input_mean]])
        self.input_std = np.array([[input_std]])
        self.categories = categories
        self.class_weight = class_weight
        self.color_map = color_map
        self.fetch_numpy_dataset_artifact()

    def fetch_numpy_dataset_artifact(self):
        numpy_dataset_artifact = (
            wandb.Api().artifact(
                self.numpy_dataset_artifact_address, type="numpy-dataset"
            )
            if wandb.run is None
            else wandb.use_artifact(
                self.numpy_dataset_artifact_address, type="numpy-dataset"
            )
        )
        artifact_dir = numpy_dataset_artifact.download()
        self.numpy_dataset_paths = glob(os.path.join(artifact_dir, "*", "*.npy"))

    def __len__(self):
        return len(self.numpy_dataset_paths)

    def load_numpy_tensor(self, numpy_file_path: tf.Tensor):
        # Load numpy file
        numpy_data = np.load(numpy_file_path).astype(np.float32, copy=False)

        input_data = numpy_data[:, :, :5]  # Get point cloud, intensity and depth
        segmentation_labels = numpy_data[:, :, 5]  # Get segmentation labels

        # Create a binary mask to cover only positive depth
        lidar_mask = input_data[:, :, 4] > 0

        # Normalize input data using the mean and standard deviation
        input_data = (input_data - self.input_mean) / self.input_std

        # Apply mask on input data and segmentation labels
        input_data[~lidar_mask] = 0.0
        segmentation_labels[~lidar_mask] = self.categories.index("None")

        # Append mask to input data
        input_data = np.append(input_data, np.expand_dims(lidar_mask, -1), axis=2)

        # construct class-wise weighting defined in the configuration
        class_weight = np.zeros(segmentation_labels.shape)
        for l in range(len(self.categories)):
            class_weight[segmentation_labels == l] = self.class_weight[int(l)]

        return (
            input_data.astype("float32"),
            lidar_mask.astype("bool"),
            segmentation_labels.astype("int32"),
            class_weight.astype("float32"),
        )

    def create_example(self, numpy_file_path: tf.Tensor):
        (
            input_data,
            lidar_mask,
            segmentation_labels,
            class_weight,
        ) = self.load_numpy_tensor(numpy_file_path)
        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    "input_data": create_tfrecord_feature(input_data),
                    "lidar_mask": create_tfrecord_feature(lidar_mask),
                    "segmentation_labels": create_tfrecord_feature(segmentation_labels),
                    "class_weight": create_tfrecord_feature(class_weight),
                }
            )
        )

    def create_tfrecord(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        pbar = tqdm(
            enumerate(self.numpy_dataset_paths),
            total=len(self.numpy_dataset_paths),
            desc="Creating TFRecords",
        )
        for idx, numpy_file in pbar:
            semantic_kitti_split = self.numpy_dataset_artifact_address.split(":")[
                0
            ].split("-")[-1]
            pbar.set_description(f"Creating TFRecords for split {semantic_kitti_split}")
            with tf.io.TFRecordWriter(
                os.path.join(output_dir, f"split-{semantic_kitti_split}-{idx}.tfrecord")
            ) as writer:
                example = self.create_example(numpy_file)
                writer.write(example.SerializeToString())
