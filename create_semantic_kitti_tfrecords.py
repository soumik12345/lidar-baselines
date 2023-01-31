import wandb
import argparse
import numpy as np
from tqdm.auto import tqdm

from lidar_baselines.dataloader.semantic_kitti.convert import (
    SemanticKITTITFRecordConverter,
)
from lidar_baselines.dataloader.semantic_kitti.maps import (
    get_segmentation_classes,
    get_segmentation_colors,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse flags for conversion of Semantic-KITTI sequences to TFRecord format from Numpy format"
    )
    parser.add_argument(
        "--wandb_project", type=str, required=True, help="Weights & Biases Project"
    )
    parser.add_argument(
        "--wandb_entity", type=str, required=True, help="Weights & Biases Entity"
    )
    args = parser.parse_args()

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        job_type="tfrecord-conversion",
    )

    color_map = get_segmentation_colors()
    segmentation_classes = get_segmentation_classes()
    num_classes = len(segmentation_classes.keys())
    categories = [""] * num_classes
    for idx in range(num_classes):
        categories[idx] = segmentation_classes[idx]

    inspection_runs = wandb.Api().runs(
        path=f"{args.wandb_entity}/{args.wandb_project}",
        filters={
            "jobType": "inspection",
            "display_name": {"$regex": "^inspect/semantic-kitti/*"},
        },
        order="+display_name",
    )

    for inspection_run in tqdm(inspection_runs):
        split_name = inspection_run.display_name.split("/")[-1]
        numpy_dataset_artifact_address = (
            f"geekyrakshit/point-cloud-voxelize/semantic-kitti-numpy-{split_name}:latest"
        )
        output_file = f"semantic-kitti-{split_name}.tfrecord"

        input_mean = [
            inspection_run.summary["mean/lidar_scan_x"],
            inspection_run.summary["mean/lidar_scan_y"],
            inspection_run.summary["mean/lidar_scan_z"],
            inspection_run.summary["mean/intensity"],
            inspection_run.summary["mean/depth"],
        ]
        input_std = [
            inspection_run.summary["std/lidar_scan_x"],
            inspection_run.summary["std/lidar_scan_y"],
            inspection_run.summary["std/lidar_scan_z"],
            inspection_run.summary["std/intensity"],
            inspection_run.summary["std/depth"],
        ]

        tfrecord_converter = SemanticKITTITFRecordConverter(
            numpy_dataset_artifact_address=numpy_dataset_artifact_address,
            input_mean=input_mean,
            input_std=input_std,
            categories=categories,
            class_weight=np.ones(num_classes),
            color_map=color_map,
        )
        tfrecord_converter.create_tfrecord(output_file=output_file)

        artifact = wandb.Artifact(
            f"Semantic-KITTI-{split_name}", type="TFRecord-Dataset"
        )
        artifact.add_file(output_file)
        wandb.log_artifact(artifact)

    wandb.finish()
