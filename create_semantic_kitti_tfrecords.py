import os
import wandb
import argparse
import numpy as np
from glob import glob
from tqdm.auto import tqdm
from google.cloud import storage

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
    parser.add_argument("--gc_project", type=str, help="Name of Google Cloud Project")
    parser.add_argument("--gs_bucket", type=str, help="Name of Google Storage Bucket")
    args = parser.parse_args()

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        job_type="tfrecord-conversion",
    )

    storage_client = storage.Client(project=args.gc_project)

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

    for inspection_run in inspection_runs:
        split_name = inspection_run.display_name.split("/")[-1]
        numpy_dataset_artifact_address = f"geekyrakshit/point-cloud-voxelize/semantic-kitti-numpy-{split_name}:latest"
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
        tfrecord_converter.create_tfrecord(output_dir=f"tfrecords/{split_name}")

        bucket = storage_client.get_bucket(args.gs_bucket)

        for filepath in tqdm(
            glob(os.path.join("tfrecords", split_name, "*.tfrecord")),
            desc=f"Uploading TFRecords for split {split_name} to gs://{args.gs_bucket}",
        ):
            blob = bucket.blob(filepath)
            blob.upload_from_filename(filepath)

        artifact = wandb.Artifact(
            f"Semantic-KITTI-{split_name}", type="TFRecord-Dataset"
        )
        artifact.add_reference(
            os.path.join("gs://", args.gs_bucket, "tfrecords", split_name)
        )
        wandb.log_artifact(artifact)

    wandb.finish()
