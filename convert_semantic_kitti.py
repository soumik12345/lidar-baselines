import argparse
import shutil

import ray
import wandb
from lidar_baselines.dataloader.semantic_kitti import SemanticKITTIConverter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse flags for conversion of Semantic-KITTI sequences to numpy format"
    )
    parser.add_argument(
        "--sequence_id", type=str, required=True, help="Semantic-KITTI sequence ID"
    )
    parser.add_argument(
        "--lower_bound_index", type=int, required=True, help="Lower bound index"
    )
    parser.add_argument(
        "--upper_bound_index", type=int, required=True, help="Upper bound index"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="semantic_kitti_data",
        help="Output directory for dumping the converted data",
    )
    args = parser.parse_args()

    run_name = "semantic-kitti/"
    run_name += f"{args.sequence_id}/"
    run_name += f"{args.lower_bound_index}-{args.upper_bound_index}"

    with wandb.init(
        project="point-cloud-voxelize",
        entity="geekyrakshit",
        name=run_name,
        job_type="numpy-conversion",
        tags=[
            "semantic-kitti",
            "numpy-conversion",
            f"sequence-{args.sequence_id}",
            f"split-{args.lower_bound_index}-{args.upper_bound_index}",
        ],
        config=args.__dict__,
    ):
        converter = SemanticKITTIConverter(
            artifact_address="geekyrakshit/point-cloud-voxelize/semantic-kitti:v1",
            sequence_id=args.sequence_id,
        )
        converter.save_data(
            output_dir=args.output_dir,
            lower_bound_index=args.lower_bound_index,
            upper_bound_index=args.upper_bound_index,
        )

    shutil.rmtree(args.output_dir)
