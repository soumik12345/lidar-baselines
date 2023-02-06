import os
import argparse

import wandb

from lidar_baselines.dataloader.semantic_kitti import SemanticKITTIConverter


def get_args():
    parser = argparse.ArgumentParser(
        description="Parse flags for conversion of Semantic-KITTI sequences to numpy format"
    )
    parser.add_argument(
        "--wandb_project", type=str, required=True, help="Weights & Biases Project Name"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="Weights & Biases Entity Name"
    )
    parser.add_argument(
        "--sequence_id", type=str, required=True, help="Semantic-KITTI sequence ID"
    )
    parser.add_argument(
        "--num_point_clouds",
        type=int,
        required=True,
        help="Number of point clouds in the sequence",
    )
    parser.add_argument(
        "--artifact_address",
        type=str,
        required=True,
        help="Artifact address of the Semantic KITTI Dataset",
    )
    parser.add_argument(
        "--split_size", type=int, default=256, help="Semantic-KITTI sequence ID"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="semantic_kitti_data",
        help="Output directory for dumping the converted data",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Produce visualizations on WandB or not",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for idx in range(0, args.num_point_clouds, args.split_size):
        converter = SemanticKITTIConverter(
            artifact_address=args.artifact_address, sequence_id=args.sequence_id
        )
        upper_bound_index = (
            idx + args.split_size
            if idx + args.split_size < len(converter)
            else len(converter)
        )
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"Semantic-KITTI/{converter.sequence_id}/{idx}-{upper_bound_index}",
            job_type="numpy-conversion",
            tags=[
                "semantic-kitti",
                "numpy-conversion",
                f"sequence-{converter.sequence_id}",
                f"split-{idx}-{upper_bound_index}",
            ],
        )
        converter.save_data(
            output_dir=args.output_dir,
            lower_bound_index=idx,
            upper_bound_index=upper_bound_index,
            log_visualizations=args.visualize,
        )
        wandb.finish()
