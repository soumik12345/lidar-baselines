import argparse

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
        "--artifact_address", type=str, required=True, help="Lower bound index"
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
    converter = SemanticKITTIConverter(
        artifact_address=args.artifact_address, sequence_id=args.sequence_id
    )
    converter.save_sequence(
        split_size=256,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        output_dir=args.output_dir,
        log_visualizations=args.visualize,
    )
