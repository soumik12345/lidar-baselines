import wandb
import argparse
from lidar_baselines.dataloader.semantic_kitti import SemanticKITTIConverter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse flags for conversion of Semantic-KITTI sequences to numpy format"
    )
    parser.add_argument(
        "--sequence_id", type=str, required=True, help="Semantic-KITTI sequence ID"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="semantic_kitti_data",
        help="Output directory for dumping the converted data",
    )
    args = parser.parse_args()

    with wandb.init(
        project="point-cloud-voxelize",
        entity="geekyrakshit",
        name=f"numpy-conversion/semantic-kitti/{args.sequence_id}",
        job_type="numpy-conversion",
    ):
        converter = SemanticKITTIConverter(
            artifact_address="geekyrakshit/point-cloud-voxelize/semantic-kitti:v1",
            sequence_id=args.sequence_id,
        )
        converter.save_data(output_dir=args.output_dir)
