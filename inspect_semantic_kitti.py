import wandb
import argparse
from lidar_baselines.dataloader.semantic_kitti import SemanticKITTIInspector


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse flags for inspection of converted Semantic-KITTI sequences in numpy format"
    )
    parser.add_argument(
        "--sequence_id", type=str, required=True, help="Semantic-KITTI sequence ID"
    )
    parser.add_argument(
        "--artifact_address", type=str, required=True, help="Address of WandB artifact"
    )
    parser.add_argument(
        "--max_version",
        type=int,
        required=True,
        help="Latest version of the WandB artifact",
    )
    args = parser.parse_args()

    with wandb.init(
        project="point-cloud-voxelize",
        entity="geekyrakshit",
        name=f"inspect/semantic-kitti/{args.sequence_id}",
        job_type="inspection",
        tags=["semantic-kitti", "inspection", f"sequence-{args.sequence_id}"],
        config=args.__dict__,
    ):
        inspector = SemanticKITTIInspector(
            sequence_artifact_address=args.artifact_address,
            max_version=args.max_version,
        )
        inspector.inspect()
        print(inspector.means)
        print(inspector.stds)
