import wandb
import argparse
from lidar_baselines.dataloader.semantic_kitti import SemanticKITTIInspector


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse flags for inspection of converted Semantic-KITTI sequences in numpy format"
    )
    parser.add_argument(
        "--wandb_project", type=str, required=True, help="Weights & Biases Project"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        required=False,
        help="Weights & Biases Entity",
    )
    parser.add_argument(
        "--sequence_id", type=str, required=True, help="Semantic-KITTI sequence ID"
    )
    args = parser.parse_args()

    inspection_runs = wandb.Api(timeout=100).runs(
        path=f"{args.wandb_entity}/{args.wandb_project}",
        filters={
            "jobType": "numpy-conversion",
            "display_name": {"$regex": f"^Semantic-KITTI/{args.sequence_id}/*"},
        },
        order="+display_name",
    )

    artifact_addresses = []
    for run in inspection_runs:
        for artifact in iter(run.logged_artifacts()):
            if artifact.type == "numpy-dataset":
                artifact_addresses.append(
                    f"{args.wandb_entity}/{args.wandb_project}/{artifact.name}"
                )

    with wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"Inspect/Semantic-Kitti/{args.sequence_id}",
        job_type="inspection",
        tags=["semantic-kitti", "inspection", f"sequence-{args.sequence_id}"],
        config=args.__dict__,
    ):
        inspection_runs = wandb.Api().runs(
            path=f"{args.wandb_entity}/{args.wandb_project}",
            filters={
                "jobType": "numpy-conversion",
                "display_name": {"$regex": f"^Semantic-KITTI/{args.sequence_id}/*"},
            },
            order="+display_name",
        )
        inspector = SemanticKITTIInspector(
            sequence_artifact_addresses=artifact_addresses
        )
        inspector.inspect()
        print(inspector.means)
        print(inspector.stds)
