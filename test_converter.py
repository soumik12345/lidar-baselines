import wandb
from lidar_baselines.dataloader.semantic_kitti import SemanticKITTIConverter


with wandb.init(project="point-cloud-voxelize", entity="geekyrakshit", job_type="test"):
    converter = SemanticKITTIConverter(
        artifact_address="geekyrakshit/point-cloud-voxelize/semantic-kitti:v0",
        sequence_id="00",
    )
    converter.save_data(output_dir="semantic_kitti_data")