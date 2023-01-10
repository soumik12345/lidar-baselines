import os
from glob import glob

import wandb
import numpy as np
from tqdm.auto import tqdm

from .utils import RunningStd


class SemanticKITTIInspector:
    def __init__(self, sequence_artifact_address: str) -> None:
        self.sequence_artifact_address = sequence_artifact_address
        self.numpy_files_path = []
        self.means = {
            "lidar_scan": {"x": [], "y": [], "z": []},
            "intensity": [],
            "depth": [],
        }
        self.stds = {
            "lidar_scan": {"x": RunningStd(), "y": RunningStd(), "z": RunningStd()},
            "intensity": RunningStd(),
            "depth": RunningStd(),
        }
        self.fetch_sequence_artifacts()

    def __len__(self):
        return len(self.numpy_files_path)

    def fetch_sequence_artifacts(self):
        sequence_split_artifact = (
            wandb.Api().artifact(self.sequence_artifact_address, type="numpy-dataset")
            if wandb.run is None
            else wandb.use_artifact(
                self.sequence_artifact_address, type="numpy-dataset"
            )
        )
        artifact_dir = sequence_split_artifact.download()
        self.numpy_files_path = glob(os.path.join(artifact_dir, "*", "*.npy"))

    def compute_means(self, lidar_data=None, lidar_mask=None):
        if lidar_data is not None and lidar_mask is not None:
            self.means["lidar_scan"]["x"].append(lidar_data[:, :, 0][lidar_mask].mean())
            self.means["lidar_scan"]["y"].append(lidar_data[:, :, 1][lidar_mask].mean())
            self.means["lidar_scan"]["z"].append(lidar_data[:, :, 2][lidar_mask].mean())
            self.means["intensity"].append(lidar_data[:, :, 3][lidar_mask].mean())
            self.means["depth"].append(lidar_data[:, :, 4][lidar_mask].mean())
        else:
            self.means["lidar_scan"]["x"] = np.mean(self.means["lidar_scan"]["x"])
            self.means["lidar_scan"]["y"] = np.mean(self.means["lidar_scan"]["y"])
            self.means["lidar_scan"]["z"] = np.mean(self.means["lidar_scan"]["z"])
            self.means["intensity"] = np.mean(self.means["intensity"])
            self.means["depth"] = np.mean(self.means["depth"])
            if wandb.run is not None:
                wandb.log(
                    {
                        "mean/lidar_scan_x": float(self.means["lidar_scan"]["x"]),
                        "mean/lidar_scan_y": float(self.means["lidar_scan"]["y"]),
                        "mean/lidar_scan_z": float(self.means["lidar_scan"]["z"]),
                        "mean/intensity": float(self.means["intensity"]),
                        "mean/depth": float(self.means["depth"]),
                    }
                )

    def compute_stds(self, lidar_data=None, lidar_mask=None):
        if lidar_data is not None and lidar_mask is not None:
            self.stds["lidar_scan"]["x"].include(lidar_data[:, :, 0][lidar_mask])
            self.stds["lidar_scan"]["y"].include(lidar_data[:, :, 1][lidar_mask])
            self.stds["lidar_scan"]["z"].include(lidar_data[:, :, 2][lidar_mask])
            self.stds["intensity"].include(lidar_data[:, :, 3][lidar_mask])
            self.stds["depth"].include(lidar_data[:, :, 4][lidar_mask])
        else:
            self.stds["lidar_scan"]["x"] = self.stds["lidar_scan"]["x"].std
            self.stds["lidar_scan"]["y"] = self.stds["lidar_scan"]["y"].std
            self.stds["lidar_scan"]["z"] = self.stds["lidar_scan"]["z"].std
            self.stds["intensity"] = self.stds["intensity"].std
            self.stds["depth"] = self.stds["depth"].std
            if wandb.run is not None:
                wandb.log(
                    {
                        "std/lidar_scan_x": float(self.stds["lidar_scan"]["x"]),
                        "std/lidar_scan_y": float(self.stds["lidar_scan"]["y"]),
                        "std/lidar_scan_z": float(self.stds["lidar_scan"]["z"]),
                        "std/intensity": float(self.stds["intensity"]),
                        "std/depth": float(self.stds["depth"]),
                    }
                )

    def inspect(self):
        for numpy_file in tqdm(self.numpy_files_path):
            lidar_data = np.load(numpy_file)
            lidar_mask = lidar_data[:, :, 4] > 0
            self.compute_means(lidar_data, lidar_mask)
            self.compute_stds(lidar_data, lidar_mask)
        self.compute_means()
        self.compute_stds()
