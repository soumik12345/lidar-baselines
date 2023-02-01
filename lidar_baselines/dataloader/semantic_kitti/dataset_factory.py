from typing import List
import tensorflow as tf


class SemanticKITTIDatasetFactory:
    def __init__(self) -> None:
        pass

    def parse_example(self, example):
        parsed_example = tf.io.parse_single_example(
            example,
            {
                "input_data": tf.io.FixedLenFeature([], tf.string),
                "lidar_mask": tf.io.FixedLenFeature([], tf.string),
                "segmentation_labels": tf.io.FixedLenFeature([], tf.string),
                "class_weight": tf.io.FixedLenFeature([], tf.string),
            },
        )
        input_data = tf.io.parse_tensor(
            parsed_example["input_data"], out_type=tf.float32
        )
        lidar_mask = tf.io.parse_tensor(parsed_example["lidar_mask"], out_type=tf.bool)
        segmentation_labels = tf.io.parse_tensor(
            parsed_example["segmentation_labels"], out_type=tf.int32
        )
        class_weight = tf.io.parse_tensor(
            parsed_example["class_weight"], out_type=tf.float32
        )
        return input_data, lidar_mask, segmentation_labels, class_weight

    def load_data(
        self,
        tfrecord_files: List[str],
        batch_size: int,
        shuffle_buffer_size: int = 10000,
    ):
        dataset = tf.data.TFRecordDataset(
            tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE
        )
        dataset = dataset.map(self.parse_example)
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        return dataset.prefetch(tf.data.AUTOTUNE)
