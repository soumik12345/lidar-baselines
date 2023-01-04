from typing import Dict


def get_label_map() -> Dict:
    """
    Refernce:

    (1) https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml#L109-L143
    (2) https://github.com/ika-rwth-aachen/PCLSegmentation/blob/main/dataset_convert/semantic_kitti_sequence.py#L71-L106
    """
    return {
        0: 0,  # "unlabeled"
        1: 0,  # "outlier" mapped to "unlabeled" --------------------------mapped
        10: 1,  # "car"
        11: 2,  # "bicycle"
        13: 5,  # "bus" mapped to "other-vehicle" --------------------------mapped
        15: 3,  # "motorcycle"
        16: 5,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
        18: 4,  # "truck"
        20: 5,  # "other-vehicle"
        30: 6,  # "person"
        31: 7,  # "bicyclist"
        32: 8,  # "motorcyclist"
        40: 9,  # "road"
        44: 10,  # "parking"
        48: 11,  # "sidewalk"
        49: 12,  # "other-ground"
        50: 13,  # "building"
        51: 14,  # "fence"
        52: 0,  # "other-structure" mapped to "unlabeled" ------------------mapped
        60: 9,  # "lane-marking" to "road" ---------------------------------mapped
        70: 15,  # "vegetation"
        71: 16,  # "trunk"
        72: 17,  # "terrain"
        80: 18,  # "pole"
        81: 19,  # "traffic-sign"
        99: 0,  # "other-object" to "unlabeled" ----------------------------mapped
        252: 1,  # "moving-car" to "car" ------------------------------------mapped
        253: 7,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
        254: 6,  # "moving-person" to "person" ------------------------------mapped
        255: 8,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
        256: 5,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
        257: 5,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
        258: 4,  # "moving-truck" to "truck" --------------------------------mapped
        259: 5,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
    }


def get_label_to_name() -> Dict:
    """
    Refernce:

    (1) https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml#L109-L143
    (2) https://github.com/ika-rwth-aachen/PCLSegmentation/blob/main/dataset_convert/semantic_kitti_sequence.py#L71-L106
    """
    return {
        0: "unlabeled",
        1: "outlier",
        10: "car",
        11: "bicycle",
        13: "bus",
        15: "motorcycle",
        16: "on-rails",
        18: "truck",
        20: "other-vehicle",
        30: "person",
        31: "bicyclist",
        32: "motorcyclist",
        40: "road",
        44: "parking",
        48: "sidewalk",
        49: "other-ground",
        50: "building",
        51: "fence",
        52: "other-structure",
        60: "lane-marking",
        70: "vegetation",
        71: "trunk",
        72: "terrain",
        80: "pole",
        81: "traffic-sign",
        99: "other-object",
        252: "moving-car",
        253: "moving-bicyclist",
        254: "moving-person",
        255: "moving-motorcyclist",
        256: "moving-on-rails",
        257: "moving-bus",
        258: "moving-truck",
        259: "moving-other",
    }


def get_color_map() -> Dict:
    """
    Reference:

    (1) https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml#L37-L71
    """
    return {
        0: [0, 0, 0][::-1],
        1: [0, 0, 255][::-1],
        10: [245, 150, 100][::-1],
        11: [245, 230, 100][::-1],
        13: [250, 80, 100][::-1],
        15: [150, 60, 30][::-1],
        16: [255, 0, 0][::-1],
        18: [180, 30, 80][::-1],
        20: [255, 0, 0][::-1],
        30: [30, 30, 255][::-1],
        31: [200, 40, 255][::-1],
        32: [90, 30, 150][::-1],
        40: [255, 0, 255][::-1],
        44: [255, 150, 255][::-1],
        48: [75, 0, 75][::-1],
        49: [75, 0, 175][::-1],
        50: [0, 200, 255][::-1],
        51: [50, 120, 255][::-1],
        52: [0, 150, 255][::-1],
        60: [170, 255, 150][::-1],
        70: [0, 175, 0][::-1],
        71: [0, 60, 135][::-1],
        72: [80, 240, 150][::-1],
        80: [150, 240, 255][::-1],
        81: [0, 0, 255][::-1],
        99: [255, 255, 50][::-1],
        252: [245, 150, 100][::-1],
        256: [255, 0, 0][::-1],
        253: [200, 40, 255][::-1],
        254: [30, 30, 255][::-1],
        255: [90, 30, 150][::-1],
        257: [250, 80, 100][::-1],
        258: [180, 30, 80][::-1],
        259: [255, 0, 0][::-1],
    }


def get_segmentation_classes() -> Dict:
    return {
        0: "None",
        1: "car",
        2: "bicycle",
        3: "motorcycle",
        4: "truck",
        5: "other-vehicle",
        6: "person",
        7: "bicyclist",
        8: "motorcyclist",
        9: "road",
        10: "parking",
        11: "sidewalk",
        12: "other-ground",
        13: "building",
        14: "fence",
        15: "vegetation",
        16: "trunk",
        17: "terrain",
        18: "pole",
        19: "traffic-sign",
    }


def get_segmentation_colors() -> Dict:
    return {
        0: [0, 0, 0][::-1],
        1: [245, 150, 100][::-1],
        2: [245, 230, 100][::-1],
        3: [150, 60, 30][::-1],
        4: [180, 30, 80][::-1],
        5: [255, 0, 0][::-1],
        6: [30, 30, 255][::-1],
        7: [200, 40, 255][::-1],
        8: [90, 30, 150][::-1],
        9: [255, 0, 255][::-1],
        10: [255, 150, 255][::-1],
        11: [75, 0, 75][::-1],
        12: [75, 0, 175][::-1],
        13: [0, 200, 255][::-1],
        14: [50, 120, 255][::-1],
        15: [0, 175, 0][::-1],
        16: [0, 60, 135][::-1],
        17: [80, 240, 150][::-1],
        18: [150, 240, 255][::-1],
        19: [0, 0, 255][::-1],
    }
