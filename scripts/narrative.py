from scipy.spatial.transform import Rotation as R
import numpy as np

def inspect_yaw_only(position, yaw_angles, pitch=0.0):
    """
    Narrative Act A:
    Fixed pitch, sweeping yaw.
    """
    poses = []

    for yaw in yaw_angles:
        poses.append({
            "position": np.array(position, dtype=float),
            "rotation": R.from_euler(
                "yx", [yaw, pitch], degrees=True
            ).as_quat(),
            "act": "inspect_yaw"
        })

    return poses
