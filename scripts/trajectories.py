import numpy as np
from scipy.spatial.transform import Rotation as R

def straight_line(start, direction, step_size, num_steps):
    """Move in a straight line"""
    poses = []
    pos = np.array(start, dtype=float)

    for i in range(num_steps):
        poses.append({
            "position": pos.copy(),
            "rotation": R.from_euler("y", 0, degrees=True).as_quat()
        })
        pos += step_size * np.array(direction)

    return poses


def yaw_sweep(position, yaw_angles_deg):
    """Rotate camera in place"""
    poses = []
    for yaw in yaw_angles_deg:
        poses.append({
            "position": np.array(position),
            "rotation": R.from_euler("y", yaw, degrees=True).as_quat()
        })
    return poses
