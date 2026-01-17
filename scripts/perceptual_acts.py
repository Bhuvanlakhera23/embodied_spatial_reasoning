# scripts/perceptual_acts.py
import numpy as np
from PIL import Image
import os
import magnum as mn

from habitat_sim.utils.common import quat_from_angle_axis, quat_rotate_vector

def save_rgb(obs, out_dir, idx):
    rgb = obs["rgb"]
    Image.fromarray(rgb).save(os.path.join(out_dir, f"{idx:03d}.png"))

def rotate_in_place(agent, sim, out_dir, start_idx, degrees=90, steps=18):
    """
    Rotate camera in place using Magnum quaternion math.
    """
    idx = start_idx
    delta = np.deg2rad(degrees / steps)

    for _ in range(steps):
        state = agent.get_state()

        # Construct delta rotation (about Y axis)
        delta_q = quat_from_angle_axis(delta, np.array([0.0, 1.0, 0.0]))

        # Magnum quaternion composition
        state.rotation = delta_q * state.rotation

        agent.set_state(state)

        obs = sim.get_sensor_observations()
        save_rgb(obs, out_dir, idx)
        idx += 1

    return idx

def move_forward(agent, sim, out_dir, start_idx, distance=0.2, steps=10):
    """
    Move forward in the agent's local -Z direction.
    """
    idx = start_idx

    for _ in range(steps):
        state = agent.get_state()

        # Habitat forward direction
        forward_local = np.array([0.0, 0.0, -1.0])

        # Rotate using Habitat helper (CRITICAL)
        forward_world = quat_rotate_vector(
            state.rotation, forward_local
        )

        state.position += forward_world * distance
        agent.set_state(state)

        obs = sim.get_sensor_observations()
        save_rgb(obs, out_dir, idx)
        idx += 1

    return idx

def pause_and_observe(agent, sim, out_dir, start_idx, steps=5):
    """
    Stay still and observe.
    """
    idx = start_idx
    for _ in range(steps):
        obs = sim.get_sensor_observations()
        save_rgb(obs, out_dir, idx)
        idx += 1

    return idx
