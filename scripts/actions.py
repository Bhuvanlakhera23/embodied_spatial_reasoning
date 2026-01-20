import numpy as np
import habitat_sim

def _apply(agent, sim, translation=None, rotation=None):
    state = agent.get_state()

    if rotation is not None:
        q = habitat_sim.utils.quat_from_angle_axis(
            np.deg2rad(rotation), np.array([0, 1, 0])
        )
        state.rotation = q * state.rotation

    if translation is not None:
        forward = np.array([0, 0, -1])
        dir_vec = habitat_sim.utils.quat_rotate_vector(
            state.rotation, forward
        )
        state.position = state.position + dir_vec * translation

    agent.set_state(state)

def rotate(agent, sim, degrees):
    _apply(agent, sim, rotation=degrees)

def move_forward(agent, sim, meters):
    _apply(agent, sim, translation=meters)
