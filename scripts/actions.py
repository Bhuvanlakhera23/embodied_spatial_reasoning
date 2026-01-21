import numpy as np
from habitat_sim.utils.common import quat_from_angle_axis

STEP_SIZE = 0.25  # meters

def rotate(agent, sim, angle_deg):
    state = agent.get_state()

    # Y-axis rotation (yaw)
    delta = quat_from_angle_axis(
        np.deg2rad(angle_deg),
        np.array([0, 1, 0], dtype=np.float32)
    )

    state.rotation = delta * state.rotation
    agent.set_state(state)

def move_forward(agent, sim, distance):
    state = agent.get_state()
    rot = state.rotation

    # Forward vector in Habitat coords
    forward = np.array([0, 0, -1], dtype=np.float32)

    # Quaternion rotate
    q = np.array([rot.w, rot.x, rot.y, rot.z], dtype=np.float32)
    t = 2.0 * np.cross(q[1:], forward)
    rotated = forward + q[0] * t + np.cross(q[1:], t)

    target = state.position + rotated * distance

    pf = sim.pathfinder

    if pf.is_navigable(target):
        state.position = target
        agent.set_state(state)
        return True

    # 🔥 Fallback: small lateral wiggle search
    for angle in [30, -30, 60, -60]:
        delta = quat_from_angle_axis(
            np.deg2rad(angle),
            np.array([0, 1, 0], dtype=np.float32)
        )

        tmp_rot = delta * rot
        q = np.array([tmp_rot.w, tmp_rot.x, tmp_rot.y, tmp_rot.z], dtype=np.float32)

        t = 2.0 * np.cross(q[1:], forward)
        rotated = forward + q[0] * t + np.cross(q[1:], t)

        candidate = state.position + rotated * distance

        if pf.is_navigable(candidate):
            state.rotation = tmp_rot
            state.position = candidate
            agent.set_state(state)
            print(f"[RECOVER] move_forward sidestep {angle}°")
            return True

    print("[BLOCKED] move_forward blocked by collision")
    return False

