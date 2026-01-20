import habitat_sim
import numpy as np
from habitat_sim.utils.common import quat_from_angle_axis

CAMERA_HEIGHT = 0.8          # realistic torso camera
MIN_BRIGHTNESS = 8.0
MIN_STD = 6.0
MAX_SPAWN_TRIES = 80


def make_sim(scene_path):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False

    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "rgb"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [512, 512]

    # ✅ ONLY place camera height is applied
    sensor.position = [0.0, CAMERA_HEIGHT, 0.0]
    sensor.hfov = 90.0

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor]

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    return sim


def _random_yaw():
    angle = np.random.uniform(0, 2 * np.pi)
    return quat_from_angle_axis(angle, np.array([0, 1, 0]))


def _is_bad_frame(img):
    if img is None:
        return True

    mean = img.mean()
    std = img.std()

    if mean < MIN_BRIGHTNESS:
        return True

    if std < MIN_STD:
        return True

    # ceiling dominance check
    top = img[:40].mean()
    mid = img[200:300].mean()
    if abs(top - mid) < 1.2:
        return True

    return False


def find_valid_spawn(sim):
    agent = sim.get_agent(0)
    pf = sim.pathfinder

    for i in range(MAX_SPAWN_TRIES):
        pos = pf.get_random_navigable_point()

        # ✅ DO NOT add camera height here
        state = habitat_sim.AgentState()
        state.position = pos
        state.rotation = _random_yaw()

        agent.set_state(state)
        obs = sim.get_sensor_observations()
        rgb = obs.get("rgb")

        if not _is_bad_frame(rgb):
            print(f"Spawn accepted after {i+1} tries")
            return state

    raise RuntimeError("Failed to find a valid spawn after multiple attempts")


def reset_agent(sim, position=None):
    agent = sim.get_agent(0)

    if position is None:
        state = find_valid_spawn(sim)
        agent.set_state(state)
        return agent

    # Manual placement path (no height hacks)
    state = habitat_sim.AgentState()
    state.position = np.array(position, dtype=np.float32)
    state.rotation = _random_yaw()

    agent.set_state(state)
    return agent


def capture_frame(sim):
    obs = sim.get_sensor_observations()
    rgb = obs.get("rgb")

    if _is_bad_frame(rgb):
        return None

    return rgb


def get_pose(agent):
    state = agent.get_state()
    q = state.rotation
    return {
        "position": [float(x) for x in state.position],
        "rotation": [float(q.w), float(q.x), float(q.y), float(q.z)]
    }
