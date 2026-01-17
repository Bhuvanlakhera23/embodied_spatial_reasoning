# scripts/camera_walk.py
import habitat_sim
import os
import getpass
from perceptual_acts import (
    rotate_in_place,
    move_forward,
    pause_and_observe
)

SCENE = f"/home/{getpass.getuser()}/Desktop/plaksha/vlm/habitat_cov/habitat_data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
OUT_DIR = "outputs/frames"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Simulator ---
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = SCENE

sensor = habitat_sim.CameraSensorSpec()
sensor.uuid = "rgb"
sensor.sensor_type = habitat_sim.SensorType.COLOR
sensor.resolution = [512, 512]
sensor.position = [0.0, 1.5, 0.0]
sensor.hfov = 90.0

agent_cfg = habitat_sim.AgentConfiguration()
agent_cfg.sensor_specifications = [sensor]

cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
sim = habitat_sim.Simulator(cfg)
agent = sim.get_agent(0)

# --- Episode ---
idx = 0
idx = pause_and_observe(agent, sim, OUT_DIR, idx)
idx = rotate_in_place(agent, sim, OUT_DIR, idx, degrees=180)
idx = move_forward(agent, sim, OUT_DIR, idx)
idx = pause_and_observe(agent, sim, OUT_DIR, idx)

sim.close()
