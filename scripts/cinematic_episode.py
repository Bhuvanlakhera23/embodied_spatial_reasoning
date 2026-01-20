import os
from scripts.embodiment import make_sim, reset_agent, capture_frame, get_pose
from scripts.actions import rotate, move_forward
from scripts.view_memory import ViewMemory
from scripts.logging_utils import make_episode_dir, save_frame, save_episode_json
from scripts.make_gallery import main as make_gallery   # 🔥 NEW

SCENE_DIR = "habitat_data/scene_datasets/habitat-test-scenes"
MAX_RETRIES = 3


def run(scene_file):
    scene_path = os.path.join(SCENE_DIR, scene_file)

    sim = make_sim(scene_path)
    agent = reset_agent(sim)

    memory = ViewMemory()
    ep_id, ep_path = make_episode_dir()
    frames_dir = os.path.join(ep_path, "frames")

    def record(action):
        frame = capture_frame(sim)

        if frame is None:
            print(f"[SKIP] Bad frame after action: {action}")
            return False

        pose = get_pose(agent)
        vid = memory.add_view(frame, pose, action)
        save_frame(frame, os.path.join(frames_dir, f"{vid}.png"))
        return True

    def do_step(action_fn, action_name, *args):
        for attempt in range(MAX_RETRIES):
            action_fn(agent, sim, *args)

            if record(action_name):
                return True

            print(f"[RETRY] {action_name} attempt {attempt+1}")

        print(f"[DROP] {action_name} failed after {MAX_RETRIES} retries")
        return False

    # -------------------------
    # Spawn frame
    # -------------------------
    for attempt in range(MAX_RETRIES):
        if record("spawn"):
            break
        print(f"[RETRY] spawn attempt {attempt+1}")
    else:
        print("[FATAL] Could not capture valid spawn frame")
        sim.close()
        return None

    # -------------------------
    # Phase 1: rotate 360
    # -------------------------
    for _ in range(12):
        do_step(rotate, "rotate+30", 30)

    # -------------------------
    # Phase 2: forward × 3
    # -------------------------
    for _ in range(3):
        do_step(move_forward, "move_forward", 0.75)

    # -------------------------
    # Phase 3: left 90
    # -------------------------
    for _ in range(3):
        do_step(rotate, "rotate+30", 30)

    # -------------------------
    # Phase 4: forward × 3
    # -------------------------
    for _ in range(3):
        do_step(move_forward, "move_forward", 0.75)

    # -------------------------
    # Phase 5: rotate 360
    # -------------------------
    for _ in range(12):
        do_step(rotate, "rotate+30", 30)

    sim.close()

    meta = {
        "episode_id": ep_id,
        "scene": scene_file,
        "num_frames": len(memory.views),
        "pattern": "micro_tour_v1"
    }

    save_episode_json(
        os.path.join(ep_path, "episode.json"),
        {"meta": meta, "trajectory": memory.export_json()}
    )

    # 🔥 AUTO-GENERATE HTML
    make_gallery(ep_path)

    print(f"Episode complete: {ep_id}")
    print(f"Frames: {len(memory.views)}")
    print(f"Gallery: {ep_path}/index.html")

    return ep_id
