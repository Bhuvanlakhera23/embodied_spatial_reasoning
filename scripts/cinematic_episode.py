import os
import json
import numpy as np

from scripts.embodiment import make_sim, reset_agent, capture_frame, get_pose
from scripts.actions import rotate, move_forward
from scripts.view_memory import SpatialMemory
from scripts.logging_utils import make_episode_dir, save_frame, save_episode_json
from scripts.make_gallery import main as make_gallery
from scripts.vlm_reasoner import VLMReasoner

SCENE_DIR = "habitat_data/scene_datasets/habitat-test-scenes"
MAX_RETRIES = 3


def run(scene_file, question="Find the bathroom"):
    scene_path = os.path.join(SCENE_DIR, scene_file)

    sim = make_sim(scene_path)
    agent = reset_agent(sim)

    # Qwen-based reasoner
    reasoner = VLMReasoner()
    memory = SpatialMemory()

    ep_id, ep_path = make_episode_dir()
    frames_dir = os.path.join(ep_path, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    last_vlm_result = None

    # -------------------------
    # Logging helper
    # -------------------------
    def record(action):
        frame = capture_frame(sim)
        pose = get_pose(agent)

        if frame is None:
            print(f"[WARN] Bad frame after action: {action}")
            memory.add_view(None, pose, action)
            return False

        vid = memory.add_view(frame, pose, action)
        path = os.path.join(frames_dir, f"{vid}.png")
        save_frame(frame, path)
        memory.views[-1]["frame_path"] = path
        return True

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
    # Control loop
    # -------------------------
    MAX_STEPS = 25
    VLM_INTERVAL = 5
    CONTEXT_FRAMES = 6

    for step in range(MAX_STEPS):
        print(f"\n[STEP {step}]")

        use_vlm = (step % VLM_INTERVAL == 0 and step > 0)

        # ==========================================
        # VLM CONTROL PHASE
        # ==========================================
        if use_vlm:
            print("[VLM] Reasoning...")

            def is_informative(v):
                return v.get("frame_path") is not None

            recent_views = [v for v in memory.views if is_informative(v)]
            last_views = recent_views[-CONTEXT_FRAMES:]

            if len(last_views) >= 2:
                frame_paths = [v["frame_path"] for v in last_views]
                memory_summary = memory.export_json()[-CONTEXT_FRAMES:]

                result = reasoner.reason(
                    question=question,
                    frame_paths=frame_paths,
                    memory_summary=memory_summary
                )

                # Update spatial memory with semantics
                node_id = memory.get_recent_node()
                memory.update_semantics(
                    node_id,
                    objects=result.get("visible_objects", []),
                    scene_type=result.get("scene_type_guess")
)



                memory.views[-1]["objects"] = result["visible_objects"]
                memory.views[-1]["scene_type"] = result["scene_type_guess"]


                last_vlm_result = result

                print("[VLM] Thought:", result.get("reasoning"))
                if result["scene_type_guess"] == "bathroom" or "toilet" in result["visible_objects"]:
                    print("[SUCCESS] Bathroom found!")
                    break

                print("[VLM] Actions:", result.get("next_action_plan"))

                # 🔥 EXECUTE SYMBOLIC ACTIONS
                for act in result.get("next_action_plan", []):
                    a = act.get("action")

                    if a == "move_forward":
                        dist = act.get("distance", 0.6)
                        ok = move_forward(agent, sim, dist)
                        record("move_forward" if ok else "blocked")

                    elif a == "rotate":
                        ang = act.get("angle_deg", 30)
                        rotate(agent, sim, ang)
                        record(f"rotate{ang:+d}")

                    elif a == "scan":
                        rotate(agent, sim, 30)
                        record("scan")

                    elif a == "stop":
                        print("[VLM] Stop requested.")
                        sim.close()
                        break

                continue  # Skip reactive fallback on VLM step

            else:
                print("[VLM] Not enough informative frames, falling back.")

        # ==========================================
        # FALLBACK REACTIVE CONTROL
        # ==========================================
        moved = move_forward(agent, sim, 0.6)
        if moved:
            record("move_forward")
        else:
            angle = 90
            print(f"[RECOVER] rotate {angle}°")
            rotate(agent, sim, angle)
            record(f"rotate{angle:+d}")

    sim.close()

    # -------------------------
    # Save Episode JSON
    # -------------------------
    meta = {
        "episode_id": ep_id,
        "scene": scene_file,
        "num_frames": len(memory.views),
        "pattern": "vlm_control_v1",
        "question": question
    }

    episode_json_path = os.path.join(ep_path, "episode.json")
    save_episode_json(
        episode_json_path,
        {"meta": meta, "trajectory": memory.export_json()}
    )

    # -------------------------
    # Save VLM reasoning
    # -------------------------
    if last_vlm_result is not None:
        reasoning_json_path = os.path.join(ep_path, "reasoning.json")
        with open(reasoning_json_path, "w") as f:
            json.dump(last_vlm_result, f, indent=2)
        print("[VLM] Final reasoning saved")

    # -------------------------
    # Auto-generate HTML gallery
    # -------------------------
    make_gallery(ep_path)

    print(f"Episode complete: {ep_id}")
    print(f"Frames: {len(memory.views)}")
    print(f"Gallery: {ep_path}/index.html")
    print(f"Reasoning: {ep_path}/reasoning.json")

    return ep_id