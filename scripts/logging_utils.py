import os
import json
from PIL import Image

def make_episode_dir(base="outputs/episodes"):
    os.makedirs(base, exist_ok=True)
    existing = sorted(d for d in os.listdir(base) if d.startswith("episode_"))
    num = len(existing) + 1
    ep_id = f"episode_{num:04d}"
    path = os.path.join(base, ep_id)
    os.makedirs(os.path.join(path, "frames"), exist_ok=True)
    return ep_id, path

def save_frame(img, path):
    Image.fromarray(img).save(path)

def save_episode_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
