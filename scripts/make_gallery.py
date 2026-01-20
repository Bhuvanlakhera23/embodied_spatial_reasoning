import os
import sys
import json

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Episode Viewer</title>
<style>
body {{
  background: #0f0f0f;
  color: #eee;
  font-family: monospace;
  margin: 0;
  padding: 12px;
}}

#viewer {{
  display: flex;
  flex-direction: column;
  align-items: center;
}}

#main-img {{
  max-width: 90vw;
  max-height: 70vh;
  border: 2px solid #333;
}}

#info {{
  margin-top: 8px;
  font-size: 14px;
}}

#controls {{
  margin-top: 10px;
  display: flex;
  gap: 8px;
  align-items: center;
}}

input[type=range] {{
  width: 400px;
}}

button {{
  background: #222;
  color: #eee;
  border: 1px solid #444;
  padding: 6px 10px;
  cursor: pointer;
}}

button:hover {{
  background: #333;
}}

#grid {{
  margin-top: 24px;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 10px;
}}

.thumb {{
  border: 1px solid #333;
  cursor: pointer;
}}

.thumb img {{
  width: 100%;
}}

.meta {{
  font-size: 11px;
  color: #aaa;
  padding: 4px;
}}
</style>
</head>
<body>

<h2>Episode: {episode_id}</h2>
<p>Scene: {scene} | Frames: {num_frames}</p>

<div id="viewer">
  <img id="main-img" src="frames/000.png">
  <div id="info"></div>

  <div id="controls">
    <button onclick="prevFrame()">◀ Prev</button>
    <input type="range" id="slider" min="0" max="{max_idx}" value="0" step="1" oninput="updateFrame(this.value)">
    <button onclick="nextFrame()">Next ▶</button>
  </div>
</div>

<hr>

<h3>Frame Grid</h3>
<div id="grid">
{grid_cards}
</div>

<script>
const frames = {frame_data};

let current = 0;
const img = document.getElementById("main-img");
const info = document.getElementById("info");
const slider = document.getElementById("slider");

function updateFrame(i) {{
  current = parseInt(i);
  const f = frames[current];
  img.src = "frames/" + f.filename;
  info.innerHTML =
    "<b>#" + current + "</b> | " +
    "action: " + f.action + "<br>" +
    "pos: [" + f.position.map(x => x.toFixed(2)).join(", ") + "]";
  slider.value = current;
}}

function prevFrame() {{
  if (current > 0) updateFrame(current - 1);
}}

function nextFrame() {{
  if (current < frames.length - 1) updateFrame(current + 1);
}}

document.addEventListener("keydown", (e) => {{
  if (e.key === "ArrowLeft") prevFrame();
  if (e.key === "ArrowRight") nextFrame();
}});

// Init
updateFrame(0);
</script>

</body>
</html>
"""


def main(episode_dir):
    frames_dir = os.path.join(episode_dir, "frames")
    episode_json = os.path.join(episode_dir, "episode.json")

    if not os.path.exists(frames_dir):
        print("No frames directory found.")
        return

    if not os.path.exists(episode_json):
        print("No episode.json found.")
        return

    with open(episode_json, "r") as f:
        data = json.load(f)

    meta = data.get("meta", {})
    traj = data.get("trajectory", [])

    frame_data = []
    grid_cards = []

    for i, step in enumerate(traj):
        img_name = f"{i:03d}.png"
        img_path = f"frames/{img_name}"

        action = step.get("action", "unknown")
        pos = step.get("pose", {}).get("position", [0, 0, 0])

        frame_data.append({
            "filename": img_name,
            "action": action,
            "position": pos,
        })

        card = f"""
        <div class="thumb" onclick="updateFrame({i})">
            <img src="{img_path}">
            <div class="meta">
                #{i} | {action}
            </div>
        </div>
        """
        grid_cards.append(card)

    html = HTML_TEMPLATE.format(
        episode_id=meta.get("episode_id", "unknown"),
        scene=meta.get("scene", "unknown"),
        num_frames=len(traj),
        max_idx=len(traj) - 1,
        frame_data=json.dumps(frame_data),
        grid_cards="\n".join(grid_cards),
    )

    out_path = os.path.join(episode_dir, "index.html")
    with open(out_path, "w") as f:
        f.write(html)

    print(f"[GALLERY] Generated: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python make_gallery.py <episode_dir>")
        sys.exit(1)

    main(sys.argv[1])
