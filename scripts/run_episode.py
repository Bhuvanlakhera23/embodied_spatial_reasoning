'''
python scripts/run_episode.py --scene apartment_1.glb
python scripts/run_episode.py --scene van-gogh-room.glb
python scripts/run_episode.py --scene skokloster-castle.glb

'''
import argparse
import os
from scripts.cinematic_episode import run

SCENE_DIR = "habitat_data/scene_datasets/habitat-test-scenes"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        required=True,
        help="Scene file in habitat_data (e.g. skokloster-castle.glb)"
    )
    args = parser.parse_args()

    scene_path = os.path.join(SCENE_DIR, args.scene)

    if not os.path.exists(scene_path):
        print("\n❌ ERROR: Scene not found\n")
        print("You asked for:", args.scene)
        print("\nAvailable scenes:\n")
        for f in sorted(os.listdir(SCENE_DIR)):
            if f.endswith(".glb"):
                print("  -", f)
        print()
        exit(1)

    run(args.scene)
