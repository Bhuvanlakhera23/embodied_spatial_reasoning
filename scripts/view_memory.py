# scripts/view_memory.py

import uuid
import numpy as np


class ViewMemory:
    """
    Raw episodic memory of agent views.
    Stores sequence of observations with pose + action.
    """

    def __init__(self):
        self.views = []

    def add_view(self, frame, pose, action, frame_path=None):
        vid = f"{len(self.views):03d}"
        self.views.append({
            "id": vid,
            "frame": frame,
            "frame_path": frame_path,
            "pose": pose,
            "action": action
        })
        return vid

    def export_json(self):
        return [
            {
                "id": v["id"],
                "action": v["action"],
                "pose": v["pose"],
                "frame_path": v.get("frame_path"),
                "objects": v.get("objects"),
                "scene_type": v.get("scene_type")
            }
            for v in self.views
        ]


class SpatialMemory(ViewMemory):
    """
    Extends ViewMemory with a topological spatial graph.
    Each view becomes a node; actions form edges.
    """

    def __init__(self):
        super().__init__()
        self.graph = {}     # node_id -> metadata
        self.edges = []     # (from, to, action)
        self.last_node = None

    def add_view(self, frame, pose, action, frame_path=None):
        vid = super().add_view(frame, pose, action, frame_path)

        # Create node
        self.graph[vid] = {
            "pose": pose,
            "frame_path": frame_path,
            "objects": [],
            "scene_type": None,
            "visited": True
        }

        # Create edge
        if self.last_node is not None:
            self.edges.append((self.last_node, vid, action))

        self.last_node = vid
        return vid

    def update_semantics(self, node_id, objects=None, scene_type=None):
        if node_id not in self.graph:
            return
        if objects is not None:
            self.graph[node_id]["objects"] = objects
        if scene_type is not None:
            self.graph[node_id]["scene_type"] = scene_type

    def get_recent_node(self):
        return self.last_node

    def unvisited_nodes(self):
        return [k for k, v in self.graph.items() if not v.get("visited", False)]

    def summary(self):
        return {
            "num_nodes": len(self.graph),
            "num_edges": len(self.edges)
        }
