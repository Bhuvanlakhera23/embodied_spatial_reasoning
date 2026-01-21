import json
import os
import re
import numpy as np
from typing import List, Dict, Any

from scripts.qwen_backend import QwenVLMBackend

try:
    import cv2
except ImportError:
    cv2 = None
    print("[VLM] WARNING: OpenCV not installed. Visual heuristics disabled.")


# ==========================================================
# Symbolic action space (CONTROL CONTRACT)
# ==========================================================
ALLOWED_ACTIONS = {"move_forward", "rotate", "scan", "stop"}


class VLMReasoner:
    """
    Perception → Scene Abstraction → Deterministic Action Synthesis

    Qwen2-VL is used ONLY for perception.
    All control is synthesized deterministically for safety and reproducibility.
    """

    def __init__(self):
        print("[VLM] Initializing Qwen2-VL backend...")
        self.backend = QwenVLMBackend()

    # ==========================================================
    # PUBLIC API
    # ==========================================================
    def reason(
        self,
        question: str,
        frame_paths: List[str],
        memory_summary: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        frame_paths = [p for p in frame_paths if p and os.path.exists(p)]
        memory_summary = memory_summary[-6:] if memory_summary else []

        if not frame_paths:
            return self._offline_reasoning(question, frame_paths, memory_summary)

        # ---------- PERCEPTION ----------
        try:
            perception = self._run_perception(frame_paths)
        except Exception as e:
            print("[VLM] Perception failure:", e)
            return self._offline_reasoning(question, frame_paths, memory_summary)

        # ---------- SCENE ABSTRACTION ----------
        scene_state = self._build_scene_state(perception)

        # ---------- ACTION SYNTHESIS ----------
        next_actions = self._synthesize_actions(scene_state, memory_summary)

        # ---------- FINAL OUTPUT ----------
        result = {
            "reasoning": scene_state["summary"],
            "visible_objects": scene_state["objects"],
            "scene_type_guess": scene_state["scene_type"],
            "uncertainties": scene_state["uncertainties"],
            "next_actions": next_actions,
            "justification": scene_state["justification"],
            "confidence": scene_state["confidence"],
            "frames_used": frame_paths
        }

        return self._sanitize_output(result)

    # ==========================================================
    # PERCEPTION (QWEN IS USED HERE ONLY)
    # ==========================================================
    def _run_perception(self, frame_paths: List[str]) -> Dict[str, Any]:
        prompt = """
You are a perception system for a mobile robot.

Return JSON only with this schema:
{
  "objects": { "door": {...}, "wall": {...}, ... },
  "scene": { "room": {"type": "..."} },
  "navigational_affordances": ["corridor", "open_space", "blocked", "door"]
}

Rules:
- Only describe what is visible
- Do NOT speculate
- No extra text
""".strip()

        raw = self.backend.run(prompt, frame_paths)
        parsed = self._safe_json_parse(raw)

        if not isinstance(parsed, dict):
            raise ValueError("Perception output not JSON object")

        return parsed

    # ==========================================================
    # SCENE ABSTRACTION
    # ==========================================================
    def _build_scene_state(self, perception: Dict[str, Any]) -> Dict[str, Any]:

        raw_objects = perception.get("objects", {})
        objects = list(raw_objects.keys()) if isinstance(raw_objects, dict) else []

        scene = perception.get("scene", {})
        affordances = perception.get("navigational_affordances", [])
        affordances = affordances if isinstance(affordances, list) else []

        # -------- Scene type guess --------
        scene_type = "unknown"
        if isinstance(scene, dict):
            room = scene.get("room")
            if isinstance(room, dict):
                scene_type = room.get("type", "unknown")

        # -------- Uncertainty detection --------
        uncertainties = []
        if not objects:
            uncertainties.append("No objects detected")
        if scene_type == "unknown":
            uncertainties.append("Scene type unclear")
        if not affordances:
            uncertainties.append("No navigational affordances detected")

        # -------- Confidence heuristic --------
        confidence = 0.8
        if uncertainties:
            confidence -= 0.15 * len(uncertainties)
        confidence = max(0.1, min(confidence, 0.9))

        # -------- Summary --------
        summary = (
            f"Objects: {objects}. "
            f"Scene: {scene_type}. "
            f"Affordances: {affordances}."
        )

        return {
            "objects": objects,
            "scene_type": scene_type,
            "affordances": affordances,
            "uncertainties": uncertainties,
            "summary": summary,
            "justification": "Derived from Qwen2-VL visual perception.",
            "confidence": confidence
        }

    # ==========================================================
    # ACTION SYNTHESIS (DETERMINISTIC CONTROL POLICY)
    # ==========================================================
    def _synthesize_actions(
        self,
        scene_state: Dict[str, Any],
        memory_summary: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        actions = []
        affordances = scene_state["affordances"]
        objects = scene_state["objects"]
        scene_type = scene_state["scene_type"]

        # ---------- Rule 1: Door present → approach ----------
        if "door" in objects or "door" in affordances:
            return [{"action": "move_forward", "distance": 0.6}]

        # ---------- Rule 2: Corridor/hallway → advance ----------
        if scene_type in {"corridor", "hallway"} or "corridor" in affordances:
            return [{"action": "move_forward", "distance": 0.6}]

        # ---------- Rule 3: Blocked or enclosed → rotate ----------
        if "blocked" in affordances or "enclosed" in affordances:
            return [{"action": "rotate", "angle_deg": 60}]

        # ---------- Rule 4: No structure → scan ----------
        return [
            {"action": "scan"},
            {"action": "rotate", "angle_deg": 30}
        ]

    # ==========================================================
    # FALLBACK REASONING (NO VLM / DEGRADED MODE)
    # ==========================================================
    def _offline_reasoning(self, question, frame_paths, memory_summary):

        recent = memory_summary[-6:]
        positions = [
            v.get("pose", {}).get("position")
            for v in recent
            if v.get("pose") and v.get("pose", {}).get("position") is not None
        ]

        stagnant = False
        if len(positions) >= 3:
            try:
                d = float(np.linalg.norm(np.array(positions[-1]) - np.array(positions[-3])))
                stagnant = d < 0.5
            except Exception:
                pass

        if stagnant:
            next_actions = [{"action": "rotate", "angle_deg": 90}]
            justification = "Low displacement → likely stuck."
        else:
            next_actions = [{"action": "move_forward", "distance": 0.6}]
            justification = "Default forward exploration."

        return {
            "reasoning": "Offline degraded perception mode.",
            "visible_objects": [],
            "scene_type_guess": "unknown",
            "uncertainties": ["VLM unavailable"],
            "next_actions": next_actions,
            "justification": justification,
            "confidence": 0.2,
            "frames_used": frame_paths
        }

    # ==========================================================
    # OUTPUT SANITIZATION
    # ==========================================================
    def _sanitize_output(self, result: Dict[str, Any]) -> Dict[str, Any]:

        defaults = {
            "reasoning": "",
            "visible_objects": [],
            "scene_type_guess": "unknown",
            "uncertainties": [],
            "next_actions": [],
            "justification": "",
            "confidence": 0.0,
            "frames_used": []
        }

        for k, v in defaults.items():
            if k not in result or result[k] is None:
                result[k] = v

        # -------- Sanitize actions --------
        cleaned = []
        for act in result["next_actions"]:
            if not isinstance(act, dict):
                continue
            if act.get("action") not in ALLOWED_ACTIONS:
                continue

            if act["action"] == "move_forward" and "distance" in act:
                cleaned.append({
                    "action": "move_forward",
                    "distance": float(act["distance"])
                })
            elif act["action"] == "rotate" and "angle_deg" in act:
                cleaned.append({
                    "action": "rotate",
                    "angle_deg": int(act["angle_deg"])
                })
            elif act["action"] in {"scan", "stop"}:
                cleaned.append({"action": act["action"]})

        result["next_actions"] = cleaned
        return result

    # ==========================================================
    # JSON PARSING
    # ==========================================================
    def _safe_json_parse(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        raise ValueError("Invalid JSON from VLM")
