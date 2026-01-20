class ViewMemory:
    def __init__(self):
        self.views = []
        self.counter = 0

    def add_view(self, frame, pose, action):
        vid = f"{self.counter:03d}"
        self.views.append({
            "id": vid,
            "action": action,
            "pose": pose,
            "frame": frame
        })
        self.counter += 1
        return vid

    def export_json(self):
        return [
            {
                "id": v["id"],
                "action": v["action"],
                "pose": v["pose"]
            } for v in self.views
        ]
