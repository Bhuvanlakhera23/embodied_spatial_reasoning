from narrative import establish_context, traverse, inspect_yaw_only

poses = []

# Act 1: Context
poses += establish_context(
    position=[0.0, 1.5, 0.0]
)

# Act 2: Move forward
poses += traverse(
    start=[0.0, 1.5, 0.0],
    direction=[0, 0, -1],
    step=0.3,
    steps=6
)

# Act 3: Inspect (A)
poses += inspect_yaw_only(
    position=poses[-1]["position"],
    yaw_angles=[-90, -60, -30, 0, 30, 60, 90],
    pitch=0.0
)
