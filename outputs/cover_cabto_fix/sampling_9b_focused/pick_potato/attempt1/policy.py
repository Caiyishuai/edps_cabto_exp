def policy(api):
    api.approach([0.35, 0.45, 0.6])
    api.descend([0.35, 0.45, 0.55])
    api.close_gripper()
    api.lift()