def policy(api):
    api.approach((0.5, 0.5, 0.5))
    api.descend((0.5, 0.5, 0.5))
    api.close_gripper()
    api.lift()