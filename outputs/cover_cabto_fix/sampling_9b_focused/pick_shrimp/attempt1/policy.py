def policy(api):
    api.approach((0.3, 0.3, 0.5))
    api.descend((0.3, 0.3, 0.5))
    api.close_gripper()
    api.lift()