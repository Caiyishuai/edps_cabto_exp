def policy(api):
    api.open_gripper()
    apple_grasp = api.grasp_point()
    api.approach(apple_grasp)
    api.descend(apple_grasp)
    api.close_gripper()
    api.lift()
