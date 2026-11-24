def policy(api):
    api.open_gripper()
    potato_grasp = api.grasp_point()
    api.approach(potato_grasp)
    api.descend(potato_grasp)
    api.close_gripper()
    api.lift()
