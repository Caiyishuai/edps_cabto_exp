def policy(api):
    api.open_gripper()
    grasp = api.grasp_point()
    api.approach(grasp)
    api.descend(grasp)
    api.close_gripper()
    api.lift()
