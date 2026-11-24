def policy(api):
    grasp_center = api.grasp_point()
    api.approach(grasp_center)
    api.descend(grasp_center)
    api.close_gripper()
    api.lift()