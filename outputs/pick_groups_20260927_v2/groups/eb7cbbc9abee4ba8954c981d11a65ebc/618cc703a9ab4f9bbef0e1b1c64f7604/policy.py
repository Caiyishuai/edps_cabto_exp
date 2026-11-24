def policy(api):
    grasp_center = api.grasp_point()
    grasp_center = (grasp_center[0], grasp_center[1], grasp_center[2] + 0.045)
    api.approach(grasp_center)
    api.descend(grasp_center)
    api.close_gripper()
    api.lift()