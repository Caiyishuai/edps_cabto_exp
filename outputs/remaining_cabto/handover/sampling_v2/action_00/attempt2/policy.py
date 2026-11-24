def policy(api):
    grasp_pose = api.grasp_pose()
    api.approach(grasp_pose)
    api.descend(grasp_pose)
    api.close_gripper()
    api.attach_contact_gated()
    api.lift()
    return api.status()