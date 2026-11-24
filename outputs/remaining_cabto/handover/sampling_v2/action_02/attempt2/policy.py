def policy(api):
    grasp_point = api.grasp_pose()
    api.descend(grasp_point)
    api.attach_contact_gated()