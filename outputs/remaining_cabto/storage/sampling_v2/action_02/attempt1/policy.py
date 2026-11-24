def policy(api):
    pose = api.target_pose()
    api.move_to(pose)
    api.settle()