def policy(api):
    api.raise_clear()
    pose = api.target_pose()
    api.move_to(pose)
    api.settle()