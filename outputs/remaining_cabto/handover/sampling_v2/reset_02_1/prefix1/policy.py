def policy(api):
    target = api.target_pose()
    if target is not None:
        api.move_held(target)
        api.settle()