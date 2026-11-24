def policy(api):
    target = api.target_pose()
    api.move_above(target)
    api.lower(target)
    api.release()
    api.retreat()