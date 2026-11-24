def policy(api):
    target = api.target_pose()
    safe = api.safe_pose()
    api.move_safe(safe)
    api.move_above(target)
    api.lower(target)
    api.release()
    api.retreat()