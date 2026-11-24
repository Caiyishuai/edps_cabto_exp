def policy(api):
    safe_pos = api.safe_pose()
    api.move_safe(safe_pos)
    target_pos = api.target_pose()
    api.lower(target_pos)
    api.release()
    api.retreat()