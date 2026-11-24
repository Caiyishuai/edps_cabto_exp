def policy(api):
    safe = api.safe_pose()
    api.move_safe(safe)
    api.release()
    api.retreat()