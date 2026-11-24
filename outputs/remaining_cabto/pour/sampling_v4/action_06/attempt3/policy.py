def policy(api):
    pose = api.target_pose()
    api.move_safe(pose)
    api.move_above(pose)
    api.align_target(pose)
    api.lower(pose)
    api.release()
    api.retreat()