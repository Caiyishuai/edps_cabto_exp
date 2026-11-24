def policy(api):
    pose = api.target_pose()
    api.move_above(pose)
    api.align_target(pose)
    api.lower(pose)
    api.settle()
    api.place_release()
    api.retreat()