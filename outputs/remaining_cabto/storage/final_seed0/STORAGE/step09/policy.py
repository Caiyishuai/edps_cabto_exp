def policy(api):
    pose = api.target_tray_pose()
    api.move_tray_to(pose)
    api.settle()