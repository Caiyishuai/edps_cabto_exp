def policy(api):
    pose = api.target_pose()
    api.carry(pose)
    api.lower(pose)
    api.open_gripper()
    api.detach()
    api.retreat_vertical()
    api.withdraw_safe()