def policy(api):
    pose = api.target_pose()
    api.carry(pose)
    api.lower(pose)
    api.open_gripper()
    api.detach()
    api.withdraw_safe()