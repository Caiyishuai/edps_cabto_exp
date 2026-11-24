def policy(api):
    arm = api.get_arm()
    if arm is not None:
        api.return_arm_home()