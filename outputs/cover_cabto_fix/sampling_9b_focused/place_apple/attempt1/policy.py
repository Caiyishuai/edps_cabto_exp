def policy(api):
    api.lower(api.destination_point())
    api.open_gripper()
    api.withdraw()