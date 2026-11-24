def policy(api):
    api.carry(api.destination_point())
    api.lower(api.destination_point())
    api.open_gripper()
    api.withdraw()