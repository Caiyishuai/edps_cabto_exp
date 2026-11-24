def policy(api):
    pan_destination = api.destination_point()
    api.carry(pan_destination)
    api.lower(pan_destination)
    api.open_gripper()
    api.withdraw()
