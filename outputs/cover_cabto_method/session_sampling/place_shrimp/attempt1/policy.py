def policy(api):
    bowl_destination = api.destination_point()
    api.carry(bowl_destination)
    api.lower(bowl_destination)
    api.open_gripper()
    api.withdraw()
