def policy(api):
    board_destination = api.destination_point()
    api.carry(board_destination)
    api.lower(board_destination)
    api.open_gripper()
    api.withdraw()
