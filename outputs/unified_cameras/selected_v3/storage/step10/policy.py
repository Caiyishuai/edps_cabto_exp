def policy(api):
    pose = (0, 0, 0, 0, 0, 0)  # Example pose for the jointly held tray

    # Open both grippers
    api.open_both()

    # Detach both declared welds
    api.detach_both()

    # Wait for the released tray to rest on the shelf
    api.settle()