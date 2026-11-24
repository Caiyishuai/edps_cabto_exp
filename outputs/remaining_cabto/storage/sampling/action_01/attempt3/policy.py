def policy(api):
    api.move_above(api.target_pose())
    api.lower(api.target_pose())
    api.place_release()