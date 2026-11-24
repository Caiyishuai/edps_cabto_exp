def policy(api):
    # Step 1: Approach the shrimp
    shrimp_pos = api.grasp_point()
    api.approach(shrimp_pos)
    
    # Step 2: Descend to grasp the shrimp
    api.descend(shrimp_pos)
    
    # Step 3: Close gripper to grasp the shrimp
    api.close_gripper()
    
    # Step 4: Lift the shrimp
    api.lift()
    
    # Step 5: Approach the bowl
    bowl_pos = api.grasp_point()
    api.approach(bowl_pos)
    
    # Step 6: Descend to place the shrimp in the bowl
    api.descend(bowl_pos)
    
    # Step 7: Open gripper to release the shrimp in the bowl
    api.open_gripper()