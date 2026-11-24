def policy(api):
    # Get the grasp point for the shrimp
    grasp_point = api.grasp_point()
    
    # Approach the shrimp at a safe height
    api.approach(grasp_point)
    
    # Descend to the grasp point
    api.descend(grasp_point)
    
    # Close the gripper to grasp the shrimp
    api.close_gripper()
    
    # Lift the shrimp to transport height
    api.lift()
