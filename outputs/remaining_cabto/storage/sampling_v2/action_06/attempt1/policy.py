def policy(api):
    handles = api.target_handles()
    api.move_both_above(handles)
    api.move_both_to(handles)
    api.close_both()
    api.attach_both()
    api.settle()