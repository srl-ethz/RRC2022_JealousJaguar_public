## TODO: create efficient way to concurrently train, evaluate, and compare different combinations of these parameters to make it easier to search for best combo
# which observations are used in the state space?
obs_used_push = ["robot/fingertip_position", "object/position", "object/delay", "robot/torque", "desired_goal", "robot/fingertip_force"]
obs_used_lift = ["robot/fingertip_position", "achieved_goal", "object/delay", "robot/torque", "desired_goal", "robot/fingertip_force"]
# how many past observations and actions are included in the state space?
history_size_push = 3
history_size_lift = 4
nsteps_push = 1
nsteps_lift = 9
