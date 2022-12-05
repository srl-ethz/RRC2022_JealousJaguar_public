import numpy as np

"""
split the reward into ones for position and orientation, since adjusting orientation then position may be easier for the robot and that may be encoded through rewards
"""

def _kernel_reward(dist):
    """
    implement clone of logistic kernel function seen in rrc code
    https://github.com/rr-learning/rrc_2022_datasets/blob/61a58d7837d3189a22676fe2c5d0a0997c0c154c/rrc_2022_datasets/sim_env.py#L193
    input: (N, x) where x can be the number of keypoints (x = 1 if only one keypoint)
    """
    _logkern_scale = 30
    _logkern_offset = 2
    _kernel_reward_weight = 4
    scaled = _logkern_scale * dist
    # Use logistic kernel
    rew = _kernel_reward_weight * np.mean(
        1.0 / (np.exp(scaled) + _logkern_offset + np.exp(-scaled)), axis=1)
    return rew

def recomputed_reward(achieved_goal_raw, desired_goal_raw):
    """
    compute the reward for orientation and position separately, and add them so that orientation can be given same weight as position in reward
    achieved_goal_raw: (N, 24)
    desired_goal_raw: (N, 24)
    could get the quat according to this
    https://github.com/rr-learning/rrc_2022_datasets/blob/61a58d7837d3189a22676fe2c5d0a0997c0c154c/rrc_2022_datasets/utils.py#L52
    and calculate angle difference between desired and target, but let's use something simpler for now
    """
    assert achieved_goal_raw.shape == desired_goal_raw.shape
    assert len(achieved_goal_raw.shape) == 2
    assert achieved_goal_raw.shape[1] == 24

    # reshape to a tensor where achieved_keypoints[i] is a list of 8 keypoints
    achieved_keypoints = achieved_goal_raw.reshape(-1, 8, 3)
    desired_keypoints = desired_goal_raw.reshape(-1, 8, 3)
    achieved_centers = achieved_keypoints.mean(axis=1)
    desired_centers = desired_keypoints.mean(axis=1)
    # get the keypoints but where the center of the cube is at the origin (i.e. mean for axis=1 should become zero)
    achieved_keypoints_centered = achieved_keypoints - achieved_centers.reshape(-1, 1, 3)
    desired_keypoints_centered = desired_keypoints - desired_centers.reshape(-1, 1, 3)

    # https://github.com/rr-learning/rrc_2022_datasets/blob/61a58d7837d3189a22676fe2c5d0a0997c0c154c/rrc_2022_datasets/sim_env.py#L193
    diff_pos = np.linalg.norm(achieved_centers - desired_centers, axis=1)
    diff_angle_ish = np.linalg.norm(achieved_keypoints_centered - desired_keypoints_centered, axis=2)

    rew_pos = _kernel_reward(diff_pos.reshape(-1, 1))
    rew_angle = _kernel_reward(diff_angle_ish)

    # print(f"rew_pos: {rew_pos}\trew_angle: {rew_angle}")
    # TODO: maybe weight this?
    return rew_pos + rew_angle


def original_reward(achieved_goal, desired_goal):
    """
    compute the default reward defined by the organizers
    """
    achived_keypoints = achieved_goal.reshape(-1, 8, 3)
    desired_keypoints = desired_goal.reshape(-1, 8, 3)
    return _kernel_reward(np.linalg.norm(achived_keypoints - desired_keypoints, axis=2))