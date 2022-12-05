import numpy as np
import torch
from os import path
import json

from rrc_2022_datasets import PolicyBase
from d3rlpy.algos import IQL
from d3rlpy_based.reward_relabeling import original_reward

import sys
# add parent directory to path
# make sure this will work no matter where it is called from
file_path = path.dirname(path.realpath(__file__))
sys.path.insert(0, file_path + '/..')

from rrc2022_utils.robot_data_loader import PushTaskDataLoader, LiftTaskDataLoader
from rrc2022_utils.data_loader_pose import get_pose_from_keypoints


class Policy(PolicyBase):
    def __init__(self, prefix, data_loader):
        """
        look into d3rlpy_based/trained_models directory with the filename [prefix]_1, [prefix]_2, etc.
        Super custom way to save the observations and history_size used for each model...
        """
        self.device = "cpu"

        # convert policy into a model loadable just with pytorch with no d3rlpy dependency
        # this is because d3rlpy.predict, which is the policy, has some overheat that slows down processing
        this_dir = path.dirname(__file__)
        # list of the different policies used
        self.policy_list = []
        # list of the parameters for the state used for each policy
        self.observations_used_list = []
        self.observation_indices_list = []
        self.history_size_list = []
        for i in range(1, 100):
            model_filename = path.join(this_dir, "trained_models", f"{prefix}_{i}.pt")
            param_filename = path.join(this_dir, "trained_models", f"{prefix}_{i}.json")
            metadata_filename = path.join(this_dir, "trained_models", f"{prefix}_metadata_{i}.json")
            # check if the file exists
            if not path.isfile(model_filename) or not path.isfile(param_filename):
                break
            d3rlpy_algo = IQL.from_json(param_filename)
            d3rlpy_algo.load_model(model_filename)
            d3rlpy_algo.save_policy("policy.pt")
            policy = torch.jit.load("policy.pt", map_location=self.device)
            self.policy_list.append(policy)
            # load the metadata
            with open(metadata_filename, "r") as f:
                metadata = json.load(f)
            self.observations_used_list.append(metadata["obs_used"])
            self.history_size_list.append(metadata["history_size"])

            self.observation_indices_list.append(data_loader.get_observation_indices(self.observations_used_list[-1]))
        self.policy_num = len(self.policy_list)
        print(f"Loaded {self.policy_num} policies")
        assert self.policy_num > 0, "No policies loaded"
        max_history_size = max(self.history_size_list)

        self.desired_goal_indices = data_loader.get_observation_indices(["desired_goal"])
        self.achieved_goal_indices = data_loader.get_observation_indices(["achieved_goal"])
        self.act_dim = self.action_space.shape[0]
        # save the past history_size observations
        self.obs_buffer = [None] * self.policy_num
        # save the past actions
        self.act_buffer = np.zeros((max_history_size - 1, self.act_dim))

        # always normalized so the sum is 1
        self.policy_scores = np.ones(self.policy_num)/self.policy_num
        self.policy_used = 0

        self.step = 0
        # resets at the beginning or every 50 * 5 steps (5 seconds)
        # todo is push task also 50 Hz
        self.mark_step = 0

    @staticmethod
    def is_using_flattened_observations():
        return True

    def reset(self):
        self.step = 0
        self.mark_step = 0
        for i in range(self.policy_num):
            self.obs_buffer[i] = None
        self.act_buffer = np.zeros_like(self.act_buffer)
        self.policy_used = 0
        self.policy_scores = np.ones(self.policy_num)/self.policy_num

    def get_action(self, observation_raw):
        self.step += 1

        # a hacky way to check it is doing the lifting task
        is_lift_task = self.desired_goal_indices.shape == (24,)

        desired_goal = observation_raw[self.desired_goal_indices]
        achieved_goal = observation_raw[self.achieved_goal_indices]

        if is_lift_task:
            # compute the desired center and orientation
            desired_keypoints = observation_raw[self.desired_goal_indices].reshape(8, 3)
            center, orientation = get_pose_from_keypoints(desired_keypoints)
            # append to the observation
            observation_raw = np.concatenate([observation_raw, center[0], orientation])

        if self.step - self.mark_step > 50 * 5 and is_lift_task:
            # every 5 seconds, check if the reward is high enough, and if not, switch the policy
            self.mark_step = self.step
            reward = original_reward(desired_goal, achieved_goal)[0]
            if reward < 0.7:
                # update the policy scores
                self.policy_scores[self.policy_used] *= 0.5
                self.policy_scores /= np.sum(self.policy_scores)
                # choose policy with the highest score
                self.policy_used = np.argmax(self.policy_scores)
                print(f"Policy switched to {self.policy_used}")

        for i in range(self.policy_num):
            # get only the relevant observations for this policy
            observation = observation_raw[self.observation_indices_list[i]]
            if self.obs_buffer[i] is None:
                # initialize the observation buffer by copying the first observation
                self.obs_buffer[i] = np.tile(observation, (self.history_size_list[i], 1))
            # append the new observation to the buffer
            self.obs_buffer[i][:-1] = self.obs_buffer[i][1:]
            self.obs_buffer[i][-1] = observation
        
        # create the extended state
        act_buffer = self.act_buffer[-(self.history_size_list[self.policy_used]-1):]
        state = np.concatenate((self.obs_buffer[self.policy_used].flatten(), act_buffer.flatten()))
        state = torch.from_numpy(state).to(self.device).float()
        with torch.no_grad():
            action_raw = self.policy_list[self.policy_used](state.unsqueeze(0))
        action = np.clip(action_raw[0], self.action_space.low, self.action_space.high)

        if len(self.act_buffer) > 0:
            # append the new action to the buffer
            self.act_buffer[:-1] = self.act_buffer[1:]
            self.act_buffer[-1] = action

        return action


class PushExpertPolicy(Policy):
    def __init__(self, action_space, observation_space, episode_length):
        self.action_space = action_space
        dl = PushTaskDataLoader(load_from_gym=False)
        super().__init__("push_expert", dl)


class PushMixedPolicy(Policy):
    def __init__(self, action_space, observation_space, episode_length):
        self.action_space = action_space
        dl = PushTaskDataLoader(load_from_gym=False)
        super().__init__("push_mixed", dl)


class LiftExpertPolicy(Policy):
    def __init__(self, action_space, observation_space, episode_length):
        self.action_space = action_space
        dl = LiftTaskDataLoader(load_from_gym=False)
        super().__init__("lift_expert", dl)


class LiftMixedPolicy(Policy):
    def __init__(self, action_space, observation_space, episode_length):
        self.action_space = action_space
        dl = LiftTaskDataLoader(load_from_gym=False)
        super().__init__("lift_mixed", dl)