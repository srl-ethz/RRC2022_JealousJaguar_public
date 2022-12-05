import matplotlib.pyplot as plt
from robot_data_loader import PushTaskDataLoader, LiftTaskDataLoader
import numpy as np

"""
visualize the dataset by plotting some data for a single episode
"""

episode_to_check = 3

expert = True
dl = LiftTaskDataLoader(expert=expert)

object_pos_indices = dl.get_observation_indices(["object/position"])
object_ori_indices = dl.get_observation_indices(["object/orientation"])
desired_pos_indices = dl.get_observation_indices(["desired_center"])
desired_ori_indices = dl.get_observation_indices(["desired_orientation"])

episode_start = dl.get_episode_ends()[episode_to_check-1] + 1 if episode_to_check > 0 else 0
episode_end = dl.get_episode_ends()[episode_to_check]

obs = dl.get_observations()[episode_start:episode_end+1,:]

for i in range(3):
    xyz = ["x", "y", "z"][i]
    plt.plot(obs[:,object_pos_indices[i]], label=f"object_{xyz}")
    plt.plot(obs[:,desired_pos_indices[i]], label=f"desired_{xyz}")
plt.title("position")
plt.legend()
plt.show()

for i in range(4):
    xyzw = ["x", "y", "z", "w"][i]
    plt.plot(obs[:,object_ori_indices[i]], label=f"object_{xyzw}")
    plt.plot(obs[:,desired_ori_indices[i]], label=f"desired_{xyzw}")
plt.title("Orientation")
plt.legend()
plt.show()