import numpy as np
import os
import d3rlpy
from sklearn.model_selection import train_test_split

from reward_relabeling import recomputed_reward


def preprocess_data(
    rdl,
    args,
    expanded_state_dim,
    act_dims,
    obs_dims,
    obs_used,
    history_size,
    filenames,
    aug=False,
):

    # preprocess the dataset to become a concatenated vector of past [history_size] observations and [history_size - 1] actions
    # the results of the preprocessing are stored in the following variables
    expanded_states = np.zeros((0, expanded_state_dim))
    act = np.zeros((0, act_dims))
    rew = np.zeros((0,))
    terminal = np.zeros((0,), dtype=bool)

    print("preprocessed files not found, preprocessing data...")
    obs_raw = rdl.get_observations(obs_used)
    act_raw = rdl.get_actions()
    rew_raw = rdl.get_rewards()

    if aug:
        aug_method = args.data_aug_method
        from data_augment import DataAug

        data_augmentor = DataAug()
        obs_raw, act_raw, rew_raw = data_augmentor.aug(
            obs_raw, act_raw, rew_raw, method=aug_method
        )

    terminal_indices = rdl.get_episode_ends()

    episode_start = 0
    count = 0
    for episode_end in terminal_indices:
        # length of the dataset created from this episode
        dataset_length = episode_end - episode_start - history_size + 2
        # the source of the dataset used for constructing the training data for this episode
        obs_raw_episode = obs_raw[episode_start : episode_end + 1, :]
        act_raw_episode = act_raw[episode_start : episode_end + 1, :]
        rew_raw_episode = rew_raw[episode_start : episode_end + 1]

        # the states for this episode are temporarily saved here before concatenating to the full dataset
        expanded_states_episode = np.zeros((dataset_length, expanded_state_dim))
        act_episode = np.zeros((dataset_length, act_dims))
        rew_episode = np.zeros((dataset_length,))
        terminal_episode = np.zeros((dataset_length,), dtype=bool)

        # this is probably the bottleneck, could probs be sped up but who cares
        for i in range(history_size):
            # copy data for i-th step
            expanded_states_episode[
                :, obs_dims * i : obs_dims * (i + 1)
            ] = obs_raw_episode[i : i + dataset_length]
            if i < history_size - 1:
                # the latest action is not included in the expanded state
                act_start_idx = (
                    obs_dims * history_size
                )  # where the action is started to be stored in the expanded state
                expanded_states_episode[
                    :, act_start_idx + act_dims * i : act_start_idx + act_dims * (i + 1)
                ] = act_raw_episode[i : i + dataset_length]
        act_episode[:] = act_raw_episode[-dataset_length:]
        rew_episode[:] = rew_raw_episode[-dataset_length:]
        terminal_episode[-1] = True  # set the very end to 1

        expanded_states = np.concatenate(
            (expanded_states, expanded_states_episode), axis=0
        )
        act = np.concatenate((act, act_episode), axis=0)
        rew = np.concatenate((rew, rew_episode), axis=0)
        terminal = np.concatenate((terminal, terminal_episode), axis=0)

        count += 1
        if count % 100 == 0:
            print(f"{count} out of {len(terminal_indices)} episodes processed...")
        episode_start = episode_end + 1
    np.save(filenames[0], expanded_states)
    np.save(filenames[1], act)
    np.save(filenames[2], rew)
    np.save(filenames[3], terminal)
    return expanded_states, act, rew, terminal


def load_data(filenames):
    print(
        f"loading preprocessed data from {filenames[0]} {filenames[1]} {filenames[2]} {filenames[3]}..."
    )
    expanded_states = np.load(filenames[0])
    act = np.load(filenames[1])
    rew = np.load(filenames[2])
    terminal = np.load(filenames[3])
    return expanded_states, act, rew, terminal


def generate_dataset(
    args, rdl, obs_used, obs_dims, act_dims, expanded_state_dim, history_size
):
    task = args.task
    aug_method = args.data_aug_method

    if aug_method != "":
        expanded_states_filename = (
            f"processed_dataset/expanded_states_{aug_method}AUG.npy"
        )
        act_filename = f"processed_dataset/act_{aug_method}AUG.npy"
        rew_filename = f"processed_dataset/rew_{aug_method}AUG.npy"
        terminal_filename = f"processed_dataset/terminal_{aug_method}AUG.npy"
        filenames_aug = [
            expanded_states_filename,
            act_filename,
            rew_filename,
            terminal_filename,
        ]

    expanded_states_filename = f"processed_dataset/expanded_states.npy"
    act_filename = f"processed_dataset/act.npy"
    rew_filename = f"processed_dataset/rew.npy"
    terminal_filename = f"processed_dataset/terminal.npy"
    filenames = [
        expanded_states_filename,
        act_filename,
        rew_filename,
        terminal_filename,
    ]

    if (
        os.path.isfile(expanded_states_filename)
        and os.path.isfile(act_filename)
        and os.path.isfile(rew_filename)
        and os.path.isfile(terminal_filename)
    ):
        expanded_states, act, rew, terminal = load_data(filenames)
    else:
        expanded_states, act, rew, terminal = preprocess_data(
            rdl,
            args,
            expanded_state_dim,
            act_dims,
            obs_dims,
            obs_used,
            history_size,
            filenames,
        )

    if aug_method != "":
        if (
            os.path.isfile(filenames_aug[0])
            and os.path.isfile(filenames_aug[1])
            and os.path.isfile(filenames_aug[2])
            and os.path.isfile(filenames_aug[3])
        ):
            expanded_states_aug, act_aug, rew_aug, terminal_aug = load_data(
                filenames_aug
            )
        else:
            expanded_states_aug, act_aug, rew_aug, terminal_aug = preprocess_data(
                rdl,
                args,
                expanded_state_dim,
                act_dims,
                obs_dims,
                obs_used,
                history_size,
                filenames_aug,
                aug=True,
            )

        expanded_states = np.vstack([expanded_states, expanded_states_aug])
        act = np.vstack([act, act_aug])
        rew = np.concatenate([rew, rew_aug])
        terminal = np.concatenate([terminal, terminal_aug])

    assert expanded_states.shape[0] == act.shape[0] == rew.shape[0] == terminal.shape[0]
    # processed_dataset must be re-generated if obs_used or history_size is changed
    assert expanded_states.shape[1] == expanded_state_dim
    assert act.shape[1] == act_dims
    assert len(rew.shape) == 1
    assert len(terminal.shape) == 1

    if task == "lift":
        pass
        # print("computing the new rewards...")
        # this is hardcoded for the observations used in the lift task...
        # indices must be changed since desired_center and desired_orientation is included in observation now
        # achieved_goal = expanded_states[
        #     :, obs_dims * (history_size - 1) + 9 : obs_dims * (history_size - 1) + 33
        # ]
        # desired_goal = expanded_states[
        #     :, obs_dims * (history_size - 1) + 43 : obs_dims * (history_size - 1) + 67
        # ]
        # rew_recomputed = recomputed_reward(achieved_goal, desired_goal)
        # # do not use the recomputed rewards for now, it only degrades performance
        # print("done")

    print(
        f"loading dataset with {expanded_states.shape[0]} timesteps and {terminal.sum()} episodes"
    )
    dataset = d3rlpy.dataset.MDPDataset(
        observations=expanded_states, actions=act, rewards=rew, terminals=terminal
    )
    # episode-wise split
    train_episodes, test_episodes = train_test_split(dataset.episodes, test_size=0.05)
    return train_episodes, test_episodes
