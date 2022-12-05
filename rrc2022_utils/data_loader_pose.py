import gym
import rrc_2022_datasets
import numpy as np
import quaternion
import h5py
import time
import os

# variables used in get_pose_from_keypoints()
loc_kps = []
for i in range(3):
    # convert to binary representation
    str_kp = "{:03b}".format(i)
    # set components of keypoints according to digits in binary representation
    loc_kp = [(1.0 if str_kp[i] == "0" else -1.0) for i in range(3)][::-1]
    loc_kps.append(loc_kp)  
K_loc = np.transpose(np.array(loc_kps))
K_loc_inv = np.linalg.inv(K_loc)


def get_pose_from_keypoints(keypoints):
    dimensions = (0.065, 0.065, 0.065) #default

    center = np.mean(keypoints, axis=0).reshape(1,3) #output center
    kp_centered = np.array(keypoints) - center
    kp_scaled = kp_centered / np.array(dimensions) * 2.0
    K_glob = np.transpose(kp_scaled[0:3])
    R = np.matmul(K_glob, K_loc_inv)
    quat_array = quaternion.from_rotation_matrix(R)

    return center, np.array([quat_array.x, quat_array.y, quat_array.z, quat_array.w])


if __name__ == "__main__":
    from robot_data_loader import LiftTaskDataLoader
    expert = True
    dataset_type_str = "expert" if expert else "mixed"
    dl = LiftTaskDataLoader(expert=expert, load_desired_pose=False)
    # example_obs = ["desired_goal"]
    # print(f"shape of observations for {example_obs}: {dl.get_observations(example_obs).shape}")
    # episode_ends = dl.get_episode_ends()
    # print(f"shape of episode_ends: {episode_ends.shape}")
    # print(f"first five episode_ends: {episode_ends[:5]}")
    # just a sanity check, since these two should be exactly the same
    # assert np.allclose(dl.get_observations(["object/position"]), dl.get_observations(["achieved_goal"]))
    
    st = time.time() # for runtime calculation

    total_obs = dl.get_observations(["desired_goal"]).shape[0] # get the total number of observations to go through row-by-row
    center = np.zeros([total_obs,3]) #center
    quat = np.zeros([total_obs,4]) #quaternion

    # get the indices of the episode ends
    episode_end_indices = dl.get_episode_ends()

    episode_start = 0
    count = 0
    # iterate through episodes
    for episode_end in episode_end_indices:
        keypoints = dl.get_observations(["desired_goal"])[episode_start,:]
        keypoints = keypoints.reshape(8,3)
        # copy to each observation in the episode
        center[episode_start:episode_end+1,:], quat[episode_start:episode_end+1,:] = get_pose_from_keypoints(keypoints)
        episode_start = episode_end + 1
        count += 1
        if count % 50 == 0:
            print(f"finished episode {count} / {len(episode_end_indices)}")
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    # save the data into a h5py dataset
    this_dir = os.path.dirname(os.path.realpath(__file__))
    filename = f'{this_dir}/../d3rlpy_based/processed_dataset/desired_pose_{dataset_type_str}.h5'
    hf = h5py.File(filename, 'w')

    hf.create_dataset('desired_center', data=center)
    hf.create_dataset('desired_orientation', data=quat)

    hf.close()

    # check if they are ok (ToDo: assert a observation)
    hf = h5py.File(filename, 'r')
    print(hf.keys())
    n1 = hf.get('desired_center')
    print(np.array(n1)[0,:])
    hf.close()
