import gym
import rrc_2022_datasets
import numpy as np
import quaternion
import h5py
import time
import os

class RobotDataLoader():
    """
    make it easier and faster to load just parts of the observation in the dataset
    because using obs_to_keep or flatten_obs=False as written in website makes the loading super slow
    params:
        load_from_gym: True by default. set to False if you only need get_observation_indices() (loading datset from gym takes some resources, so only do if necessary)
        task_name: "push" or "lift"
        real: True by default. set to False if you want to load the simulated data
        expert: True by default. set to False if you want to load the mixed data

    """
    def __init__(self, load_from_gym=True, task_name=None, real=True, expert=True):
        if not task_name in ["push", "lift"]:
            raise NotImplementedError()
        if real:
            real_or_sim = "real"
        else:
            real_or_sim = "sim"
        if expert:
            expert_or_mixed = "expert"
        else:
            expert_or_mixed = "mixed"
        assert real or expert, "mixed dataset not available for sim"
        self.obs_size = None  # set in subclass
        self._create_observation_indices()
        if not load_from_gym:
            return
        env = gym.make(
            # NOTE: There is a separate environment for each task and challenge stage.
            # See the documentation of the stages.
            f"trifinger-cube-{task_name}-{real_or_sim}-{expert_or_mixed}-v0",
            disable_env_checker=True,
            visualization=True  # enable visualization
        )
        self.dataset = env.get_dataset()
        assert self.get_observations()[0].shape[0] == self.obs_size

    def get_observations(self, names=None):
        """
        get_observations(["achieved_goal", "action"])
        returns every observation type if called empty
        the "action" in the observation dataset is the previous action taken
        """
        if names is None:
            return self.dataset["observations"]
        return self.dataset["observations"][:, self.get_observation_indices(names)]

    def _create_observation_indices(self):
        raise NotImplementedError()
    
    def get_actions(self):
        """
        thea action here is the current action taken
        """
        return self.dataset["actions"]
    
    def get_rewards(self):
        return self.dataset["rewards"]
    
    def get_episode_ends(self):
        return self.dataset["episode_ends"]
    
    def get_observation_indices(self, names):
        """
        return a list of indices that can be used to access the designated observation types
        """
        indices = np.zeros(0, dtype=int)
        for name in names:
            start_end = self.observation_indices[name]
            indices = np.concatenate([indices, np.r_[start_end[0]:start_end[1]]])
        return indices
        
class PushTaskDataLoader(RobotDataLoader):
    """
    get data for the pushing task
    IGNORES z axis for objects!!

    For reference, this is how information is structured in the dataset:
    First observation:  OrderedDict([('achieved_goal', OrderedDict([('position', array([-0.06085102,  0.00421687,  0.03348794], dtype=float32))])), ('action', array([0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)), ('desired_goal', OrderedDict([('position', array([-0.07460743, -0.01543021,  0.0325    ], dtype=float32))])), ('object_observation', OrderedDict([('confidence', array([1.], dtype=float32)), ('delay', array([0.13], dtype=float32)), ('keypoints', array([[-0.04206596, -0.03773257,  0.06598587],
       [-0.10279939, -0.01457117,  0.06598555],
       [-0.04206406, -0.03772852,  0.00098587],
       [-0.10279749, -0.01456711,  0.00098555],
       [-0.01890456,  0.02300086,  0.06599034],
       [-0.07963798,  0.04616226,  0.06599002],
       [-0.01890266,  0.02300491,  0.00099034],
       [-0.07963609,  0.04616632,  0.00099001]], dtype=float32)), ('orientation', array([ 0.69543   , -0.12810723, -0.12809496,  0.6953829 ], dtype=float32)), ('position', array([-0.06085102,  0.00421687,  0.03348794], dtype=float32))])), ('robot_observation', OrderedDict([('fingertip_force', array([0.05, 0.05, 0.05], dtype=float32)), ('fingertip_position', array([[ 0.08330598,  0.03123533,  0.11588593],
       [-0.01460241, -0.08776277,  0.11588593],
       [-0.06870358,  0.05652744,  0.11588593]], dtype=float32)), ('fingertip_velocity', array([[-3.8634771e-06, -4.7449266e-05,  1.7415987e-04],
       [-3.9160532e-05,  2.7070502e-05,  1.7415987e-04],
       [ 4.3024011e-05,  2.0378764e-05,  1.7415987e-04]], dtype=float32)), ('position', array([ 0.01553102,  0.8853546 , -1.9927638 ,  0.01553102,  0.8853546 ,
       -1.9927638 ,  0.01553102,  0.8853546 , -1.9927638 ], dtype=float32)), ('robot_id', array([0])), ('torque', array([-0.23298728,  0.21940927, -0.06443828, -0.23298728,  0.21940927,
       -0.06443828, -0.23298728,  0.21940927, -0.06443828], dtype=float32)), ('velocity', array([ 3.8008151e-05,  2.5228868e-04, -1.2730500e-03,  3.8008151e-05,
        2.5228868e-04, -1.2730500e-03,  3.8008151e-05,  2.5228868e-04,
       -1.2730500e-03], dtype=float32))]))])
    First action:  [ 0.26785403 -0.08743039  0.13737941 -0.02655286 -0.00451088 -0.03189257
    -0.10650989  0.11829376  0.03032655]
    First reward:  0.880784940276636
    """
    def __init__(self, load_from_gym=True, expert=True):
        super().__init__(load_from_gym=load_from_gym, task_name="push", expert=expert)

    def _create_observation_indices(self):
        # create dict of indices to easier access parts of the observation
        self.obs_size = 97
        self.observation_indices = {}
        index_counter = 0

        step = 3
        self.observation_indices["achieved_goal"] = [index_counter, index_counter + step - 1]
        index_counter += step

        step = 9
        self.observation_indices["action"] = [index_counter, index_counter + step]
        index_counter += step

        step = 3
        self.observation_indices["desired_goal"] = [index_counter, index_counter + step - 1]
        index_counter += step

        step = 1
        self.observation_indices["object/confidence"] = [index_counter, index_counter + step]
        index_counter += step
        self.observation_indices["object/delay"] = [index_counter, index_counter + step]
        index_counter += step
        step = 24
        self.observation_indices["object/keypoints"] = [index_counter, index_counter + step]
        index_counter += step
        step = 4
        self.observation_indices["object/orientation"] = [index_counter, index_counter + step]
        index_counter += step
        step = 3
        self.observation_indices["object/position"] = [index_counter, index_counter + step - 1]
        index_counter += step

        step = 3
        self.observation_indices["robot/fingertip_force"] = [index_counter, index_counter + step]
        index_counter += step
        # 3 dimensions for each finger
        step = 9
        self.observation_indices["robot/fingertip_position"] = [index_counter, index_counter + step]
        index_counter += step
        self.observation_indices["robot/fingertip_velocity"] = [index_counter, index_counter + step]
        index_counter += step
        self.observation_indices["robot/position"] = [index_counter, index_counter + step]
        index_counter += step
        step = 1
        self.observation_indices["robot/robot_id"] = [index_counter, index_counter + step]
        index_counter += step
        step = 9
        self.observation_indices["robot/velocity"] = [index_counter, index_counter + step]
        index_counter += step
        self.observation_indices["robot/torque"] = [index_counter, index_counter + step]
        index_counter += step

        assert self.obs_size == index_counter


class LiftTaskDataLoader(RobotDataLoader):
    """
    get data for lifting task

    The information is structured as such:
    OrderedDict([('achieved_goal', OrderedDict([('keypoints', array([[-0.16048734, -0.0457088 ,  0.00098859],
       [-0.10876109, -0.00634645,  0.00098995],
       [-0.12112499, -0.09743506,  0.00099002],
       [-0.06939874, -0.0580727 ,  0.00099137],
       [-0.16048928, -0.04570849,  0.06598859],
       [-0.10876303, -0.00634614,  0.06598995],
       [-0.12112693, -0.09743474,  0.06599002],
       [-0.06940068, -0.05807239,  0.06599137]], dtype=float32))])), ('action', array([0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)), ('desired_goal', OrderedDict([('keypoints', array([[-0.03383792, -0.09014013,  0.07581523],
       [ 0.03083405, -0.0962739 ,  0.07803186],
       [-0.03528567, -0.12518181,  0.02108877],
       [ 0.02938629, -0.13131557,  0.0233054 ],
       [-0.0401972 , -0.14454104,  0.11081667],
       [ 0.02447476, -0.1506748 ,  0.1130333 ],
       [-0.04164496, -0.17958272,  0.05609021],
       [ 0.02302701, -0.18571648,  0.05830684]], dtype=float32))])), ('object_observation', OrderedDict([('confidence', array([1.], dtype=float32)), ('delay', array([0.12], dtype=float32)), ('keypoints', array([[-0.16048734, -0.0457088 ,  0.00098859],
       [-0.10876109, -0.00634645,  0.00098995],
       [-0.12112499, -0.09743506,  0.00099002],
       [-0.06939874, -0.0580727 ,  0.00099137],
       [-0.16048928, -0.04570849,  0.06598859],
       [-0.10876303, -0.00634614,  0.06598995],
       [-0.12112693, -0.09743474,  0.06599002],
       [-0.06940068, -0.05807239,  0.06599137]], dtype=float32)), ('orientation', array([-3.1953987e-01,  9.4757283e-01, -7.0498418e-06,  1.3351728e-05],
      dtype=float32)), ('position', array([-0.11494401, -0.0518906 ,  0.03348998], dtype=float32))])), ('robot_observation', OrderedDict([('fingertip_force', array([0.05, 0.05, 0.05], dtype=float32)), ('fingertip_position', array([[ 0.08330593,  0.03123633,  0.11588624],
       [-0.01460151, -0.08776322,  0.11588624],
       [-0.06870442,  0.05652689,  0.11588624]], dtype=float32)), ('fingertip_velocity', array([[-3.0175183e-06, -6.2251689e-05,  1.6866020e-04],
       [-5.2402789e-05,  3.3739092e-05,  1.6866020e-04],
       [ 5.5420307e-05,  2.8512599e-05,  1.6866020e-04]], dtype=float32)), ('position', array([ 0.01553137,  0.8853618 , -1.9927672 ,  0.01553137,  0.8853618 ,
       -1.9927672 ,  0.01553137,  0.8853618 , -1.9927672 ], dtype=float32)), ('robot_id', array([0])), ('torque', array([-0.23298948,  0.21941872, -0.06443818, -0.23298948,  0.21941872,
       -0.06443818, -0.23298948,  0.21941872, -0.06443818], dtype=float32)), ('velocity', array([ 3.2619257e-05,  1.4335949e-04, -1.2168230e-03,  3.2619257e-05,
        1.4335949e-04, -1.2168230e-03,  3.2619257e-05,  1.4335949e-04,
       -1.2168230e-03], dtype=float32))]))])
    """
    def __init__(self, load_from_gym=True, expert=True, load_desired_pose=True):
        """
        :param load_desired_pose: load the preprocessed desired pose data output from data_loader_pose
        """
        super().__init__(load_from_gym=load_from_gym, task_name="lift", expert=expert)
        if load_desired_pose and load_from_gym:
            # load the calculated center & orientation info of the goal
            dataset_type_str = "expert" if expert else "mixed"
            this_dir = os.path.dirname(os.path.realpath(__file__))
            filename = f'{this_dir}/../d3rlpy_based/processed_dataset/desired_pose_{dataset_type_str}.h5'
            print(f"loading desired pose from {filename}...")
            with h5py.File(filename, 'r') as f:
                desired_center = f['desired_center'][:]
                desired_orientation = f['desired_orientation'][:]
            # append to the dataset
            self.dataset["observations"] = np.concatenate([self.dataset["observations"], desired_center, desired_orientation], axis=1)

    def _create_observation_indices(self):
        self.obs_size = 139
        self.observation_indices = {}
        index_counter = 0

        step = 24
        self.observation_indices["achieved_goal"] = [index_counter, index_counter + step]
        index_counter += step
        step = 9
        self.observation_indices["action"] = [index_counter, index_counter + step]
        index_counter += step
        step = 24
        self.observation_indices["desired_goal"] = [index_counter, index_counter + step]
        index_counter += step
        step = 1
        self.observation_indices["object/confidence"] = [index_counter, index_counter + step]
        index_counter += step
        step = 1
        self.observation_indices["object/delay"] = [index_counter, index_counter + step]
        index_counter += step
        step = 24
        self.observation_indices["object/keypoints"] = [index_counter, index_counter + step]
        index_counter += step
        step = 4
        self.observation_indices["object/orientation"] = [index_counter, index_counter + step]
        index_counter += step
        step = 3
        self.observation_indices["object/position"] = [index_counter, index_counter + step]
        index_counter += step

        step = 3
        self.observation_indices["robot/fingertip_force"] = [index_counter, index_counter + step]
        index_counter += step
        step = 9
        self.observation_indices["robot/fingertip_position"] = [index_counter, index_counter + step]
        index_counter += step
        self.observation_indices["robot/fingertip_velocity"] = [index_counter, index_counter + step]
        index_counter += step
        self.observation_indices["robot/position"] = [index_counter, index_counter + step]
        index_counter += step
        step = 1
        self.observation_indices["robot/robot_id"] = [index_counter, index_counter + step]
        index_counter += step
        step = 9
        self.observation_indices["robot/torque"] = [index_counter, index_counter + step]
        index_counter += step
        self.observation_indices["robot/velocity"] = [index_counter, index_counter + step]
        index_counter += step

        assert index_counter == self.obs_size  # check that raw dataset size is correct

        # set indices for data processed from data_loader_pose (not in original data)
        step = 3
        self.observation_indices["desired_center"] = [index_counter, index_counter + step]
        index_counter += step
        step = 4
        self.observation_indices["desired_orientation"] = [index_counter, index_counter + step]
        index_counter += step


# def get_pose_from_keypoints(keypoints, dimensions=(0.065, 0.065, 0.065)):
#     center = np.mean(keypoints, axis=0)
#     kp_centered = np.array(keypoints) - center
#     kp_scaled = kp_centered / np.array(dimensions) * 2.0
#     loc_kps = []
#     for i in range(3):
#         # convert to binary representation
#         str_kp = "{:03b}".format(i)
#         # set components of keypoints according to digits in binary representation
#         loc_kp = [(1.0 if str_kp[i] == "0" else -1.0) for i in range(3)][::-1]
#         loc_kps.append(loc_kp)
#     K_loc = np.transpose(np.array(loc_kps))
#     K_loc_inv = np.linalg.inv(K_loc)
#     K_glob = np.transpose(kp_scaled[0:3])
#     R = np.matmul(K_glob, K_loc_inv)
#     quat = quaternion.from_rotation_matrix(R)

#     return center, np.array([quat.x, quat.y, quat.z, quat.w])

if __name__ == "__main__":
    dl = LiftTaskDataLoader()
    # example_obs = ["desired_goal"]
    # print(f"shape of observations for {example_obs}: {dl.get_observations(example_obs).shape}")
    # episode_ends = dl.get_episode_ends()
    # print(f"shape of episode_ends: {episode_ends.shape}")
    # print(f"first five episode_ends: {episode_ends[:5]}")
    # just a sanity check, since these two should be exactly the same
    # assert np.allclose(dl.get_observations(["object/position"]), dl.get_observations(["achieved_goal"]))
    
    st = time.time() # for runtime calculation

    total_obs = dl.get_observations(["desired_goal"]).shape[0] # get the total number of observations to go through row-by-row

    dimensions=(0.065, 0.065, 0.065) #default
    center = np.zeros([total_obs,3]) #center
    quat = np.zeros([total_obs,4]) #quaternion
    loc_kps = []
    for i in range(3):
        # convert to binary representation
        str_kp = "{:03b}".format(i)
        # set components of keypoints according to digits in binary representation
        loc_kp = [(1.0 if str_kp[i] == "0" else -1.0) for i in range(3)][::-1]
        loc_kps.append(loc_kp)  
    K_loc = np.transpose(np.array(loc_kps))
    K_loc_inv = np.linalg.inv(K_loc)
    # start the loop
    for k in range(total_obs):
        keypoints = dl.get_observations(["desired_goal"])[k,:]
        keypoints = keypoints.reshape(8,3)
        center[k,:] = np.mean(keypoints, axis=0).reshape(1,3) #output center
        kp_centered = np.array(keypoints) - center[k,:]
        kp_scaled = kp_centered / np.array(dimensions) * 2.0
        K_glob = np.transpose(kp_scaled[0:3])
        R = np.matmul(K_glob, K_loc_inv)
        quat_array = quaternion.from_rotation_matrix(R)
        quat[k,:] = np.array([quat_array.x, quat_array.y, quat_array.z, quat_array.w])
    # desired_center, desired_orientation = get_pose_from_keypoints(keypoints)
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    # save the data into a h5py dataset
    hf = h5py.File('desired_data.h5', 'w')

    hf.create_dataset('desired_center', data=center)
    hf.create_dataset('desired_orientation', data=quat)

    hf.close()

    # check if they are ok (ToDo: assert a observation)
    hf = h5py.File('desired_data.h5', 'r')
    print(hf.keys())
    n1 = hf.get('desired_center')
    print(np.array(n1)[0,:])
    hf.close()
