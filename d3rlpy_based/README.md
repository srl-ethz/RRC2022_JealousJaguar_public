# d3rlpy_based
uses the excellent d3rlpy library for offline RL: https://github.com/takuseno/d3rlpy

first, set up environment as written in top level README of this repo. Make sure to activate venv before running any of the code. Then `pip install d3rlpy wandb`.

## Weights and Biases setup
Make sure you're in the SRL group https://wandb.ai/srl_ethz

In command line, `wandb login` to login

## compute desired_center and desired_orientation
convert the keypoint representation (8 points on each vertex of the cube) to center (xyz position) and orientation (quaternion) representation.

far more efficient way to track object pose than using keypoints (24 DoF -> 7 DoF)

run /rrc2022_utils/data_loader_pose.py (script must be modified if running for mixed dataset) to output preprocessed data that can be loaded later, or copy [desired_pose_expert.h5](https://drive.google.com/file/d/1EIC6AmGGJkotZ7lzIAbt8lwPOz_yOPu3/view?usp=sharing) and [desired_pose_mixed.h5](https://drive.google.com/file/d/1nf1KQYDooLPcWPxd2OqmvfI4MHwnEkPu/view?usp=sharing) to processed_dataset

## train policy
Please check out the documentation in train.py for specific quirks of the code
train.py conveniently implements a feature where it can automatically periodically execute the latest policy on the real robot system, and log that to the Weights & Biases platform!
```bash
python train.py --task lift --auto_run_robot true --dataset_type expert --name "sac_larger_lr"--comment "adjusting this and that parameter..."
```

## submit policy to robot cluster
see https://webdav.tuebingen.mpg.de/real-robot-challenge/2022/docs/robot_phase/index.html


## run policy (for sim stage only)
run policy in simulated environment using pretrained weights
(neither of the policies are completely trained yet, but will still somewhat work)

```bash
cd /path/to/RRC22/
python -m rrc_2022_datasets.evaluate_pre_stage lift d3rlpy_based.policy.LiftExpertPolicy --n-episodes=10 -v
python -m rrc_2022_datasets.evaluate_pre_stage push d3rlpy_based.policy.PushExpertPolicy --n-episodes=10 -v
```
LiftMixedPolicy and PushMixedPolicy must also be trained

