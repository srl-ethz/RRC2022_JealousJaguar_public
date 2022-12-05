import numpy as np
import torch
import os
import argparse
import shutil
from threading import Thread
import d3rlpy
from d3rlpy.algos import IQL
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import td_error_scorer
from sklearn.model_selection import train_test_split

from __init__ import obs_used_push, obs_used_lift, history_size_push, history_size_lift, nsteps_push, nsteps_lift
from reward_relabeling import recomputed_reward

# hack to enable loading scripts from parent directory
import sys
sys.path.insert(0,'..')
from rrc2022_utils.robot_data_loader import PushTaskDataLoader, LiftTaskDataLoader
from rrc2022_utils.automated_submission import automated_submission, base_url

import wandb

from data_preprocess import generate_dataset

"""
python train.py --task push --auto_run_robot true --dataset_type expert --name "sac_larger_lr"--comment "adjusting this and that parameter..."

please note:
- manually edit trifinger.toml to your task & dataset_type to match what you are training for (this is read by remote robot to determine which policy to run) (if you could )
- the result of preprocessing is saved in *.npy files in preprocessed_dataset/, but if you change anything that would change the preprocessed data, (at least one of) the *.npy files should be deleted to re-run preprocessing
- if return_mean and success_rate shows up in W&B as zero, some error caused real robot run to fail. Make sure to check the logs in W&B and the real robot server, and fix the problem.

https://wandb.ai/site/articles/intro-to-pytorch-with-wandb
https://theaisummer.com/weights-and-biases-tutorial/
"""

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True, choices=["push", "lift"], help="push or lift. required.")
parser.add_argument('--auto_run_robot', type=bool, default=True, help="automatically submit to real robot cluster at regular intervals. default is true")
parser.add_argument("--auto_run_robot_interval", type=int, default=10, help="every ? epoch, evaluate on real robot (also evaluates at the very final epoch). default is 10")
parser.add_argument("--dataset_type", type=str, default="expert", choices=["expert", "mixed"], help="use expert data or not. default is expert")
parser.add_argument("--epochs", type=int, default=61, help="number of epochs to train for. default is 61")
parser.add_argument("--name", type=str, required=True, help="mention what is being tested in this run. This is the tag used in W&B to identify this run. required")

parser.add_argument('--comment', type=str, default="", help="any additional comments. This will show up in W&B")
parser.add_argument('--data_aug_method', type=str, default="", choices=["gaussian", "uniform", "amplify",
                                                                        "dimdropout", "statemixup"])
args = parser.parse_args()
task = args.task
if not task in ["push", "lift"]:
    raise NotImplementedError()
auto_run_robot = args.auto_run_robot
auto_run_robot_interval = args.auto_run_robot_interval
data_type_str = args.dataset_type
use_expert_data = data_type_str == "expert"
n_epochs = args.epochs
name = args.name
comment = args.comment
aug_times = 1
aug_method = args.data_aug_method

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"using {device}")

if task == "push":
    rdl = PushTaskDataLoader(expert=use_expert_data)
    obs_used = obs_used_push
    history_size = history_size_push
    nsteps = nsteps_push
elif task == "lift":
    rdl = LiftTaskDataLoader(expert=use_expert_data)
    obs_used = obs_used_lift
    history_size = history_size_lift
    nsteps = nsteps_lift


wandb.init(project=f"rrc22-{task}-{data_type_str}", entity="srl_ethz", name=name, notes=comment)
wandb.config.update({"task": task,
                     "n_epochs": n_epochs,
                     "history_size": history_size,
                     "nsteps": nsteps,
                     "auto_run_robot": auto_run_robot,
                     "use_expert_data": use_expert_data,
                     "data_aug_method": aug_method})

obs_dims = len(rdl.get_observation_indices(obs_used))
print(obs_dims)
act_dims = 9
expanded_state_dim = obs_dims * history_size + act_dims * (history_size - 1)

# generate dataset
train_episodes, test_episodes = generate_dataset(args, rdl, obs_used, obs_dims, act_dims, expanded_state_dim, history_size)

iql = IQL(use_gpu=use_cuda, batch_size=64, scaler="standard", action_scaler="min_max", n_steps=nsteps, actor_learning_rate=0.00015, critic_learning_rate=0.00015)
# uncomment and use if re-starting training
# need preview version (1.12.1 or newer) of torch for loading like this...
# https://github.com/pytorch/pytorch/issues/80809
# iql = IQL.from_json("params.json", use_gpu=use_cuda)
# iql.load_model("iql.pt")

# this next variable must be changed if algorithm is changed from IQL!
algorithm_name = "IQL"

# remove old log directory
if os.path.exists("d3rlpy_logs/IQL"):
    shutil.rmtree("d3rlpy_logs/IQL")


def auto_evaluate(epoch):
    """
    run the current policy on robot and log the results to weights and biases
    """
    job_id, results, commit_hash = automated_submission()
    if results is None:
        # detect failure and make sure that user can see it in wandb as 0 return and 0 success
        return_mean = 0
        success_rate = 0
    else:
        return_mean = results["statistics"]["return_mean"]
        success_rate = results["statistics"]["success_rate"]

    wandb.log({"epoch": epoch,
               "return_mean": return_mean,
               "success_rate": success_rate,
               "result_url": wandb.Html(f'<a href="{base_url}/{job_id}">{job_id}</a>'),
               "github_link": wandb.Html(f'<a href="https://github.com/srl-ethz/RRC22/commit/{commit_hash}">{commit_hash}</a>'),
               "full_results": results})


threads = []

# use the iterative version of iql.fit so that metric can be acquired and logged to wandb
for epoch, metrics in iql.fitter(train_episodes,
                                 n_epochs=n_epochs,
                                 with_timestamp=False,
                                 eval_episodes=test_episodes,
                                 scorers={
                                          'td_error': td_error_scorer,
                                          'value_scale': average_value_estimation_scorer}
                                ):
    print(f"epoch: {epoch}, metrics: {metrics}")
    wandb.log({"epoch": epoch,
               "metrics": metrics})
    if not auto_run_robot:
        continue
    if epoch % auto_run_robot_interval == 0 or epoch == n_epochs:
        # evaluate on robot. But first, 
        # find the latest model in d3rlpy_logs using the filename
        model_files = os.listdir(f"d3rlpy_logs/{algorithm_name}")
        model_files = [f for f in model_files if (f.startswith("model_") and f.endswith(".pt"))]
        # find model_file with the largest epoch number
        model_file = sorted(model_files, key=lambda x: int(x.split("_")[1][:-3]))[-1]
        print(f"evaluating on real robot with model {model_file} and config file")
        # copy model_file
        shutil.copy(f"d3rlpy_logs/{algorithm_name}/{model_file}", f"trained_models/{task}_{data_type_str}.pt")
        shutil.copy(f"d3rlpy_logs/{algorithm_name}/params.json", f"trained_models/{task}_{data_type_str}.json")
        # make thread that calls auto_evaluate
        t = Thread(target=auto_evaluate, args=(epoch,))
        threads.append(t)
        t.start()

iql.save_model("iql.pt")
iql.save_policy("iql_policy.pt")

# Make a copy of weights and other params
wandb.save("iql.pt")
wandb.save("iql_policy.pt")

if os.path.isfile("params.json"):
    wandb.save("params.json")

wandb.config.update(iql.get_params(deep=True))

# Add all your parameters here to save to wandb logs

print("making sure all threads running real robot evaluation are done...")
for thread in threads:
    thread.join()
print("done")
