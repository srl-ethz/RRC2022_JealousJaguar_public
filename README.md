# RRC 2022 JealousJaguar team Training & Policies
This was the code used for the JealousJaguar team's submission to the 2022 Real Robot Challenge, which has achieved 3rd place in the competition.

![](/img/rrc.png)

Offline RL is the focus of the Real Robot Challenge for this year, a contest in the NeurIPS 2022 Competition Track. The two tasks to solve are the *push* task and *lift* task, and both policies must be learned from an offline dataset provided by the organizers. Although model-based manipulation methods based on motion primitives were the winner of the 2020 challenge and the runner-up for the 2021 challenge, the updated rules for this year precludes the use of such methods, and all of the policy must be learned from the offline dataset without using any hardcoded behavior. These rules, along with the real robot submission system, enables a unique competition where each team can focus on the challenge of solving offline RL on physical robots.

## Resources
* [Real Robot Challenge website](https://real-robot-challenge.com/)
* [JealousJaguar team report](https://drive.google.com/file/d/1Nx2QPvqlko5jvbc_2xwahnjvGAXFpLwO/view?usp=sharing)
* [Software documentation for the Challenge](https://webdav.tuebingen.mpg.de/real-robot-challenge/2022/docs/)

## Installation and setup

### Directory structure
* d3rlpy_based: The actual code for our training & policy are stored here. Read the README here for how to get started with the training and inference.
* rrc2022: sample policy from the organizers.
* rrc2022_utils: custom scripts by our team to conveniently load some parts of the data, evaluate policy automatically, etc.
* scripts: scripts from the organizers for validation and simple automated submission.

## prepare for the automatic submission system
![](/img/rrc_automated_submission.png)

As the training algorithm does not have access to the robot, it is difficult to evaluate the performance of the policy during the training. The loss curves alone are not very helpful in indicating how well the policy works when applied to the real robot. Therefore, we have developed an integrated pipeline, where at predefined intervals during the training, the policy is submitted to the real robot cluster, the results are received, and logged to an online platform.

```bash
# used in automated_submission.py
sudo apt install sshpass
```
in the rrc2022_utils folder, create `credentials.txt` file with four rows: Condor username, Condor password, your email address (you receive an email when the robot is done), and the slack token for the SlackBot(last one is optional).

before running training script, make that you can login to the remote robot server computer with `ssh jealousjaguar@robots.real-robot-challenge.com` (use the Condor password from credentials.txt)


## installing using venv and pip

```bash
# pin- pinocchio, the dynamics computation library was not found by pip for python3.10...
python3.9 -m venv ~/venv/rrc2022
source ~/venv/rrc2022/bin/activate
pip install -U pip
pip install git+https://github.com/rr-learning/rrc_2022_datasets.git@v1.1.0
pip install torch  # used in example.py
# used in automated_submission.py
pip install slack_sdk requests
```

below, the original README from the sample repo:
---


This is currently a clone of the official example package for the [Real Robot Challenge
2022](https://real-robot-challenge.com).  We can use it as base for your own
package when participating in the challenge.


Example Policies
----------------

The package contains two example policies for the pre-stage to show how your
package/code should be set up for running the evaluation.  You use them to test the
evaluation.

For the push task:

    $ python3 -m rrc_2022_datasets.evaluate_pre_stage push rrc2022.example.TorchPushPolicy --n-episodes=3 -v

For the lift task:

    $ python3 -m rrc_2022_datasets.evaluate_pre_stage lift rrc2022.example.TorchLiftPolicy --n-episodes=3 -v

The policy classes are implemented in `rrc2022/example.py`.  The corresponding torch
models are in `rrc2022/policies` and are installed as package_data so they can be loaded
at runtime (see `setup.cfg`).


Documentation
-------------

For more information, please see the [challenge
website](https://real-robot-challenge.com) and the [software
documentation](https://webdav.tuebingen.mpg.de/real-robot-challenge/2022/docs/).
