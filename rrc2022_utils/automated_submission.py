import subprocess
import os
import json
from time import sleep
import re
import requests
from random import randint
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import uuid

"""
Various functions to automate the submission of the current code to the real robot cluster

Partly based on
https://github.com/rr-learning/rrc2022/blob/master/scripts/automated_submission.sh
Some memos and considerations:
https://docs.google.com/document/d/1eS-X1JYQBSbqfTJFiq_MlhdrWvsPuSYEeb2HzDb9srE/edit?usp=sharing

prepare credentials.txt file in this directory with first row ID, second row password, third row your email, fourth row slack token
"""

# read the ID, PW, and email from credentials.txt
slack_token = None
# get current directory
script_dir = os.path.dirname(os.path.realpath(__file__))
with open(f'{script_dir}/credentials.txt', "r") as f:
    lines = f.readlines()
    username = lines[0].strip()
    password = lines[1].strip()
    email = lines[2].strip()
    try:
        slack_token = lines[3].strip()
    except IndexError:
        # slack credentials are optional
        pass

# URL to the webserver at which the recorded data can be downloaded
base_url = f"https://robots.real-robot-challenge.com/output/{username}/data"
remote_url = f"{username}@robots.real-robot-challenge.com"
repo_url = "git@github.com:srl-ethz/RRC22.git"
# top of repository is one level above this script
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# this folder will be used for the auto-submission system, instead of the usual .git folder
git_auto_foldername = ".git_auto"
# use random hash for branch name so it won't clash with existing branches
# start with z so it will show up at bottom of GitHub branch list for convenience
branch_name = f"z_auto_sub_{uuid.uuid4().hex}"

def curl_check_if_exists(job_id):
    """
    check if the file exists in the server (i.e. if the task is done)
    """
    url = f"{base_url}/{job_id}/results.json"
    cmd = f"curl -sI --user {username}:{password} -o /dev/null -w '%{{http_code}}' {url}"
    output = subprocess.check_output(cmd, shell=True)
    return output in [b'200', b'301']


def download_results(job_id):
    """
    download the data from the server, save it to results.json, and return the dict
    returns None for results if the file doesn't exist
    """
    print(f"downloading results for job {job_id}")
    # download the data
    url = f"{base_url}/{job_id}/results.json"
    cmd = f"curl -u {username}:{password} -o /tmp/results.json {url}"
    subprocess.run(cmd, shell=True)
    # load the json file
    with open(f"/tmp/results.json", "r") as f:
        try:
            results = json.load(f)
        except json.decoder.JSONDecodeError:
            print(f"results file for job {job_id} is empty, probably failed")
            results = None
    return results


def check_if_best(latest_job_id, latest_results):
    """
    check if the current results are the best so far
    returns True if it is, False otherwise
    """
    print(f"checking if job {latest_job_id} is the best so far")
    if latest_results is None:
        return False
    # get all directories that exist in base_url
    page_text = requests.get(base_url, auth=(username, password)).text
    job_ids = re.findall(r'/">([0-9]+)/</a>', page_text)
    for job_id in job_ids:
        if job_id == latest_job_id:
            continue
        # download the results
        results = download_results(job_id)
        if results is None:
            # this job_id failed
            continue
        if results['task'] != latest_results['task']:
            # task type is different
            continue
        if results['dataset_type'] != latest_results['dataset_type']:
            # dataset type is different
            continue

        # check if the latest results are better than the current results
        if results['statistics']['return_mean'] > latest_results['statistics']['return_mean']:
            return False
    return True


def submit_job():
    """
    construct roboch.json file and send the command start evaluation on robot
    (assumes that the branch is already pushed)
    returns None if the job is not accepted, otherwise returns the job_id
    """
    print("submitting job...")
    # construct the roboch.json file and send to robot
    roboch_json = {"repository": repo_url,
                   "branch": branch_name,
                   "email": email,
                   "git_deploy_key": "id_rsa_jealousjaguar"}
    # write json to file
    with open(f"{repo_dir}/roboch.json", "w") as f:
        json.dump(roboch_json, f)
    subprocess.run(f"sshpass -p {password} scp roboch.json {remote_url}:", cwd=repo_dir, shell=True)

    output = subprocess.check_output(f"sshpass -p {password} ssh -T {remote_url} <<<submit", shell=True, executable='/bin/bash')
    output = output.decode('utf-8')  # to string
    # extract the job_id from output string
    if job_id_re_result := re.search("to cluster ([0-9]+).", output):
        job_id = job_id_re_result.group(1)
    else:
        print(f"job not found in output [{output}], probably tried to do concurrent submissions...")
        return None
    job_id = int(job_id)
    print(f"job submitted with ID {job_id}")

    return job_id


# set the environment variable so that git will use the other folder instead of .git
env_var = os.environ.copy()
env_var['GIT_DIR'] = f'./{git_auto_foldername}'
env_var['GIT_WORK_TREE'] = './'


def publish_current_code_to_branch():
    """
    publish the current state of the repo to the _auto_sub branch
    returns commit hash
    very hackish script
    will only work if that branch doesn't exist
    """
    print("publishing current code to branch")
    # First, create a new git repo in the current one with a separate git folder
    # (this is to avoid messing up the original repo)

    # delete the folder if it already exists
    subprocess.run(f"rm -rf {git_auto_foldername}", cwd=repo_dir, shell=True)
    # create the folder at the top of the repo
    subprocess.run(f"mkdir {git_auto_foldername}", cwd=repo_dir, shell=True)
    # initialize the repo
    subprocess.run("git init", cwd=repo_dir, shell=True, env=env_var)

    # then, add all files to the repo and push to custom branch
    # add all files
    subprocess.run("git add .", cwd=repo_dir, shell=True, env=env_var)

    subprocess.run(f"git remote add origin {repo_url}", cwd=repo_dir, shell=True, env=env_var)

    subprocess.run("git commit -m 'auto-submission'", cwd=repo_dir, shell=True, env=env_var)
    subprocess.run(f"git checkout -b {branch_name}", cwd=repo_dir, shell=True, env=env_var)
    # push to the branch
    subprocess.run(f"git push -f --set-upstream origin {branch_name}", cwd=repo_dir, shell=True, env=env_var)
    commit_hash = subprocess.check_output("git rev-parse HEAD", cwd=repo_dir, shell=True, env=env_var).decode('utf-8').strip()
    return commit_hash


def wait_until_robot_available():
    """
    wait until all jobs are gone from the robot and it can accept a new job
    """
    print("waiting until robot is available...")
    # if the robot is available from the very beginning
    available_from_start_flag = True
    while True:
        try:
            cmd = f"sshpass -p {password} ssh -T {remote_url} <<<status"
            output = subprocess.check_output(cmd, cwd=repo_dir, shell=True, executable='/bin/bash')
            output = output.decode('utf-8')  # to string
            if re.search("0 jobs; 0 completed, 0 removed, 0 idle, 0 running, 0 held, 0 suspended", output):
                # no jobs
                break
        except subprocess.CalledProcessError as e:
            # maybe rare, but had some issues with: "kex_exchange_identification: read: Connection reset by peer"
            # this would at least make it try again without killing the whole process
            print(f"error while checking status: {e}")
        sleep(randint(20, 40))
        available_from_start_flag = False
    if not available_from_start_flag:
        # after robot status is changed to become available, wait a minute as per the recommendations
        sleep(60)
    print("robot is available")


def post_message_to_slack(text):
    """
    Post message to Slack with JaguarBot
    https://api.slack.com/apps/A0417K7F1K4/general
    """
    print(f"posting message to slack: {text}")
    if slack_token is None:
        return
    client = WebClient(token=slack_token)
    try:
        response = client.chat_postMessage(
            channel="real-robot-challenge-2022",
            text=text
        )
    except SlackApiError as e:
        print(e)


def automated_submission():
    """
    runs the code on the real robot cluster with the current state of the repo, and return the results
    (weights must be updated before running this function)
    returns job_id, results, and commit hash
    """
    print("running automated submission procedure...")
    # wait until robot is available
    # publish the current state of the repo to the branch
    commit_hash = publish_current_code_to_branch()
    wait_until_robot_available()
    while True:
        # submit job
        job_id = submit_job()
        if job_id is not None:
            # submission successful
            break
        # otherwise try again, since another program may be trying to submit a job
        # trying a weird way to make sure that at some point, one of the many concurrent programs trying to submit at the same time will be faster than the others and prevent race conditions from simultaneous submissions
        sleep(randint(20, 40))
        wait_until_robot_available()
    print(f"submitted job {job_id} to robot cluster")

    # wait until job is done
    sleep(60)  # it will definitely take more than 60 seconds
    while not curl_check_if_exists(job_id):
        # do not poll more often than one per minute
        sleep(60)
    # download the results
    results = download_results(job_id)
    # check if this is the best results so far
    is_best = check_if_best(job_id, results)

    if is_best:
        # post to slack channel
        task_type = results["task"]
        dataset_type = results["dataset_type"]
        return_mean = results["statistics"]["return_mean"]
        success_rate = results["statistics"]["success_rate"]
        slack_message = f"Congratulations to {email}! New best results for task:[{task_type}] on dataset:[{dataset_type}], with score [{return_mean}] and success rate [{success_rate}]. Check out the data at {base_url}/{job_id} ."
        post_message_to_slack(slack_message)

    return job_id, results, commit_hash