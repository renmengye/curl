"""
Aggregate performance of multiple runs of a model for a specific environment.

First argument is path to environment (e.g, "./tmp/cartpole")
Second argument is the model name: either "curl_sac" or "curl_oupn".
Third argument is batch size.
Fourth argument is number of environment steps.
"""

import os
import sys

import numpy as np

path = sys.argv[1]
model = sys.argv[2]
batch_size = sys.argv[3]
env_steps = sys.argv[4]

rewards = []
for d in os.listdir(path):
    d_path = os.path.join(path, d)
    if not os.path.isdir(d_path):
        continue
    if not ("b" + batch_size in d_path and "nes" + env_steps in d_path and model in d_path):
        continue
    eval_path = os.path.join(d_path, "eval.log")
    assert os.path.isfile(eval_path)
    with open(eval_path, "r") as f:
        lines = f.readlines()
        reward = float(lines[-1].strip().split(",")[1].split(":")[-1])
        rewards.append(reward)

assert len(rewards) == 5

print("Reward: {:.0f} +/- {:.0f}".format(np.round(np.mean(rewards)), np.round(np.std(rewards) / np.sqrt(len(rewards)))))
