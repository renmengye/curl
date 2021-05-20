"""
Plot loss and associated metrics throughout training.

First argument is path to directory for a specific training run
  (e.g., /home/tscott/Documents/curl/tmp/cartpole/cartpole-swingup-05-19-im84-b125-nes100000-s406565-pixel-curl_sac)
"""

from collections import defaultdict
import os
import sys

import matplotlib
import matplotlib.pyplot as plt


def curl_sac_plot(data, path, domain):
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
    ms = 2

    ax[0, 0].plot(data["episode"], data["actor_loss"], "-o", ms=ms)
    ax[0, 0].set_title("Average Actor Loss")
    ax[0, 0].set_ylabel("Loss")

    ax[0, 1].plot(data["episode"], data["critic_loss"], "-o", ms=ms)
    ax[0, 1].set_title("Average Critic Loss")
    ax[0, 1].set_ylabel("Loss")

    ax[1, 0].plot(data["episode"], data["curl_loss"], "-o", ms=ms)
    ax[1, 0].set_title("Average Contrastive Loss")
    ax[1, 0].set_xlabel("Train Episode")
    ax[1, 0].set_ylabel("Loss")

    ax[1, 1].plot(data["episode"], data["episode_reward"], "-o", ms=ms)
    ax[1, 1].set_title("Episode Reward")
    ax[1, 1].set_xlabel("Train Episode")
    ax[1, 1].set_ylabel("Reward")

    plt.suptitle("CURL: " + domain)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(path, "metrics.pdf"))


def curl_oupn_plot(data, path, domain):
    fig, ax = plt.subplots(3, 3, figsize=(12, 12), sharex=True)
    ms = 2

    ax[0, 0].plot(data["episode"], data["ent_loss"], "-o", ms=ms)
    ax[0, 0].set_title("Average Entropy Loss")
    # ax[0, 0].set_xlabel("Train Episode")
    ax[0, 0].set_ylabel("Loss")

    ax[0, 1].plot(data["episode"], data["con_loss"], "-o", ms=ms)
    ax[0, 1].set_title("Average Contrastive Loss")
    # ax[0, 1].set_xlabel("Train Episode")
    ax[0, 1].set_ylabel("Loss")

    ax[0, 2].plot(data["episode"], data["new_loss"], "-o", ms=ms)
    ax[0, 2].set_title("Average New Prototype Loss")
    # ax[0, 2].set_xlabel("Train Episode")
    ax[0, 2].set_ylabel("Loss")

    ax[1, 0].plot(data["episode"], data["actor_loss"], "-o", ms=ms)
    ax[1, 0].set_title("Average Actor Loss")
    # ax[1, 0].set_xlabel("Train Episode")
    ax[1, 0].set_ylabel("Loss")

    ax[1, 1].plot(data["episode"], data["critic_loss"], "-o", ms=ms)
    ax[1, 1].set_title("Average Critic Loss")
    # ax[1, 1].set_xlabel("Train Episode")
    ax[1, 1].set_ylabel("Loss")

    ax[2, 0].plot(data["episode"], data["min_new_prob"], "-o", ms=ms)
    ax[2, 0].set_title("Average Min Pr(New Prototype)")
    ax[2, 0].set_xlabel("Train Episode")
    ax[2, 0].set_ylabel("Probability")

    ax[2, 1].plot(data["episode"], data["max_new_prob"], "-o", ms=ms)
    ax[2, 1].set_title("Average Max Pr(New Prototype)")
    ax[2, 1].set_xlabel("Train Episode")
    ax[2, 1].set_ylabel("Probability")

    ax[2, 2].plot(data["episode"], data["episode_reward"], "-o", ms=ms)
    ax[2, 2].set_title("Episode Reward")
    ax[2, 2].set_xlabel("Train Episode")
    ax[2, 2].set_ylabel("Reward")

    plt.suptitle("OUPN: " + domain)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(path, "metrics.pdf"))


domain_name_map = {
    "cartpole": "Cartpole, Swing Up",
    "finger": "Finger, Spin",
    "cheetah": "Cheetah, Run",
    "reacher": "Reacher, Easy",
    "walker": "Walker, Walk",
    "ball_in_cup": "Ball in Cup, Catch",
}

path = sys.argv[1]
domain = domain_name_map[path.split("/")[1].lower()]

agg_data = defaultdict(list)

with open(os.path.join(path, "train.log"), "r") as f:
    lines = f.readlines()[1:-1]
    for line in lines:
        data = eval(line.strip())
        for k, v in data.items():
            agg_data[k].append(v)

if "curl_sac" in path:
    curl_sac_plot(agg_data, path, domain)
elif "curl_oupn" in path:
    curl_oupn_plot(agg_data, path, domain)
else:
    raise ValueError("Unknown model name. Expected curl_sac or curl_oupn.")
