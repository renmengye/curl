import torch
import numpy as np
import torch.nn as nn
import gym
import os
from collections import deque
import random
from torch.utils.data import Dataset, DataLoader
import time
from skimage.util.shape import view_as_windows


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2 ** bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2 ** (8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        capacity,
        batch_size,
        device,
        image_size=84,
        transform=None,
        center_crop_anchor=True,
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform
        self.center_crop_anchor = center_crop_anchor
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        if self.full:
            self.obses[0:-1] = self.obses[1:]
            self.actions[0:-1] = self.actions[1:]
            self.rewards[0:-1] = self.rewards[1:]
            self.next_obses[0:-1] = self.next_obses[1:]
            self.not_dones[0:-1] = self.not_dones[1:]

            np.copyto(self.obses[-1], obs)
            np.copyto(self.actions[-1], action)
            np.copyto(self.rewards[-1], reward)
            np.copyto(self.next_obses[-1], next_obs)
            np.copyto(self.not_dones[-1], not done)
        else:
            np.copyto(self.obses[self.idx], obs)
            np.copyto(self.actions[self.idx], action)
            np.copyto(self.rewards[self.idx], reward)
            np.copyto(self.next_obses[self.idx], next_obs)
            np.copyto(self.not_dones[self.idx], not done)

            self.idx = (self.idx + 1) % self.capacity
            self.full = self.full or self.idx == 0

    def sample_proprio(self):
        # idxs = np.random.randint(
        #     0, self.capacity if self.full else self.idx, size=self.batch_size
        # )
        obses = self.obses
        next_obses = self.next_obses

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions, device=self.device)
        rewards = torch.as_tensor(self.rewards, device=self.device)
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones, device=self.device)
        return obses, actions, rewards, next_obses, not_dones

    def sample_cpc(self):
        start = time.time()
        # idxs = np.random.randint(
        #     0, self.capacity if self.full else self.idx, size=self.batch_size
        # )
        obses = self.obses
        next_obses = self.next_obses
        pos = obses.copy()

        # Instead of augmenting both anchor and positives, only augment positives
        # For anchor, use center crop
        if self.center_crop_anchor:
            obses = center_crop_image(obses, self.image_size)
            next_obses = center_crop_image(next_obses, self.image_size)
        else:
            obses = random_crop(obses, self.image_size)
            next_obses = random_crop(next_obses, self.image_size)
        pos = random_crop(pos, self.image_size)

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions, device=self.device)
        rewards = torch.as_tensor(self.rewards, device=self.device)
        not_dones = torch.as_tensor(self.not_dones, device=self.device)

        pos = torch.as_tensor(pos, device=self.device).float()
        cpc_kwargs = dict(
            obs_anchor=obses, obs_pos=pos, time_anchor=None, time_pos=None
        )

        return obses, actions, rewards, next_obses, not_dones, cpc_kwargs

    def __getitem__(self, idx):
        # idx = np.random.randint(0, self.capacity if self.full else self.idx, size=1)
        # idx = idx[0]
        # obs = self.obses[idx]
        # action = self.actions[idx]
        # reward = self.rewards[idx]
        # next_obs = self.next_obses[idx]
        # not_done = self.not_dones[idx]

        # if self.transform:
        #     obs = self.transform(obs)
        #     next_obs = self.transform(next_obs)

        # return obs, action, reward, next_obs, not_done
        raise NotImplementedError("__getitem__ in replay buffer should not be used")

    def __len__(self):
        return self.capacity


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype,
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


def random_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(imgs, (1, output_size, output_size, 1))[..., 0, :, :, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs


def center_crop_image(image, output_size):
    new_h, new_w = output_size, output_size
    h, w = image.shape[-2:]
    top = (h - new_h) // 2
    left = (w - new_w) // 2

    if len(image.shape) == 3:
        image = image[:, top : top + new_h, left : left + new_w]
    elif len(image.shape) == 4:
        image = image[:, :, top : top + new_h, left : left + new_w]
    else:
        raise ValueError("Unknown image dimensions")

    return image
