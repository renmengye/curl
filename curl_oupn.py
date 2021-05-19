import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os

import utils
from encoder import make_encoder
from curl_sac import gaussian_logprob, squash, weight_init, Actor, QFunction, Critic

LOG_FREQ = 10000


class PrototypeMemory(nn.Module):
    # NOTE: Prototype memory assumes the batch size is 1
    def __init__(self, capacity, device, thresh, decay):
        super(PrototypeMemory, self).__init__()
        self.capacity = capacity
        self.device = device
        self.thresh = thresh
        self.decay = decay
        self.prototypes = None  # [n_prototypes x d]
        self.usages = None  # [n_prototypes x 1]
        self.beta = nn.Parameter(
            torch.tensor(
                [-12.0], dtype=torch.float32, device=self.device, requires_grad=True
            )
        )
        self.gamma = nn.Parameter(
            torch.tensor(
                [1.0], dtype=torch.float32, device=self.device, requires_grad=True
            )
        )
        self.temp = nn.Parameter(
            torch.tensor(
                [10.0], dtype=torch.float32, device=self.device, requires_grad=True
            )
        )
        self.cross_entropy = nn.CrossEntropyLoss()

    def reset(self):
        self.prototypes = None
        self.usages = None

    def detach_prototypes(self):
        self.prototypes = self.prototypes.detach()
        self.usages = self.usages.detach()

    def forward(self, z):
        # z: [1, d]
        # If no prototypes exist, automatically add
        first_proto = False
        if self.prototypes is None:
            self.prototypes = z.clone()
            self.usages = torch.tensor(
                [[1.0]], dtype=torch.float32, device=self.device, requires_grad=False
            )
            first_proto = True

        logits = self.compute_logits(z)
        u = torch.sigmoid((-torch.max(logits).detach() - self.beta) / self.gamma)

        if first_proto:
            label = torch.argmax(logits, dim=1)
            return self.cross_entropy(logits, label), label, u

        # Add new prototype
        if u >= self.thresh:
            assert self.prototypes.size(0) <= self.capacity
            if self.prototypes.size(0) == self.capacity:
                idx = torch.argmin(self.usages, dim=0)
                self.prototypes = torch.cat(
                    [self.prototypes[:idx], self.prototypes[idx + 1 :], z.clone()],
                    dim=0,
                )
                self.usages = torch.cat(
                    [
                        self.usages[:idx],
                        self.usages[idx + 1 :],
                        torch.tensor(
                            [[1.0]],
                            dtype=torch.float32,
                            device=self.device,
                            requires_grad=False,
                        ),
                    ],
                    dim=0,
                )
            else:
                self.prototypes = torch.cat([self.prototypes, z.clone()], dim=0)
                self.usages = torch.cat(
                    [
                        self.usages,
                        torch.tensor(
                            [[1.0]],
                            dtype=torch.float32,
                            device=self.device,
                            requires_grad=False,
                        ),
                    ],
                    dim=0,
                )
        # Dense update to existing prototypes
        else:
            y = F.softmax(logits, dim=1)
            delta = torch.transpose(y * (1.0 - u), 0, 1)
            self.prototypes = z * (delta / (delta + self.usages)) + self.prototypes * (
                self.usages / (self.usages + delta)
            )
            self.usages = self.usages + delta

        # Decay all prototype usages
        self.usages = self.usages * self.decay

        logits = self.compute_logits(z)
        label = torch.argmax(logits, dim=1)
        return self.cross_entropy(logits, label), label, u

    def compute_logits(self, z):
        prototypes = self.prototypes / torch.norm(
            self.prototypes, p=2, dim=1, keepdim=True
        )
        z = z / torch.norm(z, p=2, dim=1, keepdim=True)
        return self.temp * torch.matmul(z, torch.transpose(prototypes, 0, 1))


class CURL(nn.Module):
    """
    CURL
    """

    def __init__(
        self,
        obs_shape,
        z_dim,
        batch_size,
        critic,
        critic_target,
        device,
        mem_capacity,
        thresh,
        decay,
        output_type="continuous",
    ):
        super(CURL, self).__init__()
        self.batch_size = batch_size
        self.encoder = critic.encoder
        self.encoder_target = critic_target.encoder
        self.prototype_memory = PrototypeMemory(mem_capacity, device, thresh, decay)
        self.output_type = output_type

    def encode(self, x, detach=False, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_loss(self, z):
        return self.prototype_memory(z)

    def compute_logits(self, z):
        return self.prototype_memory.compute_logits(z)


class CurlOUPNAgent(object):
    """CURL representation learning with SAC."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type="pixel",
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        num_layers=4,
        num_filters=32,
        cpc_update_freq=1,
        log_interval=100,
        detach_encoder=False,
        curl_latent_dim=128,
        mem_capacity=128,
        lambda_ent=0.5,
        lambda_con=1.0,
        lambda_new=0.5,
        proto_thresh=0.5,
        usage_decay=0.995,
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.curl_latent_dim = curl_latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type
        self.mem_capacity = mem_capacity
        self.lambda_ent = lambda_ent
        self.lambda_con = lambda_con
        self.lambda_new = lambda_new
        self.proto_thresh = proto_thresh
        self.usage_decay = usage_decay

        self.actor = Actor(
            obs_shape,
            action_shape,
            hidden_dim,
            encoder_type,
            encoder_feature_dim,
            actor_log_std_min,
            actor_log_std_max,
            num_layers,
            num_filters,
        ).to(device)

        self.critic = Critic(
            obs_shape,
            action_shape,
            hidden_dim,
            encoder_type,
            encoder_feature_dim,
            num_layers,
            num_filters,
        ).to(device)

        self.critic_target = Critic(
            obs_shape,
            action_shape,
            hidden_dim,
            encoder_type,
            encoder_feature_dim,
            num_layers,
            num_filters,
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic, and CURL and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        if self.encoder_type == "pixel":
            self.CURL = CURL(
                obs_shape,
                encoder_feature_dim,
                self.curl_latent_dim,
                self.critic,
                self.critic_target,
                self.device,
                self.mem_capacity,
                self.proto_thresh,
                self.usage_decay,
                output_type="continuous",
            ).to(self.device)

            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

            self.oupn_optimizer = torch.optim.Adam(
                self.CURL.prototype_memory.parameters(), lr=encoder_lr
            )
        self.cross_entropy = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder_type == "pixel":
            self.CURL.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)

        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(
            obs, action, detach_encoder=self.detach_encoder
        )
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        # if step % self.log_interval == 0:
        L.log("train_critic/loss", critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        # if step % self.log_interval == 0:
        L.log("train_actor/loss", actor_loss, step)
        L.log("train_actor/target_entropy", self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(
            dim=-1
        )
        # if step % self.log_interval == 0:
        L.log("train_actor/entropy", entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
        # if step % self.log_interval == 0:
        L.log("train_alpha/loss", alpha_loss, step)
        L.log("train_alpha/value", self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_oupn(self, obs_anchor, obs_pos, cpc_kwargs, L, step):
        z_a = self.CURL.encode(obs_anchor)
        z_pos = self.CURL.encode(obs_pos)
        T = z_a.size(0)

        u_vals = []
        running_u = torch.zeros(
            1, dtype=torch.float32, device=self.device, requires_grad=True
        )
        l_ent = torch.zeros(
            1, dtype=torch.float32, device=self.device, requires_grad=True
        )
        l_con = torch.zeros(
            1, dtype=torch.float32, device=self.device, requires_grad=True
        )
        for i in range(T):
            z_ai = z_a[i]
            l_ent_t, label, u = self.CURL.compute_loss(z_ai.view(1, z_a.size(1)))
            l_ent = l_ent + l_ent_t
            u_vals.append(u.item())
            running_u = running_u + u
            z_posi = z_pos[i]
            pos_logits = self.CURL.compute_logits(z_posi.view(1, z_pos.size(1)))
            l_con = l_con + self.cross_entropy(pos_logits, label)

        l_ent = self.lambda_ent * (l_ent / T)
        l_con = self.lambda_con * (l_con / T)
        l_new = self.lambda_new * (
            torch.abs((running_u / T) - self.CURL.prototype_memory.thresh)
        )
        l = l_ent + l_con + l_new

        self.oupn_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        l.backward()
        self.oupn_optimizer.step()
        self.encoder_optimizer.step()

        self.CURL.prototype_memory.detach_prototypes()

        # if step % self.log_interval == 0:
        L.log("train/ent_loss", l_ent, step)
        L.log("train/con_loss", l_con, step)
        L.log("train/new_loss", l_new, step)
        L.log("train/min_new_prob", np.amin(u_vals), step)
        L.log("train/max_new_prob", np.amax(u_vals), step)

    def update(self, replay_buffer, L, step):
        if self.encoder_type == "pixel":
            (
                obs,
                action,
                reward,
                next_obs,
                not_done,
                cpc_kwargs,
            ) = replay_buffer.sample_cpc()
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio()

        # if step % self.log_interval == 0:
        L.log("train/batch_reward", reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder, self.encoder_tau
            )

        if step % self.cpc_update_freq == 0 and self.encoder_type == "pixel":
            obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
            self.update_oupn(obs_anchor, obs_pos, cpc_kwargs, L, step)

    def save(self, model_dir, step):
        for root, _, files in os.walk(model_dir):
            for f in files:
                if (
                    f.startswith("actor")
                    or f.startswith("critic")
                    or f.startswith("curl")
                ) and f.endswith(".pt"):
                    os.remove(os.path.join(root, f))
        torch.save(self.actor.state_dict(), "%s/actor_%s.pt" % (model_dir, step))
        torch.save(self.critic.state_dict(), "%s/critic_%s.pt" % (model_dir, step))
        torch.save(self.CURL.state_dict(), "%s/curl_%s.pt" % (model_dir, step))

    def load(self, model_dir, step):
        self.actor.load_state_dict(torch.load("%s/actor_%s.pt" % (model_dir, step)))
        self.critic.load_state_dict(torch.load("%s/critic_%s.pt" % (model_dir, step)))
        self.CURL.load_state_dict(torch.load("%s/curl_%s.pt" % (model_dir, step)))
