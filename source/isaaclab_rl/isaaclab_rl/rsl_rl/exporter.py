# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import torch
import torch.nn as nn


def export_policy_as_jit(policy: object, normalizer: object | None, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporter(policy, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, policy, normalizer=None):
        super().__init__()
        self.is_recurrent = policy.is_recurrent
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.forward = self.forward_lstm
            self.reset = self.reset_memory
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x):
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward(self, x):
        return self.actor(self.normalizer(x))

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(nn.Module):
    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = getattr(policy, "is_recurrent", False)
        self.policy = copy.deepcopy(policy)
        self.policy.eval()
        if normalizer is not None:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = nn.Identity()

    def forward(self, obs):
        obs = self.normalizer(obs)
        actions = self.policy.act_inference(obs)
        return actions

    def forward_lstm(self, obs, h_in, c_in):
        obs = self.normalizer(obs)
        features = self.policy.encode(obs)
        out, (h_out, c_out) = self.policy.memory_a.rnn(features.unsqueeze(0), (h_in, c_in))
        out = out.squeeze(0)
        actions = self.policy.actor(out)
        return actions, h_out, c_out

    def export(self, path, filename="policy.onnx"):
        self.to("cpu")
        self.eval()
        obs_dim = (self.policy.depth_dim + self.policy.goal_dim + self.policy.vel_dim + self.policy.grav_dim)
        save_path = os.path.join(path, filename)
        if self.is_recurrent:
            self.forward = self.forward_lstm
            hidden_size = self.policy.memory_a.rnn.hidden_size
            num_layers = self.policy.memory_a.rnn.num_layers
            obs = torch.zeros(1, obs_dim, dtype=torch.float32)
            h_in = torch.zeros(num_layers, 1, hidden_size, dtype=torch.float32)
            c_in = torch.zeros(num_layers, 1, hidden_size, dtype=torch.float32)
            torch.onnx.export(self, (obs, h_in, c_in), save_path, export_params=True, opset_version=17, verbose=self.verbose, input_names=["obs", "h_in", "c_in"], output_names=["actions", "h_out", "c_out"], dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}, "h_in": {1: "batch"}, "c_in": {1: "batch"}, "h_out": {1: "batch"}, "c_out": {1: "batch"}})
        else:
            obs = torch.zeros(1, obs_dim, dtype=torch.float32)
            torch.onnx.export(self, obs, save_path, export_params=True, opset_version=17, verbose=self.verbose, input_names=["obs"], output_names=["actions"], dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}})
        print(f"Exported ONNX policy -> {save_path}")