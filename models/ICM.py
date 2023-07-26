from __future__ import annotations

from typing import Optional

import torch
from torch.nn import Module
from torchrl.data.tensor_specs import TensorSpec
from torchrl.envs.transforms import Transform
from tensordict.utils import NestedKey
from tensordict import TensorDictBase, TensorDict
from torchrl.objectives.common import LossModule
from torchrl.data.tensor_specs import CompositeSpec, UnboundedContinuousTensorSpec


# TODO
# Different prediction errors between prediceted and true encoded states
#   ie. Mean Squared etc rather than just absolute
# 
# Support observation space featurization
#
# Explicit reward weighting
#
# out_key
#
# Use TensorDict Modules instead of nn modules for feature, forward and inverse
#
# Debug and logging information in ICM loss

class IntrinsicCuriosityReward(Transform):
    """ Intrinsic Curiosity Model
    https://arxiv.org/pdf/1705.05363.pdf

    Adds intrinsic curiosity reward to the reward value of an environment

    Args:
        feature_model: nn.module = Encodes s_t and s_t+1
        forward_model: nn.module = Given the current action and encoded state, 
            predict the next encoded state
        inverse_model: nn.module = From the encodings of both states to the intervening action.
        reward_key
        action_key
        observation_key
        out_key = Where put the new, combined reward
    """

    def __init__(self,
                 feature_model: Module,
                 forward_model: Module,
                 encoding_size: int,
                 n_agents: int,
                 reward_key: NestedKey = "reward",
                 action_key: NestedKey = "action",
                 observation_key: NestedKey = "observation",
                 out_key: Optional[NestedKey] = None):
        out_key = reward_key if out_key is None else out_key
        super().__init__(in_keys=[reward_key, action_key, observation_key], out_keys=[out_key])
        self._feature_model = feature_model
        self._forward_model = forward_model
        self._reward_key = reward_key
        self._action_key = action_key
        self._observation_key = observation_key
        self._out_key = out_key
        self._encoding_size = encoding_size
        self._n_agents = n_agents

        self.previous_state = None

    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Resets a tranform if it is stateful."""
        self.previous_state = None
        return tensordict

    def _step(self, tensordict: TensorDictBase):
        if self.previous_state == None:
            self.previous_state = tensordict
            return tensordict

        a_0 = tensordict[self._action_key]
        s_0, s_1 = self.previous_state[self._observation_key], tensordict['next'][self._observation_key]
        r = tensordict['next'][self._reward_key]

        with torch.no_grad():
            # True encoded states
            phi_0 = self._feature_model(s_0)
            phi_1 = self._feature_model(s_1)
            phi_1_pred = self._forward_model(torch.cat((a_0, phi_0), dim=-1))

        # Set the new rewards
        r = r + torch.linalg.vector_norm(phi_1-phi_1_pred)
        tensordict['next'].set(self._out_key, r)
        self.previous_state = tensordict

        # Set the training targets for IntrinsicCuriosityLoss
        # And for debugging
        tensordict['next'].set(("ICM", "a_0"), a_0)
        tensordict['next'].set(("ICM", "s_0"), s_0)
        tensordict['next'].set(("ICM", "s_1"), s_1)
        tensordict['next'].set(("ICM", "phi_0"), phi_0)
        tensordict['next'].set(("ICM", "phi_1"), phi_1)

        return tensordict

    def _call(self, tensordict: TensorDictBase):
        # Called in reset
        return tensordict
    
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        icm_spec = {}

        phi_0 = UnboundedContinuousTensorSpec(
            shape = (*self.parent.batch_size, self._n_agents, self._encoding_size),
            device = self.parent.device
        )
        phi_1 = UnboundedContinuousTensorSpec(
            shape = (*self.parent.batch_size, self._n_agents, self._encoding_size),
            device = self.parent.device
        )
        icm_spec.update({
            "ICM":CompositeSpec(
                a_0 = self.parent.action_spec,
                s_0 = observation_spec.clone(),
                s_1 = observation_spec.clone(),
                phi_0 = phi_0,
                phi_1 = phi_1,
                shape = self.parent.batch_size
            )
        })

        # Add new specs
        if not isinstance(observation_spec, CompositeSpec):
            observation_spec = CompositeSpec(
                observation=observation_spec, shape=self.parent.batch_size
            )
        observation_spec.update(icm_spec)

        return observation_spec

class IntrinsicCuriosityLoss(LossModule):
    """ Intrinsic Curiosity Model
    
    Use in conjunction with an environment transformed by IntrinsicCuriosityReward    
    Trains the forward model and the inverse model
    """
    def __init__(self,
                 forward_model: Module,
                 inverse_model: Module,
                 feature_model: Module):
        super().__init__()

        self.convert_to_functional(
            forward_model,
            "forward_model",
            create_target_params=False,
        )

        self.convert_to_functional(
            inverse_model,
            "inverse_model",
            create_target_params=False,
        )
        self.convert_to_functional(
            feature_model,
            "feature_model",
            create_target_params=False,
        )

        self.forward_model = forward_model
        self.inverse_model = inverse_model
        self.feature_model = feature_model

    def forward(self, tensordict: TensorDictBase):
        # Inverse loss
        # print(tensordict["ICM", "s_0", "agents", "observation"].shape)
        s_0 = tensordict[("ICM", "s_0", "agents", "observation")]
        s_1 = tensordict[("ICM", "s_1", "agents", "observation")] # TODO (bug)
        a_0 = tensordict[("ICM", "a_0")]
        loss_inverse = self.inverse_model(s_0, s_1) - a_0

        phi_0 = self.feature_model(s_0)
        loss_forward = self.forward_model(torch.cat((a_0, phi_0),dim=-1))

        return TensorDict(
            source={
                "loss_inverse": loss_inverse.mean(),
                "loss_forward": loss_forward.mean()
            },
            batch_size=[]
        )


class InverseModel(Module):
    def __init__(self,
                 feature_network,
                 inverse_network):
        super().__init__()
        self._feature_network = feature_network
        self._inverse_network = inverse_network

    def forward(self, s_0, s_1):
        phi_0 = self._feature_network(s_0)
        phi_1 = self._feature_network(s_1)
        a     = self._inverse_network(torch.cat((phi_0, phi_1), dim=-1))
        return a
