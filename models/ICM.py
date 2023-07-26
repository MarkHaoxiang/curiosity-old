from __future__ import annotations

from typing import Optional

import torch
from torch.nn import Module
from torchrl.envs.transforms import Transform
from tensordict.utils import NestedKey
from tensordict import TensorDictBase, TensorDict
from torchrl.objectives.common import LossModule


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
    """

    def __init__(self,
                 feature_model: Module,
                 forward_model: Module,
                 reward_key: NestedKey = "reward",
                 action_key: NestedKey = "action",
                 observation_key: NestedKey = "observation",
                 out_key: Optional[NestedKey] = None):
        self._feature_model = feature_model
        self._forward_model = forward_model
        self._reward_key = reward_key
        self._action_key = action_key
        self._observation_key = observation_key
        self.out_key = self._reward_key if out_key is None else out_key

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
        tensordict['next'].set(self.out_key, r)
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

class IntrinsicCuriosityLoss(LossModule):
    """ Intrinsic Curiosity Model
    
    Use in conjunction with an environment transformed by IntrinsicCuriosityReward    
    Trains the forward model and the inverse model
    """
    def __init__(self,
                 forward_model: Module,
                 inverse_model: Module,
                 feature_model: Module):
        self.forward_model = forward_model
        self.inverse_model = inverse_model
        self.feature_model = feature_model # TODO: Feature model should really be a part of inverse model. 
                                           # Will refactor on the TensorDict Module refactor
        

    def _forward(self, tensordict: TensorDictBase):
        # Inverse loss
        s_0 = tensordict[("ICM", "s_0")]
        s_1 = tensordict[("ICM", "s_1")]
        a_0 = tensordict[("ICM", "a_0")]
        loss_inverse = self.inverse_model(torch.cat((s_0, s_1), dim=-1)) - a_0

        phi_0 = self.feature_model(s_0)
        loss_forward = self.forward_model(torch.cat(a_0, phi_0), dim=-1)

        return TensorDict(
            source={
                "loss_inverse": loss_inverse.mean(),
                "loss_forward": loss_forward.mean()
            }
        )
