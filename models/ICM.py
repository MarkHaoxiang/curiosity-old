from __future__ import annotations
import torch

from torch.nn import Module
from torchrl.envs.transforms import Transform
from tensordict.utils import NestedKey
from tensordict import TensorDictBase


# TODO
# Different prediction errors between prediceted and true encoded states
#   ie. Mean Squared etc rather than just absolute (line 73)
# 
# Support observation space featurization
#
# Explicit reward weighting (line 73)
# out_key
#
# Write tests

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
                 observation_key: NestedKey = "observation"):
        super().__init__(in_keys=[reward_key, action_key, observation_key])
        self._feature_model = feature_model
        self._forward_model = forward_model
        self._reward_key = reward_key
        self._action_key = action_key
        self._observation_key = observation_key

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

        r = r + torch.linalg.vector_norm(phi_1-phi_1_pred)
        tensordict['next'].set(self._reward_key, r)
        self.previous_state = tensordict
        return tensordict
    
    def _call(self, tensordict: TensorDictBase):
        # Called in reset
        return tensordict
