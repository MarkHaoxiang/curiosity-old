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
from torchrl.data.replay_buffers import ReplayBuffer

class IntrinsicCuriosityModule(Transform):
    """ Intrinsic Curiosity Mdoule
    https://arxiv.org/pdf/1705.05363.pdf

    Adds intrinsic curiosity reward to the reward value of an environment
    """
    def __init__(
        self,
        feature_net: Module,
        inverse_net: Module,
        forward_net: Module,
        encoding_size: int = 64,
        reward_key: NestedKey = "reward",
        action_key: NestedKey = "action",
        observation_key: NestedKey = "observation",
        eta: float = 0.01, # Intrinsic reward scaling factor
        beta: float = 0.2,  # Forward module scaling factor
        only_intrinsic_reward: bool = False
    ):
        super().__init__(in_keys=[reward_key, action_key, observation_key], out_keys=[reward_key, "icm"])

        self._feature_net = feature_net
        self._inverse_net = _InverseNet(feature_net, inverse_net)
        self._forward_net = forward_net
        self._encoding_size = encoding_size
        self._reward_key = reward_key
        self._action_key = action_key
        self._observation_key = observation_key
        self._eta = eta
        self._beta = beta
        self._only_intrinsic_reward = only_intrinsic_reward

        self._previous_state = None
        self._icm_observation_spec = None
        self.loss_module = IntrinsicCuriosityLoss(feature_net, forward_net, self._inverse_net, action_key=action_key, beta=beta)

    def reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self._icm_observation_spec is None:
            self.transform_observation_spec(self.parent.observation_spec)
        self._previous_state = tensordict
        tensordict.set("icm", self._icm_observation_spec.zero(tensordict.shape))
        return tensordict

    def _step(self, tensordict: TensorDictBase):
        a_0 = tensordict[self._action_key]
        s_0, s_1 = self._previous_state[self._observation_key], tensordict['next'][self._observation_key]
        s_0, s_1 = s_0.to(torch.float32), s_1.to(torch.float32)
        r_e = tensordict['next'][self._reward_key]

        with torch.no_grad():
            # True encoded states
            phi_0 = self._feature_net(s_0)
            phi_1 = self._feature_net(s_1)
            phi_1_pred = self._forward_net(torch.cat((a_0, phi_0), dim=-1))

        # Set the new rewards
        r_i = (torch.linalg.vector_norm(phi_1-phi_1_pred) * self._eta / 2).reshape(r_e.shape)
        if not self._only_intrinsic_reward:
            r = r_i + r_e
        else:
            r = r_i
        tensordict['next'].set(self._reward_key, r)
        self._previous_state = tensordict

        # Set the training targets for IntrinsicCuriosityLoss
        # And for debugging
        tensordict['next'].set(("icm", "s_0"), s_0)
        tensordict['next'].set(("icm", "s_1"), s_1)
        tensordict['next'].set(("icm", "r_i"), r_i)
        tensordict['next'].set(("icm", "r_e"), r_e)

        return tensordict
    
    def _call(self, tensordict: TensorDictBase):
        # Called in reset
        return tensordict
    
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        icm_spec = {}

        icm_spec.update({
            "icm":CompositeSpec(
                s_0 = observation_spec[self._observation_key],
                s_1 = observation_spec[self._observation_key],
                r_e = UnboundedContinuousTensorSpec(shape=(*self.parent.batch_size, 1), device=observation_spec.device),
                r_i = UnboundedContinuousTensorSpec(shape=(*self.parent.batch_size, 1), device=observation_spec.device),
                shape = self.parent.batch_size
            )
        })

        self._icm_observation_spec = icm_spec['icm']

        # Add new specs
        if not isinstance(observation_spec, CompositeSpec):
            observation_spec = CompositeSpec(
                observation=observation_spec, shape=self.parent.batch_size
            )
        observation_spec.update(icm_spec)

        return observation_spec


class _InverseNet(Module):
    def __init__(self,
                 feature_net,
                 inverse_head):
        super().__init__()
        self._feature_network = feature_net
        self._inverse_network = inverse_head

    def forward(self, s_0, s_1):
        phi_0 = self._feature_network(s_0)
        phi_1 = self._feature_network(s_1)
        a     = self._inverse_network(torch.cat((phi_0, phi_1), dim=-1))
        return a

class IntrinsicCuriosityLoss(LossModule):
    """ Intrinsic Curiosity Model
    
    Use in conjunction with an environment transformed by IntrinsicCuriosityReward    
    Trains the forward model and the inverse model
    """
    def __init__(self,
                 feature_model: Module,
                 forward_model: Module,
                 inverse_model: Module,
                 action_key: NestedKey = "action",
                 beta: float = 0.2):
        super().__init__()

        self._action_key = action_key
        self._beta = beta

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
        s_0 = tensordict[("icm", "s_0")]
        s_1 = tensordict[("icm", "s_1")]
        a_0 = tensordict[self._action_key]

        loss_inverse = (self.inverse_model(s_0, s_1) - a_0) ** 2

        phi_0 = self.feature_model(s_0)
        phi_1 = self.feature_model(s_1)
        loss_forward = (self.forward_model(torch.cat((a_0, phi_0),dim=-1)) - phi_1) ** 2

        return TensorDict(
            source={
                "loss_inverse": loss_inverse.mean() * (1-self._beta),
                "loss_forward": loss_forward.mean() * self._beta
            },
            batch_size=[]
        )
