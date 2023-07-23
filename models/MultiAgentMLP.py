from typing import  Optional, Tuple, Type

import torch
import torch.nn as nn
from torchrl.modules.models import MLP

class MultiAgentMLP(nn.Module):
    """ Multi-agent MLP

    A customized copy in place of the unreleased torchrl version.
    https://github.com/matteobettini/rl/blob/2d35754d03da319c2f878b7d49333221a2921b1f/torchrl/modules/models/multiagent.py#L18

    # Adjustments:
        Centralized parameter sharing agents expand output dimension to n_agent_outputs * n_agents
    """

    def __init__(self,
                 n_agent_inputs: int,
                 n_agent_outputs: int,
                 n_agents: int,
                 centralised: bool = True,
                 share_params: bool = True,
                 device: str = 'cpu',
                 depth: int = 2,
                 num_cells: int = 32,
                 activation_class: Type[nn.Module] = nn.LeakyReLU) -> None:
        """
        Args:
            n_agent_inputs: Number of inputs for each agent
            n_agent_outputs: Number of outputs for each agent
            n_agents: Number of agents
        """
        super().__init__()
        self.n_agents = n_agents
        self.n_agent_inputs = n_agent_inputs
        self.n_agent_outputs = n_agent_outputs
        self.share_params = share_params
        self.centralised = centralised
        self.networks = nn.ModuleList(
            [
                MLP(in_features = n_agent_inputs if not centralised else n_agent_inputs * n_agents,
                    out_features = n_agent_outputs if not (centralised and share_params) else n_agent_outputs * n_agents,
                    depth=depth,
                    num_cells=num_cells,
                    activation_class=activation_class,
                    device=device)
                for _ in range(1 if share_params else self.n_agents)
            ]
        )

    def forward(self, *inputs: Tuple[torch.tensor]) -> torch.Tensor:

        # Rearrange into single torch tensor
        if len(inputs) > 1:
            inputs = torch.cat([*inputs], -1)
        else:
            inputs = inputs[0]

        assert inputs.shape[-2:] == (self.n_agents, self.n_agent_inputs), \
            f"Last two dimensions must be equivalent to ({self.n_agents}, {self.n_agent_inputs}) but got {inputs.shape}"
        
        # Sharing parameters and centralization
        if self.share_params:
            if self.centralised:
                inputs = inputs.reshape(*inputs.shape[:-2], self.n_agents * self.n_agent_inputs)
                output = self.networks[0](inputs)
                output = output.reshape(*output.shape[:-1], self.n_agents, self.n_agent_outputs)
            else:
                output = self.networks[0](inputs)
        else:
            if self.centralised:
                inputs = inputs.reshape(*inputs.shape[:-2], self.n_agents * self.n_agent_inputs)
                output = torch.stack([net(inputs) for i, net in enumerate(self.networks)], dim=-2)
            else:
                output = torch.stack([net(inputs[..., i, :]) for i, net in enumerate(self.networks)], dim=-2)
        return output
        