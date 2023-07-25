# Configurations
import hydra
from omegaconf import DictConfig

import torch
from torchrl.envs import TransformedEnv, RewardSum
from torchrl.envs.utils import set_exploration_type, ExplorationType
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.modules import ProbabilisticActor, AdditiveGaussianWrapper, TanhDelta, ValueOperator
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from tensordict.nn import TensorDictModule
from torchrl.objectives import DDPGLoss, ValueEstimators
from torchrl.record.loggers.utils import get_logger, generate_exp_name

from tqdm import tqdm

from models.MultiAgentMLP import MultiAgentMLP
from utils.logging import log_evaluation
# Derived from
# https://github.com/pytorch/rl/pull/1027

@hydra.main(version_base=None, config_path="conf", config_name="maddpg")
def train(cfg: "DictConfig"):
    # General setup
    cfg.train.device = "cpu" if not torch.backends.cuda.is_built() else "cuda:0"
    cfg.env.device = cfg.train.device

    # Seeding
    torch.manual_seed(cfg.seed)
    
    # Sampling
    cfg.env.num_envs = cfg.collector.frames_per_batch // cfg.env.max_steps
    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters
    cfg.buffer.memory_size = cfg.collector.frames_per_batch

    # Create env and test
    env = VmasEnv(scenario=cfg.env.scenario_name,
                  num_envs=cfg.env.num_envs,
                  continuous_actions=True,
                  max_steps=cfg.env.max_steps,
                  device=cfg.train.device,
                  seed=cfg.seed,
                  **cfg.env.scenario)

    env_eval = VmasEnv(scenario=cfg.env.scenario_name,
                       num_envs=cfg.eval.evaluation_episodes,
                       continuous_actions=True,
                       max_steps=cfg.env.max_steps,
                       device=cfg.train.device,
                       seed=cfg.seed,
                       **cfg.env.scenario) 

    env = TransformedEnv(env,
                         RewardSum(in_keys=[env.reward_key],
                                   out_keys=[("agents","episode_reward")]
                                   )
    )
    # Policy
    policy_network = MultiAgentMLP(n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
                                   n_agent_outputs=env.action_spec.shape[-1],
                                   n_agents=env.n_agents,
                                   centralised=cfg.model.centralised,
                                   share_params=cfg.model.shared_parameters,
                                   device=cfg.train.device,
                                   depth=cfg.model.depth,
                                   num_cells=cfg.model.num_cell
    )
    policy_module = TensorDictModule(policy_network,
                                     in_keys=[("agents", "observation")],
                                     out_keys=[("agents", "param")]
    )

    policy = ProbabilisticActor(module=policy_module,
                                spec=env.unbatched_action_spec,
                                in_keys=[("agents", "param")],
                                out_keys=[env.action_key],
                                distribution_class = TanhDelta,
                                distribution_kwargs={
                                    "min": env.unbatched_action_spec[("agents", "action")].space.minimum,
                                    "max": env.unbatched_action_spec[("agents", "action")].space.maximum
                                },
                                return_log_prob=False
    )

    exploration_module = AdditiveGaussianWrapper(
        policy,
        annealing_num_steps=int(cfg.collector.total_frames * 0.5),
        action_key=env.action_key
    )

    # Critic
    critic_network = MultiAgentMLP(n_agent_inputs=env.observation_spec[("agents", "observation")].shape[-1]
                                        + env.action_spec.shape[-1], # Q critic
                                   n_agent_outputs=1,
                                   n_agents=env.n_agents,
                                   centralised=cfg.model.centralised,
                                   share_params=cfg.model.shared_parameters,
                                   device=cfg.train.device,
                                   depth=cfg.model.depth,
                                   num_cells=cfg.model.num_cell
    )
    
    value_module = ValueOperator(module=critic_network,
                                 in_keys=[("agents", "observation"), env.action_key],
                                 out_keys=[("agents","state_action_value")]
    )

    # Data
    collector = SyncDataCollector(env,
                                  exploration_module,
                                  device=cfg.env.device,
                                  storing_device = cfg.train.device,
                                  frames_per_batch=cfg.collector.frames_per_batch,
                                  total_frames=cfg.collector.total_frames
    )

    replay_buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(max_size=cfg.buffer.memory_size,
                                                                     device=cfg.train.device),
                                           sampler=SamplerWithoutReplacement(),
                                           batch_size=cfg.train.minibatch_size
    )

    
    # Train
    loss_module = DDPGLoss(actor_network=policy,
                           value_network=value_module
    )
    loss_module.set_keys(state_action_value=("agents", "state_action_value"),
                         reward=env.reward_key)
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=cfg.loss.gamma)

    optim = torch.optim.Adam(loss_module.parameters(), cfg.train.lr)

    # Logging
    model_name = "Testing-MADDPG"
    logger = get_logger(experiment_name=generate_exp_name(cfg.env.scenario_name, model_name),
                        logger_name="logs",
                        logger_type=cfg.logger.backend
    )

    # Training Loop
    print("Initialization complete. Begin training.")

    for i, tensordict_data in tqdm(enumerate(collector)):
        tensordict_data.set(key=("next","done"),
                            item=tensordict_data.get(("next", "done"))
                            .unsqueeze(-1)
                            .expand(tensordict_data.get(("next", env.reward_key)).shape)
        )

        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        for _ in range(cfg.train.num_epochs):
            for _ in range(cfg.collector.frames_per_batch // cfg.train.minibatch_size):
                # Sample
                minibatch = replay_buffer.sample()

                # Calculate Loss
                loss_values = loss_module(minibatch)
                loss_value = loss_values["loss_actor"] + loss_values["loss_value"]
                loss_value.backward()

                # Keep gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.train.max_grad_norm)

                # Step
                optim.step()
                optim.zero_grad()

        # Logging
        if i % cfg.eval.evaluation_interval == 0:
            with torch.no_grad() and set_exploration_type(ExplorationType.MEAN):
                env_eval.frames = []
                rollouts = env_eval.rollout(max_steps=cfg.env.max_steps,
                                            policy=policy,
                                            callback=rendering_callback,
                                            auto_cast_to_device=True,
                                            break_when_any_done=False
                )
                log_evaluation(logger, rollouts, env_eval, loss_value, i)


def rendering_callback(env, td):
    env.frames.append(env.render(mode="rgb_array", agent_index_focus=None))

if __name__=="__main__":
    train()
