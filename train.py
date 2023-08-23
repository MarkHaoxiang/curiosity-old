# Runs a Gym environment with PPO and optionally intrinsic curiosity loss
import math

# Configurations
import hydra
from omegaconf import DictConfig
from tqdm import tqdm


from tensordict.nn import TensorDictModule, NormalParamExtractor
import torch
from functorch.experimental import replace_all_batch_norm_modules_
from torchvision.transforms import Resize
from torch.optim import Adam
from torch import nn
from torchrl.data.tensor_specs import DiscreteBox
from torchrl.envs.libs.gym import GymEnv
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE, TD0Estimator
from torchrl.record.loggers.utils import get_logger, generate_exp_name
from torchrl.envs import (
    TransformedEnv,
    Compose,
    RewardSum,
    ExplorationType,
    set_exploration_type,
    DoubleToFloat,
    check_env_specs
)
from torchrl.modules import (
    MLP,
    OneHotCategorical,
    TanhNormal,
    ValueOperator,
    ProbabilisticActor
)
from tqdm import tqdm

from models.ICM import IntrinsicCuriosityModule
from utils.logging import log_evaluation, log_training
def build_models(env, cfg):

    n_features = env.observation_spec[cfg.env.observation_key].shape[0]
    if isinstance(env.action_spec.space, DiscreteBox):
        continuous_actions = False
        n_actions = env.action_spec.shape[-1]
        distribution_class = OneHotCategorical
        distribution_kwargs = {}
    else:
        continuous_actions = True
        n_actions = env.action_spec.shape[-1] * 2
        distribution_class = TanhNormal
        distribution_kwargs = {
            "min": env.action_spec.space.minimum,
            "max": env.action_spec.space.maximum,
            "tanh_loc": False,
        }

    if not cfg.env.from_pixels:
        # MLP for vector observation
        common_net = MLP(
            in_features = n_features,
            out_features = cfg.model.num_cell,
            num_cells=cfg.model.num_cell,
            depth = cfg.model.depth // 2 + cfg.model.depth % 2,
            activation_class=nn.Tanh,
            activate_last_layer=True,
            device=cfg.train.device
        )
    else:
        # Image input, MobileNet CNN
        mobilenet = torch.hub.load(
            'pytorch/vision:v0.10.0',
            'mobilenet_v2',
            pretrained=True
        ).to(cfg.train.device)
        mobilenet = replace_all_batch_norm_modules_(mobilenet)
        # Freeze lower layers (19 total, freeze 18)
        cnt = 0
        for child in mobilenet.features.children():
            cnt += 1
            if cnt <= 18:
                for param in child.parameters():
                    param.requires_grad = False

        # Need to use our own final layers instead of classifier to avoid dropout!

        class CommonNet(nn.Module):
            def forward(self, x: torch.Tensor):
                x = x.to(torch.float32)
                if x.shape[-1] == 3:
                    x = x.movedim(-1,-3)
                s = x.shape
                assert s[-3] == 3, "Incorrect number of dimensions"
                # Resize if needed
                scale = 224 / min(s[-1], s[-2])
                if scale > 1:
                    x = Resize(224)(x)
                assert x.shape[-1] >= 224 and x.shape[-2] >= 224, "Height and width should be at least 224"
                if len(s) == 3:
                    x = x.unsqueeze(0)
                x = torch.flatten(x, start_dim=-0, end_dim=-4).contiguous()
                result = mobilenet(x)
                assert result.shape[-1] == 1000
                result = result.reshape((*s[:-3], 1000))
                return result

        common_net = CommonNet()

    # Policy Definition
    policy_head = MLP(
            in_features = cfg.model.num_cell if not cfg.env.from_pixels else 1000,
            out_features = n_actions,
            num_cells=cfg.model.num_cell,
            depth = cfg.model.depth // 2,
            activation_class=nn.Tanh,
            activate_last_layer=False,
            device=cfg.train.device
        )

    if continuous_actions:
        policy_net = nn.Sequential(
            common_net,
            policy_head,
            NormalParamExtractor()
        )
    else:
        policy_net = nn.Sequential(
            common_net,
            policy_head
        )

    policy_module = TensorDictModule(
        module=policy_net,
        in_keys=[cfg.env.observation_key],
        out_keys=["loc", "scale"] if continuous_actions else ['logits']
    )

    policy_module = ProbabilisticActor(
        module=policy_module,
        in_keys=["loc", "scale"] if continuous_actions else ['logits'],
        out_keys=[env.action_key],
        spec=env.action_spec,
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM
    )

    # Value definition
    value_head = MLP(
            in_features = cfg.model.num_cell if not cfg.env.from_pixels else 1000,
            out_features = 1,
            num_cells=cfg.model.num_cell,
            depth = cfg.model.depth // 2,
            activation_class=nn.Tanh,
            activate_last_layer=False,
            device=cfg.train.device
        )

    value_net = nn.Sequential(
        common_net,
        value_head
    )
    
    value_module = ValueOperator(
        module=value_net,
        in_keys=[cfg.env.observation_key]
    )
    # Feature definition
    feature_head = MLP(
        in_features=cfg.model.num_cell if not cfg.env.from_pixels else 1000,
        out_features=cfg.curiosity.encoding_size,
        num_cells=cfg.model.num_cell,
        depth = cfg.model.depth // 2 + cfg.model.depth % 2,
        activation_class=nn.Tanh,
        activate_last_layer=True,
        device=cfg.train.device
    )

    feature_net = nn.Sequential(
        common_net,
        feature_head
    )

    forward_net = MLP(
        in_features=cfg.curiosity.encoding_size + n_actions,
        depth=cfg.model.depth // 2,
        out_features=cfg.curiosity.encoding_size,
        device=cfg.train.device        
    )

    inverse_head = MLP(
        in_features=cfg.curiosity.encoding_size * 2,
        depth=cfg.model.depth // 2,
        out_features=n_actions,
        device=cfg.train.device
    )

    icm = IntrinsicCuriosityModule(
        feature_net=feature_net,
        inverse_net=inverse_head,
        forward_net=forward_net,
        eta=cfg.curiosity.eta,
        beta=cfg.curiosity.beta,
        only_intrinsic_reward=cfg.curiosity.intrinsic_only,
        observation_key=cfg.env.observation_key
    )

    return policy_module, value_module, icm


@hydra.main(version_base=None, config_path="conf", config_name="ppo")
def train(cfg: "DictConfig"):
    # General setup
    cfg.train.device = "cpu" if not torch.backends.cuda.is_built() else "cuda:0"
    cfg.env.observation_key = "observation" if not cfg.env.from_pixels else "pixels"
    # Seeding
    torch.manual_seed(cfg.seed)
    
    # Metadata
    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters // cfg.env.frame_skip
    cfg.buffer.memory_size = cfg.collector.frames_per_batch = cfg.collector.frames_per_batch // cfg.env.frame_skip

    # Create env and test
    env = GymEnv(env_name=cfg.env.env_name, frame_skip=cfg.env.frame_skip, device=cfg.train.device)
    env_eval = GymEnv(env_name=cfg.env.env_name, frame_skip=cfg.env.frame_skip, device=cfg.train.device, render_mode='rgb_array')
    env = TransformedEnv(
        env,
        Compose(
            DoubleToFloat(in_keys=[cfg.env.observation_key]),
            RewardSum(
                in_keys=env.reward_key,
                out_keys=["episode_reward"]
            )
        )
    )

    # Curiosity
    # Create Models    
    policy_module, value_module, icm = build_models(env, cfg)
    policy_module(env.reset(seed=cfg.seed))
    value_module(env_eval.reset(seed=cfg.seed))
    
    env = TransformedEnv(
        env,
        icm
    )

    env.reset()

    env_eval = TransformedEnv(
        env_eval,
        DoubleToFloat(in_keys=[cfg.env.observation_key])
    )

    check_env_specs(env)

    # Data
    collector = SyncDataCollector(
        env,
        policy_module,
        device=cfg.train.device,
        storing_device="cpu",
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(
            max_size=cfg.buffer.memory_size,
            device=cfg.train.device
        ),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.train.minibatch_size
    )

    # Loss
    advantage_module = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.lmbda,
        value_network=value_module,
        average_gae=True
    )
    loss_module = ClipPPOLoss(
        actor=policy_module,
        critic=value_module,
        normalize_advantage=True
    )

    # Optimizers
    optim = Adam(loss_module.parameters(), cfg.train.lr, weight_decay=cfg.train.decay)
    optim_curiosity = Adam(icm.loss_module.parameters(), cfg.train.lr * 2, weight_decay=cfg.train.decay)

    # Logging
    if cfg.logger.enable:
        model_name = "Testing-PPO"
        logger = get_logger(experiment_name=generate_exp_name(cfg.env.env_name, model_name),
                            logger_name="logs",
                            logger_type=cfg.logger.backend
        )

    # Training loop
    for i, tensordict_data in tqdm(enumerate(collector)):
        replay_buffer.extend(tensordict_data)

        loss = []
        loss_c = []
        for _ in range(cfg.train.num_epochs):
            # GAE
            with torch.no_grad():
                tensordict_data = advantage_module(tensordict_data.to(cfg.train.device)).cpu()
            data_reshape = tensordict_data.reshape(-1)
            replay_buffer.extend(data_reshape)

            for _ in range(cfg.collector.frames_per_batch // cfg.train.minibatch_size):
                # Sample
                minibatch = replay_buffer.sample()

                # Loss
                loss_values = loss_module(minibatch)
                loss_value = loss_values['loss_critic'] + loss_values['loss_entropy'] + loss_values['loss_objective']
                loss_value.backward()
                loss.append(loss_value.item())

                loss_curiosity_values = icm.loss_module(minibatch)
                loss_curiosity_value = loss_curiosity_values['loss_inverse'] + loss_curiosity_values['loss_forward']
                loss_curiosity_value.backward()
                loss_c.append(loss_curiosity_value.item())

                # Step
                optim.step()
                optim.zero_grad()
                optim_curiosity.step()
                optim_curiosity.zero_grad()

        if cfg.logger.enable:
            log_training(
                cfg, logger,
                tensordict_data, sum(loss) / len(loss), sum(loss_c) / len(loss_c), i,
            )
    
        # Logging
        if cfg.logger.enable and i % cfg.eval.evaluation_interval == 0:
            with torch.no_grad() and set_exploration_type(ExplorationType.MODE):
                env_eval.frames = []
                rollouts = env_eval.rollout(max_steps=cfg.env.max_steps,
                                            policy=policy_module,
                                            callback=rendering_callback,
                                            auto_cast_to_device=True,
                                            break_when_any_done=False
                )
                log_evaluation(cfg, logger, rollouts, env_eval, loss_value, i)


def rendering_callback(env, td):
    env.frames.append(env.render())

if __name__=="__main__":
    train()
