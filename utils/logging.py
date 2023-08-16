import numpy as np
import wandb

import torch
from torchrl.record.loggers import Logger
from torchrl.record.loggers.wandb import WandbLogger
from tensordict import TensorDictBase

""" VMAS

def log_evaluation(logger: Logger,
                   rollouts,
                   env,
                   loss,
                   step):
    rollouts = list(rollouts.unbind(0))
    for k, r in enumerate(rollouts):
        next_done = r.get(("next", "done")).sum(
            tuple(range(r.batch_dims, r.get(("next", "done")).ndim)),
            dtype=torch.bool,
        )
        done_index = next_done.nonzero(as_tuple=True)[0][
            0
        ]  # First done index for this traj
        rollouts[k] = r[: done_index + 1]

    rewards = [td.get(("next", "agents", "reward")).sum(0).mean() for td in rollouts]

    vid = torch.tensor(
        np.transpose(env.frames[: rollouts[0].batch_size[0]], (0, 3, 1, 2)),
        dtype=torch.uint8,
    ).unsqueeze(0)
    
    logger.log_scalar(name="eval/reward", value=sum(rewards)/len(rollouts), step=step)
    logger.log_scalar(name="train/loss",  value=loss.item(), step=step)
    logger.log_video(name="eval/video",   video=vid, step=step)
"""

def log_evaluation(
    cfg,
    logger: WandbLogger,
    rollout: TensorDictBase,
    env_test,
    evaluation_time: float,
    step: int,
):
    reward = rollout.get(("next", "reward")).mean()
    to_log = {
        "eval/episode_reward": reward / cfg.env.frame_skip,
        "eval/episode_len": len(rollout),
        "eval/evaluation_time": evaluation_time / cfg.env.frame_skip,
    }

    vid = torch.tensor(np.array(env_test.frames).transpose(0,3,1,2)).unsqueeze(0)

    if isinstance(logger, WandbLogger):
        logger.experiment.log(to_log, commit=False)
        logger.experiment.log(
            {
                "eval/video": wandb.Video(vid, fps=20, format="mp4"),
            },
            commit=False,
        )
    else:
        for key, value in to_log.items():
            logger.log_scalar(key.replace("/", "_"), value, step=step)
        logger.log_video("eval_video", vid, step=step)

def log_training(
    cfg, 
    logger: Logger,
    training_td: TensorDictBase,
    loss: float,
    loss_curiosity: float,
    step: int
):
    reward = training_td.get(("next", "reward")).mean() / cfg.env.frame_skip
    logger.log_scalar("train/loss", loss / cfg.env.frame_skip)
    logger.log_scalar("train/loss_curiosity", loss_curiosity / cfg.env.frame_skip)
    logger.log_scalar("train/reward", reward / cfg.env.frame_skip)
    logger.log_scalar("train/step", step)
