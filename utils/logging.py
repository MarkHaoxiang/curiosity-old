import numpy as np

import torch
from torchrl.record.loggers import Logger

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
