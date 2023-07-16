import argparse

import torch
from torchrl import envs
import vmas

if __name__ == "__main__":
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr", dest="learning_rate",
        type=float, default=3e-4,
        help="learning rate"
    )
    parser.add_argument(
        "--seed", dest="seed",
        type=float, default=torch.seed(),
        help="seed"
    )
    parser.add_argument(
        "--frames_per_batch", dest="frames_per_batch",
        type=int, default=100,
        help="frames per batch"
    )
    parser.add_argument(
        "--total_frames", dest="total_frames",
        type=int, default=10000,
        help="total number of frames"
    )
    parser.add_argument(
        "--frame_skip", dest="frame_skip",
        type=int, default=1,
        help="frames to skip (repeat actions)"
    )
    parser.add_argument(
        "--scenario", dest="scenario",
        type=str, default='waterfall',
        help="environment scenario"
    )
    args = parser.parse_args()
    args.frames_per_batch = args.frames_per_batch // args.frame_skip
    args.total_frames = args.total_frames // args.frame_skip

    # Torch Setup
    device = "cpu" if not torch.has_cuda else "cuda:0"
    torch.manual_seed(args.seed)

    # Environment Setup
    base_env = vmas.make_env(
        scenario=args.scenario,
        num_envs=32,
        device=device,
        continuous_actions=True,
        max_steps=args.total_frames,
        seed=args.seed,
        dict_spaces=True
    )
    env = envs.TransformedEnv(env=base_env, transform = envs.Compose(
        envs.FrameSkipTransform(args.frame_skip),
        None))
    # Policy
    policy = None # TODO

