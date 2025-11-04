import os
import argparse
from datetime import datetime
import sys
import torch
sys.path.append("/home/tjx/MTDS")
from envs.env_for_push_reach import ur5Env_1
from pcrl.algo.pcrl_two import PPO
from pcrl.trainer_one_AC import Trainer



def run(args):
    env = ur5Env_1()
    env_test = ur5Env_1()
    device = torch.device("cuda")

    algo = PPO(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device,
        seed=args.seed
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, 'sac', f'seed{args.seed}-{time}')

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num_steps', type=int, default=10**7)
    p.add_argument('--eval_interval', type=int, default=10**4)
    p.add_argument('--env_id', type=str, default='Hopper-v2')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)
