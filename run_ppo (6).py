import os
import yaml
import pickle
import argparse
import numpy as np
import gym
import torch
from torch.optim import Adam
import torch.nn as nn
import lop.envs
from lop.algos.rl.buffer import Buffer
from lop.nets.policies import MLPPolicy
from lop.nets.valuefs import MLPVF
from lop.algos.rl.agent import Agent
from lop.algos.rl.ppo import PPO

def save_data(cfg, rets, termination_steps, stable_rank, mu, pol_weights, val_weights, weight_change=None, friction=-1.0, num_updates=0, previous_change_time=0, gating_loss=None, current_task_idx=None, seed=None):
    data_dict = {
        'rets': np.array(rets, dtype=np.float32) if len(rets) > 0 else np.array([]),
        'termination_steps': np.array(termination_steps, dtype=np.int64) if len(termination_steps) > 0 else np.array([]),
        'stable_rank': np.array(stable_rank, dtype=np.float32) if len(stable_rank) > 0 else np.array([]),
        'action_output': np.array(mu, dtype=np.float32) if len(mu) > 0 else np.array([]),
        'pol_weights': np.array(pol_weights, dtype=np.float32) if len(pol_weights) > 0 else np.array([]),
        'val_weights': np.array(val_weights, dtype=np.float32) if len(val_weights) > 0 else np.array([]),
        'weight_change': np.array(weight_change, dtype=np.float32) if len(weight_change) > 0 else np.array([]),
        'friction': float(friction),
        'num_updates': int(num_updates),
        'previous_change_time': int(previous_change_time),
        'gating_loss': np.array(gating_loss, dtype=np.float32) if len(gating_loss) > 0 else np.array([]),
        'current_task_idx': np.array(current_task_idx, dtype=np.int32) if len(current_task_idx) > 0 else np.array([]),
        'seed': seed
    }
    with open(cfg['log_path'], 'wb') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)

def load_data(cfg):
    try:
        with open(cfg['log_path'], 'rb') as f:
            data_dict = pickle.load(f)
        return data_dict
    except FileNotFoundError:
        print(f"No log file found at {cfg['log_path']}, starting with empty data.")
        return {}

def save_checkpoint(cfg, step, learner, seed=None):
    ckpt_dict = {
        'step': step,
        'num_subnetworks': len(learner.E_t_list),
        'E_t_list': [E_t.state_dict() for E_t in learner.E_t_list],
        'S_t_list': [S_t.state_dict() for S_t in learner.S_t_list],
        'log_std': learner.log_std.data,
        'opt': learner.opt.state_dict(),
        'optimizer_gating': learner.optimizer_gating.state_dict() if learner.optimizer_gating else None,
        'seed': seed
    }
    torch.save(ckpt_dict, cfg['ckpt_path'])
    if step % 60000 == 0:
        print(f'Saved checkpoint at step {step} for seed={seed}')

def load_start_step(cfg):
    start_step = 0
    if os.path.exists(cfg['ckpt_path']):
        ckpt_dict = torch.load(cfg['ckpt_path'], map_location='cpu')
        start_step = ckpt_dict['step']
        if start_step % 60000 == 0:
            print(f"Loaded start_step from checkpoint: {start_step}")
    return start_step

def load_checkpoint(cfg, device, learner):
    step = 0
    if os.path.exists(cfg['ckpt_path']):
        ckpt_dict = torch.load(cfg['ckpt_path'], map_location=device)
        step = ckpt_dict['step']
        learner.E_t_list = nn.ModuleList().to(device)
        learner.S_t_list = nn.ModuleList().to(device)
        num_subnetworks = ckpt_dict.get('num_subnetworks', len(ckpt_dict.get('E_t_list', [])))
        if 'num_subnetworks' not in ckpt_dict and step % 5 == 0:
            print("Warning: 'num_subnetworks' not in checkpoint. Inferred from E_t_list length.")
        for _ in range(num_subnetworks):
            learner.add_new_task()
        for i in range(min(len(ckpt_dict['E_t_list']), len(learner.E_t_list))):
            learner.E_t_list[i].load_state_dict(ckpt_dict['E_t_list'][i])
        for i in range(min(len(ckpt_dict['S_t_list']), len(learner.S_t_list))):
            learner.S_t_list[i].load_state_dict(ckpt_dict['S_t_list'][i])
        if 'log_std' in ckpt_dict:
            learner.log_std.data.copy_(ckpt_dict['log_std'].to(device))
            if step % 5 == 0:
                print(f"Loaded log_std: {learner.log_std}")
        learner._initialize_optimizers()
        try:
            learner.opt.load_state_dict(ckpt_dict['opt'])
            if ckpt_dict.get('optimizer_gating') and learner.optimizer_gating:
                learner.optimizer_gating.load_state_dict(ckpt_dict['optimizer_gating'])
        except Exception as e:
            if step % 5 == 0:
                print(f"Warning: Optimizer state mismatch: {e}. Reinitializing optimizers.")
            learner._initialize_optimizers()
        if step % 5 == 0:
            print(f"Recovered from checkpoint: {cfg['ckpt_path']}")
    return step, learner

def main():
    print("Starting main execution...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='cfg/ant/std.yml')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--device', '-d', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--pretrained_path', '-p', type=str, default='pretrained_model.pth', help='Path to pretrained model checkpoint')
    args = parser.parse_args()
    device = args.device
    cfg = yaml.safe_load(open(args.config))
    seed = args.seed
    cfg['seed'] = seed
    cfg['log_path'] = os.path.join(cfg['dir'], f'{seed}.log')
    cfg['ckpt_path'] = os.path.join(cfg['dir'], f'{seed}.pth')
    cfg['done_path'] = os.path.join(cfg['dir'], f'{seed}.done')
    os.makedirs(cfg['dir'], exist_ok=True)

    # Convert n_steps to integer
    cfg['n_steps'] = int(float(cfg['n_steps']))

    env = gym.make(cfg['env_name'])
    env.name = None

    np.random.seed(seed)
    random_state = np.random.get_state()
    torch_seed = np.random.randint(1, 2**31 - 1)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)

    opt = Adam
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    pol = MLPPolicy(o_dim, a_dim, act_type=cfg['act_type'], h_dim=cfg['h_dim'], device=device, init=cfg.get('init', 'lecun'))
    pol.input_dim = o_dim
    pol.output_dim = a_dim
    vf = MLPVF(o_dim, act_type=cfg['act_type'], h_dim=cfg['h_dim'], device=device, init=cfg.get('init', 'lecun'))
    np.random.set_state(random_state)

    buf = Buffer(o_dim, a_dim, cfg['bs'], device=device)

    # Load pretrained model with compatibility for original PPO format
    pretrained_path = args.pretrained_path
    load_main = pretrained_path if os.path.exists(pretrained_path) else None
    if load_main:
        try:
            ckpt_dict = torch.load(pretrained_path, map_location=device)
            if 'pol_state_dict' in ckpt_dict:
                pol.load_state_dict(ckpt_dict['pol_state_dict'])
                vf.load_state_dict(ckpt_dict['vf_state_dict'])
            elif 'state' in ckpt_dict and any(str(i) in ckpt_dict['state'] for i in range(13)):  # Assuming 13 params from original PPO
                backbone_state = pol.backbone.state_dict()
                for i in range(13):
                    param_data = ckpt_dict['state'].get(str(i), {}).get('exp_avg', None)
                    if param_data is not None and i < len(backbone_state):
                        list(backbone_state.values())[i].copy_(param_data)
                pol.backbone.load_state_dict(backbone_state)
                print(f"Loaded pretrained backbone weights from {pretrained_path} using optimizer state")
            else:
                print(f"Warning: No 'pol_state_dict' or compatible optimizer state found in {pretrained_path}, using default initialization.")
            print(f"Loaded pretrained model from {pretrained_path}")
        except Exception as e:
            print(f"Error loading pretrained model from {pretrained_path}: {e}, using default initialization.")

    start_step = load_start_step(cfg)

    learner = PPO(
        pol=pol,
        buf=buf,
        lr=cfg['lr'],
        g=cfg['g'],
        vf=vf,
        lm=cfg['lm'],
        Opt=opt,
        u_epi_up=cfg['u_epi_ups'],
        device=device,
        n_itrs=cfg['n_itrs'],
        n_slices=cfg['n_slices'],
        u_adv_scl=cfg['u_adv_scl'],
        clip_eps=cfg['clip_eps'],
        max_grad_norm=cfg.get('max_grad_norm', 0.5),
        init=cfg.get('init', 'lecun'),
        wd=float(cfg.get('wd', 0)),
        betas=(cfg.get('beta_1', 0.9), cfg.get('beta_2', 0.999)),
        eps=float(cfg.get('eps', 1e-8)),
        no_clipping=cfg.get('no_clipping', False),
        loss_type=cfg.get('loss_type', 'ppo'),
        perturb_scale=cfg.get('perturb_scale', 0),
        util_type_val=cfg.get('util_type_val', 'contribution'),
        replacement_rate=cfg.get('rr', 1e-4),
        decay_rate=cfg.get('decay_rate', 0.99),
        vgnt=cfg.get('vgnt', 0),
        pgnt=cfg.get('pgnt', 0),
        util_type_pol=cfg.get('util_type_pol', 'contribution'),
        mt=cfg.get('mt', 10000),
        redo=cfg.get('redo', False),
        threshold=cfg.get('threshold', 0.03),
        reset_period=cfg.get('reset_period', 1000),
        h_dim=cfg['h_dim'],
        gating_thr=cfg.get('gating_thr', 0.0001),
        max_subnetworks=10,
        load_main=load_main,
        ckpt_path=cfg['ckpt_path']
    )

    to_log = cfg['to_log']
    agent = Agent(pol, learner, device=device, to_log_features=len(to_log) > 0)

    num_updates = 0
    previous_change_time = 0
    rets = []
    termination_steps = []
    mu = []
    weight_change = []
    stable_rank = []
    pol_weights = []
    val_weights = []
    gating_loss = []
    current_task_idx = []
    pol_features = []

    if os.path.exists(cfg['ckpt_path']):
        start_step, agent.learner = load_checkpoint(cfg, device, learner)
    else:
        start_step = 0
        if os.path.exists(cfg['log_path']):
            data_dict = load_data(cfg)
            num_updates = data_dict.get('num_updates', 0)
            previous_change_time = data_dict.get('previous_change_time', 0)
            rets = list(data_dict.get('rets', []))
            termination_steps = list(data_dict.get('termination_steps', []))
            stable_rank = list(data_dict.get('stable_rank', []))
            mu = list(data_dict.get('action_output', []))
            pol_weights = list(data_dict.get('pol_weights', []))
            val_weights = list(data_dict.get('val_weights', []))
            weight_change = list(data_dict.get('weight_change', []))
            gating_loss = list(data_dict.get('gating_loss', []))
            current_task_idx = list(data_dict.get('current_task_idx', []))

    if 'mu' in to_log:
        mu = np.zeros((cfg['n_steps'], a_dim), dtype=np.float32) if not mu else np.array(mu, dtype=np.float32)
    if 'pol_weights' in to_log:
        pol_layers = (len(learner.E_t_list[0]) + 1) // 2 if learner.E_t_list else 1
        pol_weights = np.zeros((cfg['n_steps'] // 1000 + 2, pol_layers), dtype=np.float32) if not pol_weights else np.array(pol_weights, dtype=np.float32)
    if 'val_weights' in to_log:
        val_layers = (len(learner.value_head) + 1) // 2
        val_weights = np.zeros((cfg['n_steps'] // 1000 + 2, val_layers), dtype=np.float32) if not val_weights else np.array(val_weights, dtype=np.float32)
    if 'stable_rank' in to_log:
        stable_rank = np.zeros(cfg['n_steps'] // 10000 + 2, dtype=np.float32) if not stable_rank else np.array(stable_rank, dtype=np.float32)
    if 'pol_features_activity' in to_log:
        pol_features = np.zeros((cfg['n_steps'] // 1000 + 2, learner.h_dim[-1]), dtype=np.float32)

    ret = 0.0
    epi_steps = 0
    o = env.reset().astype(np.float32)
    learner.set_task_at_start(torch.tensor(o, dtype=torch.float32).to(device))
    if start_step % 60000 == 0:
        print(f"start_step: {start_step} for seed={seed}")

    max_episode_steps = 1000
    episode_count = 0
    prev_total_reward = 0.0

    try:
        for step in range(start_step, cfg['n_steps']):
            a, logp, dist, _ = agent.get_action(o)
            a_np = a if isinstance(a, np.ndarray) else a.cpu().numpy()
            op, r, done, infos = env.step(a_np)
            epi_steps += 1
            op_ = op.astype(np.float32)

            if step % 100 == 0:
                print(f"Step {step}: Action={np.array2string(a_np[:5], formatter={'float': lambda x: f'{x:.2f}'})}, Reward={r:.4f}, Total Reward={ret:.2f}, Buffer Size={buf.size}, Can Sample={buf.can_sample()}, Subnetwork={learner.current_task_idx} for seed={seed}")
                if abs(ret - prev_total_reward) > 50.0:
                    print(f"Warning: Significant reward drop detected at step {step}: {prev_total_reward:.2f} to {ret:.2f}")

            val_logs = agent.log_update(o, a_np, r, op_, logp, dist, done)

            if epi_steps >= max_episode_steps and not done:
                done = True
                if step % 100 == 0:
                    print(f"Episode terminated due to max steps ({max_episode_steps}) for seed={seed}")

            current_gating_loss = 0.0
            if buf.can_sample():
                result = learner.learn(step=step, n_steps=cfg['n_steps'])
                current_gating_loss = result['gating_loss']
            gating_loss.append(current_gating_loss)
            current_task_idx.append(learner.current_task_idx)

            with torch.no_grad():
                if 'weight_change' in to_log and 'weight_change' in val_logs:
                    weight_change.append(val_logs['weight_change'])
                if 'mu' in to_log:
                    mu[step] = a_np
                if step % 100 == 0:
                    if step % 10000 == 0 and 'stable_rank' in to_log:
                        stable_rank[step // 10000] = 0
                    if 'pol_weights' in to_log and learner.E_t_list:
                        for layer_idx in range((len(learner.E_t_list[0]) + 1) // 2):
                            if layer_idx < len(learner.E_t_list[0]) // 2:
                                pol_weights[step // 1000, layer_idx] = learner.E_t_list[0][2 * layer_idx].weight.data.abs().mean().item()
                            else:
                                pol_weights[step // 1000, layer_idx] = 0
                    if 'val_weights' in to_log:
                        for layer_idx in range((len(learner.value_head) + 1) // 2):
                            val_weights[step // 1000, layer_idx] = learner.value_head[2 * layer_idx].weight.data.abs().mean().item() if layer_idx < len(learner.value_head) // 2 else 0
                    if 'pol_features_activity' in to_log:
                        features = learner.get_features(torch.as_tensor(o, device=device).unsqueeze(0))
                        pol_features[step // 1000] = features.cpu().numpy().mean(axis=0)

            o = op_
            prev_total_reward = ret
            ret += r

            if step % 1000 == 0:
                save_data(
                    cfg=cfg,
                    rets=rets,
                    termination_steps=termination_steps,
                    stable_rank=stable_rank,
                    mu=mu,
                    pol_weights=pol_weights,
                    val_weights=val_weights,
                    weight_change=weight_change,
                    num_updates=num_updates,
                    previous_change_time=previous_change_time,
                    gating_loss=gating_loss,
                    current_task_idx=current_task_idx,
                    seed=seed
                )
                if buf.size < cfg['bs'] * 0.5 and step % 1000 == 0:
                    print(f"Warning: Buffer size {buf.size} is less than half of bs {cfg['bs']} at step {step}")

            if step % (cfg['n_steps'] // 100) == 0 or step == cfg['n_steps'] - 1:
                forward_log = learner.get_forward_log()
                print(f"\nStep {step}:")
                print(f"  Reward: {r:.2f}, Total Reward: {ret:.2f}, Done: {done}, Episode Steps: {epi_steps}")
                print(f"  Buffer Size: {buf.size}, Can Sample: {buf.can_sample()}")
                print(f"  Gating Loss: {current_gating_loss:.2f}, Current Task: {learner.current_task_idx}")
                if forward_log:
                    print(f"  Forward Log: Mean={forward_log['mean'].mean().item():.2f}, Log Std={forward_log['log_std'].mean().item():.2f}, Action={forward_log['action'].mean().item():.2f}")
                save_checkpoint(cfg, step, agent.learner, seed=seed)

            if done:
                rets.append(ret)
                termination_steps.append(step)
                episode_count += 1
                if episode_count % 10 == 0 or episode_count == 1 or step == cfg['n_steps'] - 1:
                    print(f"Episode finished at step {step}: Total Reward={ret:.2f}, Episode Steps={epi_steps} for seed={seed}")
                ret = 0.0
                epi_steps = 0
                o = env.reset().astype(np.float32)
                learner.set_task_at_start(torch.tensor(o, dtype=torch.float32).to(device))

    except Exception as e:
        print(f"Error occurred: {e} for seed={seed}")
        raise

    with open(cfg['done_path'], 'w') as f:
        f.write('All done!')
    print(f'Experiment successfully completed for seed={seed}!')

if __name__ == "__main__":
    main()