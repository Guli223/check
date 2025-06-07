import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from lop.algos.rl.learner import Learner

class PPO(nn.Module, Learner):
    def __init__(self, pol, buf, lr, g, vf, lm,
                 Opt,
                 device='cpu',
                 u_epi_up=0,
                 n_itrs=10,
                 n_slices=16,
                 u_adv_scl=1,
                 clip_eps=0.2,
                 max_grad_norm=0.5,
                 util_type_val='contribution',
                 util_type_pol='contribution',
                 replacement_rate=1e-4,
                 decay_rate=0.99,
                 mt=10000,
                 vgnt=0,
                 pgnt=0,
                 init='lecun',
                 wd=0,
                 perturb_scale=0,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 no_clipping=False,
                 loss_type='ppo',
                 redo=False,
                 threshold=0.03,
                 reset_period=1000,
                 h_dim=[256, 256],
                 gating_thr=0.0001,
                 load_main=None,
                 max_subnetworks=2,
                 ckpt_path=None):
        nn.Module.__init__(self)
        Learner.__init__(self)
        self.pol = pol
        self.buf = buf
        self.lr = lr
        self.gamma = g
        self.vf = vf  # Store the value function
        self.lm = lm
        self.device = device
        self.u_epi_up = u_epi_up
        self.n_itrs = n_itrs
        self.n_slices = n_slices
        self.u_adv_scl = u_adv_scl
        self.clip_eps = clip_eps
        self.max_grad_norm = max_grad_norm
        self.vgnt = vgnt
        self.pgnt = pgnt
        self.perturb_scale = perturb_scale
        self.no_clipping = no_clipping
        self.loss_type = loss_type
        self.to_perturb = self.perturb_scale != 0
        self.gating_thr = gating_thr
        self.h_dim = h_dim

        print(f"Received load_main: {load_main}")

        state_dim = pol.input_dim
        action_dim = pol.output_dim

        layers = []
        in_dim = state_dim
        for hidden_size in h_dim:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, action_dim))
        self.backbone = nn.Sequential(*layers).to(device)

        for param in self.backbone.parameters():
            param.requires_grad = True

        self.load_success = False
        if load_main and isinstance(load_main, str):
            try:
                pretrained_dict = torch.load(load_main, map_location=device)
                if 'actor' in pretrained_dict:
                    actor_dict = pretrained_dict['actor']
                    backbone_state = self.backbone.state_dict()
                    for name, param in actor_dict.items():
                        if name.startswith('mean_net'):
                            new_name = name.replace('mean_net.', '')
                            if new_name in backbone_state:
                                backbone_state[new_name].copy_(param.data)
                    self.backbone.load_state_dict(backbone_state)
                    if 'log_std' in pretrained_dict:
                        self.log_std = nn.Parameter(pretrained_dict['log_std'].to(device))
                        print(f"Loaded log_std: {self.log_std}")
                    else:
                        print("Pretrained model lacks 'log_std', using default.")
                        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5000, device=device))
                    print(f"Loaded pretrained actor from {load_main}")
                    self.load_success = True
                elif 'state' in pretrained_dict and any(str(i) in pretrained_dict['state'] for i in range(13)):  # Match original PPO params
                    backbone_state = self.backbone.state_dict()
                    for i in range(13):
                        param_data = pretrained_dict['state'].get(str(i), {}).get('exp_avg', None)
                        if param_data is not None and i < len(backbone_state):
                            list(backbone_state.values())[i].copy_(param_data)
                    self.backbone.load_state_dict(backbone_state)
                    self.log_std = nn.Parameter(torch.full((action_dim,), -0.5000, device=device))  # Default if not in pretrained
                    print(f"Loaded pretrained backbone weights from {load_main} using optimizer state")
                    self.load_success = True
                else:
                    raise KeyError("Pretrained checkpoint lacks 'actor' or compatible optimizer state")
            except Exception as e:
                print(f"Failed to load pretrained model from {load_main}: {e}")
                self.log_std = nn.Parameter(torch.full((action_dim,), -0.5000, device=device))
                self.load_success = False
        else:
            self.log_std = nn.Parameter(torch.full((action_dim,), -0.5000, device=device))
            print(f"No pretrained model, initialized log_std: {self.log_std}")

        self.E_t_list = nn.ModuleList().to(device)
        self.S_t_list = nn.ModuleList().to(device)
        self.current_task_idx = None
        self.max_subnetworks = max_subnetworks
        self.total_steps = 0  # Initialize total_steps

        self.value_head = nn.Sequential(
            nn.Linear(h_dim[-1], h_dim[-1]),
            nn.ReLU(),
            nn.Linear(h_dim[-1], 1)
        ).to(device)

        self.latest_forward_log = None
        self.latest_features = None

        self.Opt = Opt
        self.wd = wd
        self.betas = betas
        self.eps = eps
        self.opt = None
        self.optimizer_gating = None
        self._initialize_optimizers(step=0)

        if not self.E_t_list and (ckpt_path is None or not os.path.exists(ckpt_path)):
            self.current_task_idx = self.add_new_task()
            print(f"Step {self.total_steps}: Initialized with default subnetwork at index: {self.current_task_idx}")

    def _initialize_optimizers(self, step=0):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if step == 0:
            print(f"Step {step}: Trainable parameters: {[name for name, param in self.named_parameters() if param.requires_grad]}")
            print(f"Step {step}: Optimizer initialized with {len(trainable_params)} trainable parameters.")
        if self.opt is None:
            self.opt = self.Opt(trainable_params, lr=self.lr, weight_decay=self.wd, betas=self.betas, eps=self.eps)
        if self.S_t_list:
            gating_params = [p for S_t in self.S_t_list for p in S_t.parameters() if p.requires_grad]
            if self.optimizer_gating is None:
                self.optimizer_gating = self.Opt(gating_params, lr=self.lr, weight_decay=self.wd, betas=self.betas, eps=self.eps)
        else:
            self.optimizer_gating = None

    def add_new_task(self):
        if len(self.E_t_list) >= self.max_subnetworks:
            print(f"Step {self.total_steps}: Max subnetworks ({self.max_subnetworks}) reached, reusing last subnetwork.")
            return len(self.E_t_list) - 1
        E_t = nn.Sequential(
            nn.Linear(self.backbone[0].in_features, self.h_dim[0]),
            nn.ReLU(),
            nn.Linear(self.h_dim[0], self.h_dim[1]),
            nn.ReLU(),
            nn.Linear(self.h_dim[1], self.h_dim[-1])  # Output matches h_dim[-1] (256)
        ).to(self.device)
        with torch.no_grad():
            E_t[0].weight.copy_(self.backbone[0].weight)
            E_t[0].bias.copy_(self.backbone[0].bias)
            E_t[2].weight.copy_(self.backbone[2].weight)
            E_t[2].bias.copy_(self.backbone[2].bias)
            # No need to copy the last layer since output dimension is now h_dim[-1]
        S_t = nn.Sequential(
            nn.Linear(self.h_dim[-1], 1),  # Input matches h_dim[-1] (256)
            nn.Sigmoid()
        ).to(self.device)

        self.E_t_list.append(E_t)
        self.S_t_list.append(S_t)
        new_idx = len(self.E_t_list) - 1
        print(f"Step {self.total_steps}: Added new subnetwork at index {new_idx}")
        return new_idx

    def set_task_at_start(self, initial_state):
        """
        Check initial state against existing subnetworks to decide whether to use an existing one
        or create a new subnetwork based on dataset similarity.
        """
        if self.current_task_idx is None:
            if not self.E_t_list:
                self.current_task_idx = self.add_new_task()
                print(f"Step {self.total_steps}: No existing subnetworks, created new subnetwork at index {self.current_task_idx}")
            else:
                h_0 = self.backbone[:-2](initial_state.unsqueeze(0))
                c_values = []
                for i, S_t in enumerate(self.S_t_list):
                    c_t = S_t(h_0)
                    c_values.append(c_t.mean().item())
                if c_values and max(c_values) >= self.gating_thr:
                    self.current_task_idx = np.argmax(c_values)
                    print(f"Step {self.total_steps}: Selected existing subnetwork {self.current_task_idx} with c_values={c_values}")
                else:
                    self.current_task_idx = self.add_new_task()
                    print(f"Step {self.total_steps}: No suitable subnetwork found, created new subnetwork at index {self.current_task_idx}")

    def log(self, o, a, r, op, logpb, dist, done):
        self.total_steps += 1
        self.buf.store(o, a, r, op, logpb, dist, done)
        if self.total_steps % 60000 == 0:
            print(f"Step {self.total_steps}: Storing data - o={o.shape}, a={a.shape}, r={r}, op={op.shape}, logpb={logpb}, dist={dist}, done={done}")
            print(f"Step {self.total_steps}: Buffer size: {self.buf.size}")

    def get_features(self, s):
        h_0 = self.backbone[:-2](s)
        h_sum = h_0.clone()
        if self.E_t_list and self.current_task_idx is not None:
            h_t = self.E_t_list[self.current_task_idx](s)
            h_sum += h_t  # Fixed subnetwork contribution
        self.latest_features = h_sum
        return h_sum

    def forward(self, s):
        h_0 = self.backbone[:-2](s)
        h_sum = h_0.clone()
        if self.E_t_list and self.current_task_idx is not None:
            h_t = self.E_t_list[self.current_task_idx](s)
            h_sum += h_t  # Use fixed subnetwork
        features = h_sum
        mean = self.backbone[-1](features)  # Final layer produces action dimension
        mean = torch.tanh(mean)
        dist = torch.distributions.Normal(mean, self.log_std.exp())
        action = dist.rsample()
        action = torch.tanh(action)
        self.latest_forward_log = {
            'mean': mean,
            'log_std': self.log_std,
            'action': action
        }
        return action, self.log_std

    def value(self, s):
        h_0 = self.backbone[:-2](s)
        h_sum = h_0.clone()
        if self.E_t_list and self.current_task_idx is not None:
            h_t = self.E_t_list[self.current_task_idx](s)
            h_sum += h_t  # Use fixed subnetwork
        features = h_sum
        return self.value_head(features)

    def get_rets_advs(self, rs, dones, values):
        rs = rs.to(self.device)
        dones = dones.to(self.device)
        values = values.to(self.device)
        
        returns = torch.zeros_like(rs)
        advantages = torch.zeros_like(rs)
        running_return = torch.tensor(0.0, device=self.device)
        running_advantage = torch.tensor(0.0, device=self.device)
        prev_value = torch.tensor(0.0, device=self.device)
        gamma = self.gamma
        lambda_ = self.lm

        for t in range(rs.shape[0] - 1, -1, -1):
            r = rs[t]
            done = dones[t]
            value = values[t]
            running_return = r + gamma * running_return * (1 - done)
            td_error = r + gamma * prev_value * (1 - done) - value
            running_advantage = td_error + gamma * lambda_ * running_advantage * (1 - done)
            advantages[t] = running_advantage
            returns[t] = running_return + value
            prev_value = value

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def learn(self, step=0, n_steps=None):
        if step % 60000 == 0:
            print(f"Step {step}: Checking buffer for learning...")
        os, acts, rs, op, logpbs, distbs, dones = self.buf.get(self.pol.dist_stack)
        if step % 60000 == 0:
            print(f"Step {step}: Buffer size: {self.buf.size}, Can sample: {self.buf.can_sample()}")

        if os.numel() == 0:
            if step % 60000 == 0:
                print(f"Step {step}: Empty buffer, skipping learning.")
            return {'gating_loss': 0.0, 'weight_change': 0}

        with torch.no_grad():
            features = self.get_features(os)
            pre_vals = self.value_head(features)
        v_rets, advs = self.get_rets_advs(rs, dones, pre_vals.squeeze(-1))

        if step % 1000 == 0:
            total_reward = sum(rs) if isinstance(rs, (list, np.ndarray)) else rs.sum().item()
            print(f"Step {step}: Total Reward: {total_reward:.2f}")

        inds = np.arange(len(os) if isinstance(os, (list, tuple)) else os.shape[0])
        mini_bs = self.buf.bs // self.n_slices if self.buf.bs else len(os)

        for _ in range(self.n_itrs):
            np.random.shuffle(inds)
            for start in range(0, len(inds), mini_bs):
                end = min(start + mini_bs, len(inds))
                ind = inds[start:end]
                if not isinstance(os, (list, tuple)):
                    obs_batch = os[ind]
                else:
                    obs_batch = torch.stack([os[i] for i in ind])
                mean, log_std = self.forward(obs_batch)
                dist = torch.distributions.Normal(mean, log_std.exp())
                logpts = dist.log_prob(acts[ind]).sum(dim=-1)
                grad_sub = (logpts - logpbs[ind]).exp()
                p_loss0 = -(grad_sub * advs[ind])
                p_loss = p_loss0.mean() if self.no_clipping else -torch.min(p_loss0, torch.clamp(grad_sub, 1 - self.clip_eps, 1 + self.clip_eps) * advs[ind]).mean()

                vals = self.value_head(self.get_features(obs_batch))
                v_loss = (v_rets[ind] - vals.squeeze(-1)).pow(2).mean()

                total_loss = p_loss + self.lm * v_loss
                self.opt.zero_grad()
                total_loss.backward()

                grad_norm = nn.utils.clip_grad_norm_(
                    [p for p in self.parameters() if p.requires_grad],
                    self.max_grad_norm
                )
                if step % 60000 == 0:
                    print(f"Step {step}: Gradient norm: {grad_norm:.4f}")

                self.opt.step()

                if step % (n_steps // 100) == 0:
                    print(f"Step {step}: Iteration loss - Policy: {p_loss.item():.2f}, Value: {v_loss.item():.2f}")
                    print(f"Step {step}: Sample reward: {rs[ind[:5]].mean().item() if len(ind) > 0 else 0:.2f}")

        if step % 60000 == 0:
            print(f"Step {step}: Buffer size after learn: {self.buf.size}")

        return {'gating_loss': 0.0, 'weight_change': 0}

    def get_forward_log(self):
        """Return the latest forward pass log."""
        return self.latest_forward_log

    def load_state_dict(self, state_dict, strict=True):
        new_state_dict = {}
        for key, value in state_dict.items():
            if not key.startswith('E_t_list') and not key.startswith('S_t_list'):
                new_state_dict[key] = value
        super().load_state_dict(new_state_dict, strict=False)
        print(f"Loaded state dict, preserved E_t_list and S_t_list. E_t_list length: {len(self.E_t_list)}")