import numpy as np
import collections as c
import torch

class Buffer:
    def __init__(self, o_dim, a_dim, bs, device='cpu'):
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.bs = bs
        self.device = device
        self.o_buf = c.deque(maxlen=bs)
        self.a_buf = c.deque(maxlen=bs)
        self.r_buf = c.deque(maxlen=bs)
        self.logpb_buf = c.deque(maxlen=bs)
        self.distb_buf = c.deque(maxlen=bs)
        self.done_buf = c.deque(maxlen=bs)
        self.op = np.zeros((1, o_dim), dtype=np.float32)
        self.size = 0

    def store(self, o, a, r, op, logpb, dist, done):
        if not isinstance(o, np.ndarray) or o.shape[-1] != self.o_dim:
            raise ValueError(f"Observation shape {o.shape if hasattr(o, 'shape') else type(o)} mismatch, expected {self.o_dim}")
        if not isinstance(a, np.ndarray) or a.shape[-1] != self.a_dim:
            raise ValueError(f"Action shape {a.shape if hasattr(a, 'shape') else type(a)} mismatch, expected {self.a_dim}")
        self.o_buf.append(o)
        self.a_buf.append(a)
        self.r_buf.append(r)
        self.logpb_buf.append(logpb)
        self.distb_buf.append(dist)
        self.done_buf.append(float(done))
        self.op[:] = op
        self.size = min(self.size + 1, self.bs)

    def pop(self):
        if self.size > 0:
            self.o_buf.popleft()
            self.a_buf.popleft()
            self.r_buf.popleft()
            self.logpb_buf.popleft()
            self.distb_buf.popleft()
            self.done_buf.popleft()
            self.size -= 1

    def clear(self):
        self.o_buf.clear()
        self.a_buf.clear()
        self.r_buf.clear()
        self.logpb_buf.clear()
        self.distb_buf.clear()
        self.done_buf.clear()
        self.size = 0

    def get(self, dist_stack):
        if self.size < self.bs:
            raise ValueError(f"Buffer size {self.size} < batch size {self.bs}")
        rang = range(self.bs)
        os = torch.as_tensor(np.array([self.o_buf[i] for i in rang]), dtype=torch.float32, device=self.device).view(-1, self.o_dim)
        acts = torch.as_tensor(np.array([self.a_buf[i] for i in rang]), dtype=torch.float32, device=self.device).view(-1, self.a_dim)
        rs = torch.as_tensor(np.array([self.r_buf[i] for i in rang]), dtype=torch.float32, device=self.device).view(-1, 1)
        op = torch.as_tensor(self.op, device=self.device).view(-1, self.o_dim)
        logpbs = torch.as_tensor(np.array([self.logpb_buf[i] for i in rang]), dtype=torch.float32, device=self.device).view(-1, 1)
        distbs = dist_stack([self.distb_buf[i] for i in rang], device=self.device)
        dones = torch.as_tensor(np.array([self.done_buf[i] for i in rang]), dtype=torch.float32, device=self.device).view(-1, 1)
        return os, acts, rs, op, logpbs, distbs, dones

    def can_sample(self):
        return self.size >= self.bs

    @property
    def size_steps(self):
        return self.size