"""
Policy Gradient Learning Methods for Stochastic Control with Exit Time
and Applications to Share Repurchase Pricing

Replication of: Hamdouche, Henry-Labordere, Pham (2023)
arXiv:2302.07320

Implementation in PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional
import time
import os


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Problem and algorithm parameters."""
    # Market parameters
    S0: float = 1.0          # Initial stock price
    sigma: float = 0.2       # Volatility
    gamma: float = 0.0       # Permanent market impact parameter
    
    # Contract parameters
    B: float = 1.0           # Number of shares to buy
    T_days: int = 60         # Contract horizon in trading days
    a_bar: float = 25.2      # Maximum trading rate
    lam: float = 5.0         # Penalty parameter
    beta: float = 0.0        # Transaction cost parameter
    
    # Discretization
    dt: float = 1.0 / 252.0  # Time step (1 trading day)
    
    # Algorithm
    n_episodes: int = 3000   # Number of training episodes (outer loop)
    batch_size: int = 64     # Mini-batch size K
    lr_policy: float = 1e-3  # Learning rate for policy (eta or eta_G)
    lr_value: float = 1e-3   # Learning rate for value function (eta_V)
    
    # Evaluation
    n_mc: int = 100_000      # Monte Carlo paths for price evaluation
    
    # Neural network
    hidden_dim: int = 8      # Hidden layer dimension
    n_hidden: int = 2        # Number of hidden layers
    
    # Device
    device: str = "cpu"
    
    @property
    def T(self) -> float:
        """Contract horizon in years."""
        return self.T_days * self.dt
    
    @property
    def N(self) -> int:
        """Number of time steps."""
        return self.T_days
    
    def __post_init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Neural Networks
# ============================================================================

class PolicyNetwork(nn.Module):
    """
    Softmax policy for bang-bang control: A = {0, a_bar}.
    
    Outputs the probability of choosing action a_bar (buy at max rate).
    Uses a single neural network phi_theta(t, x) and applies sigmoid
    (equivalent to softmax with 2 classes where one logit is fixed at 0).
    
    Input: (t, s, v, q, c) — time and state variables
    Output: probability of choosing a = a_bar
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Input: (t/T, s, v, q/B, c) — 5 features (we normalize t and q)
        input_dim = 5
        
        layers = []
        prev_dim = input_dim
        for _ in range(config.n_hidden):
            layers.append(nn.Linear(prev_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = config.hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # Single logit output
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights (small initialization helps stability)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def forward(self, t: torch.Tensor, s: torch.Tensor, v: torch.Tensor,
                q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Returns probability of choosing a = a_bar.
        
        Args:
            t: time, shape (batch,)
            s: stock price, shape (batch,)
            v: VWAP, shape (batch,)
            q: inventory, shape (batch,)
            c: cumulated cost, shape (batch,)
        
        Returns:
            prob: P(a = a_bar | t, x), shape (batch,)
        """
        # Normalize inputs for better training
        t_norm = t / self.config.T   # [0, 1]
        q_norm = q / self.config.B   # [0, 1] roughly
        
        x = torch.stack([t_norm, s, v, q_norm, c], dim=-1)
        logit = self.net(x).squeeze(-1)
        prob = torch.sigmoid(logit)
        
        # Clamp for numerical stability
        prob = torch.clamp(prob, 1e-6, 1.0 - 1e-6)
        return prob
    
    def log_prob(self, t: torch.Tensor, s: torch.Tensor, v: torch.Tensor,
                 q: torch.Tensor, c: torch.Tensor, 
                 a: torch.Tensor) -> torch.Tensor:
        """
        Compute log rho_theta(t, x, a).
        
        a should be 0 or a_bar. We encode: a=0 -> class 0, a=a_bar -> class 1.
        """
        prob_buy = self.forward(t, s, v, q, c)
        # is_buy = 1 if a == a_bar, 0 if a == 0
        is_buy = (a > 0.5 * self.config.a_bar).float()
        log_p = is_buy * torch.log(prob_buy) + (1 - is_buy) * torch.log(1 - prob_buy)
        return log_p
    
    def sample_action(self, t: torch.Tensor, s: torch.Tensor, v: torch.Tensor,
                      q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Sample action from the policy."""
        prob_buy = self.forward(t, s, v, q, c)
        buy = torch.bernoulli(prob_buy)
        action = buy * self.config.a_bar
        return action


class ValueNetwork(nn.Module):
    """
    Value function V_phi(t, x) for the actor-critic algorithm.
    
    Input: (t, s, v, q, c) — time and state variables
    Output: scalar value estimate
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        input_dim = 5
        
        layers = []
        prev_dim = input_dim
        for _ in range(config.n_hidden):
            layers.append(nn.Linear(prev_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = config.hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # Scalar output
        
        self.net = nn.Sequential(*layers)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def forward(self, t: torch.Tensor, s: torch.Tensor, v: torch.Tensor,
                q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        t_norm = t / self.config.T
        q_norm = q / self.config.B
        x = torch.stack([t_norm, s, v, q_norm, c], dim=-1)
        return self.net(x).squeeze(-1)


# ============================================================================
# Environment: Share Repurchase Simulator
# ============================================================================

class ShareRepurchaseEnv:
    """
    Simulates the share repurchase dynamics.
    
    State: X = (S, V, Q, C)
    - S: stock price with optional permanent market impact
      dS = S * (gamma * alpha * dt + sigma * dW)
    - V: VWAP, dV = (S - V)/t * dt, V0 = S0
    - Q: inventory, dQ = alpha * dt, Q0 = 0
    - C: cumulated cost, dC = alpha * S * dt, C0 = 0
    
    Exit time: tau = inf{t > 0 : Q_t >= B} ∧ T
    
    Terminal reward (PnL):
      g(X_tau) = B*(V_tau - S_tau) - lambda*(B - Q_tau)^+ - beta*B*C_tau
    """
    
    def __init__(self, config: Config):
        self.cfg = config
        self.device = config.device
    
    def reward(self, s: torch.Tensor, v: torch.Tensor, 
               q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Compute terminal reward g(x)."""
        B = self.cfg.B
        pnl = B * (v - s) - self.cfg.lam * torch.relu(B - q) - self.cfg.beta * B * c
        return pnl
    
    def simulate_step(self, t: torch.Tensor, s: torch.Tensor, v: torch.Tensor,
                      q: torch.Tensor, c: torch.Tensor, 
                      a: torch.Tensor, dW: torch.Tensor) -> Tuple:
        """
        One Euler step of the dynamics.
        
        Args:
            t: current time (scalar or batch)
            s, v, q, c: current state (batch,)
            a: action (batch,)
            dW: Brownian increment ~ N(0, dt) (batch,)
        
        Returns:
            s_new, v_new, q_new, c_new
        """
        dt = self.cfg.dt
        gamma = self.cfg.gamma
        sigma = self.cfg.sigma
        
        # Stock price with market impact
        s_new = s * torch.exp((gamma * a - 0.5 * sigma**2) * dt + sigma * dW)
        
        # VWAP update: dV = (S - V)/t * dt
        # At t=0, V0 = S0, so we handle t > 0 only
        # Use t + dt/2 to avoid division by zero at t=0
        t_safe = torch.clamp(t, min=dt * 0.5)
        v_new = v + (s - v) / t_safe * dt
        
        # Inventory
        q_new = q + a * dt
        
        # Cumulated cost
        c_new = c + a * s * dt
        
        return s_new, v_new, q_new, c_new
    
    def rollout(self, policy: PolicyNetwork, batch_size: int,
                return_trajectories: bool = False):
        """
        Generate trajectories under the given policy.
        
        Returns:
            If return_trajectories=False:
                rewards: (batch,) terminal rewards g(X_tau)
                log_probs_list: list of (batch,) log probs at each step before tau
                alive_masks: list of (batch,) indicators 1_{t_i < tau}
                states_list: list of state tuples for AC algorithm
            If return_trajectories=True:
                Also returns full trajectory data for plotting.
        """
        cfg = self.cfg
        device = self.device
        N = cfg.N
        
        # Initialize state
        s = torch.full((batch_size,), cfg.S0, device=device)
        v = torch.full((batch_size,), cfg.S0, device=device)
        q = torch.zeros(batch_size, device=device)
        c = torch.zeros(batch_size, device=device)
        
        # Track exit
        alive = torch.ones(batch_size, dtype=torch.bool, device=device)
        exited = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Store terminal state
        s_exit = torch.zeros(batch_size, device=device)
        v_exit = torch.zeros(batch_size, device=device)
        q_exit = torch.zeros(batch_size, device=device)
        c_exit = torch.zeros(batch_size, device=device)
        
        # For policy gradient
        log_probs_list = []
        alive_masks = []
        states_list = []  # (t, s, v, q, c) at each step
        states_next_list = []  # Next states for AC
        
        # For trajectory plotting
        if return_trajectories:
            traj_s = [s.clone()]
            traj_v = [v.clone()]
            traj_q = [q.clone()]
        
        for i in range(N):
            t_i = torch.full((batch_size,), i * cfg.dt, device=device)
            
            # Store current alive mask (1_{t_i < tau})
            alive_float = alive.float()
            alive_masks.append(alive_float.clone())
            
            # Store current states
            states_list.append((t_i.clone(), s.clone(), v.clone(), 
                              q.clone(), c.clone()))
            
            # Sample action from policy
            with torch.no_grad():
                a = policy.sample_action(t_i, s, v, q, c)
            
            # Compute log probability (need gradients here)
            log_p = policy.log_prob(t_i, s, v, q, c, a)
            log_probs_list.append(log_p)
            
            # Simulate dynamics
            dW = torch.randn(batch_size, device=device) * np.sqrt(cfg.dt)
            s_new, v_new, q_new, c_new = self.simulate_step(
                t_i, s, v, q, c, a, dW)
            
            # Check exit condition: Q >= B
            just_exited = alive & (q_new >= cfg.B)
            
            # Store exit states
            s_exit = torch.where(just_exited, s_new, s_exit)
            v_exit = torch.where(just_exited, v_new, v_exit)
            q_exit = torch.where(just_exited, q_new, q_exit)
            c_exit = torch.where(just_exited, c_new, c_exit)
            
            # Update alive status
            alive = alive & ~just_exited
            
            # Update state (only for alive paths, but we update all for simplicity)
            s = s_new
            v = v_new
            q = q_new
            c = c_new
            
            # Store next states
            t_next = torch.full((batch_size,), (i + 1) * cfg.dt, device=device)
            states_next_list.append((t_next.clone(), s.clone(), v.clone(),
                                   q.clone(), c.clone()))
            
            if return_trajectories:
                traj_s.append(s.clone())
                traj_v.append(v.clone())
                traj_q.append(q.clone())
        
        # Paths that never exited: terminal condition at T
        still_alive = alive
        s_exit = torch.where(still_alive, s, s_exit)
        v_exit = torch.where(still_alive, v, v_exit)
        q_exit = torch.where(still_alive, q, q_exit)
        c_exit = torch.where(still_alive, c, c_exit)
        
        # Compute terminal reward
        rewards = self.reward(s_exit, v_exit, q_exit, c_exit)
        
        result = {
            'rewards': rewards,
            'log_probs_list': log_probs_list,
            'alive_masks': alive_masks,
            'states_list': states_list,
            'states_next_list': states_next_list,
        }
        
        if return_trajectories:
            result['traj_s'] = torch.stack(traj_s, dim=1)
            result['traj_v'] = torch.stack(traj_v, dim=1)
            result['traj_q'] = torch.stack(traj_q, dim=1)
        
        return result


# ============================================================================
# Algorithm 1: Stochastic Gradient Policy (SGP / REINFORCE)
# ============================================================================

def train_sgp(config: Config, verbose: bool = True):
    """
    Algorithm 1 from the paper: Stochastic Gradient Policy.
    
    Uses gradient representation (2.2):
    ∇_θ J(θ) = E[g(X_τ) * Σ_{t_i < τ} ∇_θ log ρ_θ(t_i, X_ti, α_ti)]
    """
    device = config.device
    env = ShareRepurchaseEnv(config)
    policy = PolicyNetwork(config).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=config.lr_policy)
    
    price_history = []
    
    for episode in range(config.n_episodes):
        optimizer.zero_grad()
        
        # Rollout batch of trajectories
        data = env.rollout(policy, config.batch_size)
        rewards = data['rewards']           # (K,)
        log_probs = data['log_probs_list']  # list of N tensors, each (K,)
        alive_masks = data['alive_masks']   # list of N tensors, each (K,)
        
        # Compute policy gradient estimate (eq 2.2):
        # Γ_θ^(k) = G^(k) * Σ_{t_i < τ} ∇_θ log ρ_θ(t_i, X_ti, α_ti)
        # We maximize J, so loss = -J for gradient descent
        
        # Sum of log probs weighted by alive mask
        weighted_log_prob = torch.zeros(config.batch_size, device=device)
        for i in range(config.N):
            weighted_log_prob += log_probs[i] * alive_masks[i]
        
        # REINFORCE loss: -E[G * Σ log ρ * 1_{t<τ}]
        # Use detached rewards as the "weight" (no gradient through rewards)
        loss = -(rewards.detach() * weighted_log_prob).mean()
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Log progress
        if verbose and (episode + 1) % 100 == 0:
            mean_reward = rewards.mean().item()
            price_history.append(mean_reward)
            print(f"SGP Episode {episode+1}/{config.n_episodes}, "
                  f"Mean PnL: {mean_reward:.6f}, "
                  f"Price (bp): {mean_reward * 1e4:.2f}")
    
    return policy, price_history


# ============================================================================
# Algorithm 2: Actor-Critic (Offline)
# ============================================================================

def train_ac_offline(config: Config, verbose: bool = True):
    """
    Algorithm 2 from the paper: Actor-Critic (offline).
    
    Uses gradient representation (2.7) with baseline:
    ∇_θ J(θ) = E[Σ_{t_i < τ} (V_φ(t_{i+1}, X_{i+1}) - V_φ(t_i, X_i)) 
                              * ∇_θ log ρ_θ(t_i, X_ti, α_ti)]
    
    Critic updated by minimizing TD error:
    E[(V_φ(t_{i+1}, X_{i+1}) - V_φ(t_i, X_i))^2 * 1_{X_ti ∈ O}]
    """
    device = config.device
    env = ShareRepurchaseEnv(config)
    policy = PolicyNetwork(config).to(device)
    value_fn = ValueNetwork(config).to(device)
    
    optimizer_policy = optim.Adam(policy.parameters(), lr=config.lr_policy)
    optimizer_value = optim.Adam(value_fn.parameters(), lr=config.lr_value)
    
    price_history = []
    
    for episode in range(config.n_episodes):
        # ---- Generate trajectories ----
        data = env.rollout(policy, config.batch_size)
        rewards = data['rewards']
        log_probs = data['log_probs_list']
        alive_masks = data['alive_masks']
        states = data['states_list']
        states_next = data['states_next_list']
        
        # ---- Compute TD errors and value targets ----
        # For each time step i, compute:
        #   δ_i = V_φ(t_{i+1}, X_{i+1}) - V_φ(t_i, X_i)
        # At exit: V_φ(τ, X_τ) = g(X_τ) (the reward)
        
        td_errors = []
        value_losses = []
        
        for i in range(config.N):
            t_i, s_i, v_i, q_i, c_i = states[i]
            t_next, s_next, v_next, q_next, c_next = states_next[i]
            
            # Current value
            v_curr = value_fn(t_i, s_i, v_i, q_i, c_i)
            
            # Next value: if exited at step i+1, use g(X_{i+1})
            # Check if path just exited at this step
            if i < config.N - 1:
                alive_next = alive_masks[i + 1] if i + 1 < config.N else torch.zeros_like(alive_masks[0])
            else:
                alive_next = torch.zeros_like(alive_masks[0])
            
            just_exited = alive_masks[i] * (1 - alive_next)
            still_alive_next = alive_masks[i] * alive_next
            
            # For paths that just exited or at terminal: use reward
            v_next_val = value_fn(t_next, s_next, v_next, q_next, c_next)
            
            # At exit boundary or terminal, the value is the reward
            reward_at_exit = env.reward(s_next, v_next, q_next, c_next)
            v_next_target = still_alive_next * v_next_val + just_exited * reward_at_exit
            
            # If this is the last step and still alive, terminal condition
            if i == config.N - 1:
                v_next_target = alive_masks[i] * reward_at_exit
            
            td_error = (v_next_target.detach() - v_curr) * alive_masks[i]
            td_errors.append(td_error)
            
            # Value loss: minimize TD error squared
            value_losses.append((td_error ** 2).mean())
        
        # ---- Actor update ----
        optimizer_policy.zero_grad()
        
        actor_loss = torch.tensor(0.0, device=device)
        for i in range(config.N):
            # Γ_θ = Σ (V_{i+1} - V_i) * ∇ log ρ * 1_{t_i < τ}
            actor_loss -= (td_errors[i].detach() * log_probs[i] * alive_masks[i]).mean()
        
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer_policy.step()
        
        # ---- Critic update ----
        optimizer_value.zero_grad()
        
        critic_loss = sum(value_losses) / config.N
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_fn.parameters(), max_norm=1.0)
        optimizer_value.step()
        
        # Log progress
        if verbose and (episode + 1) % 100 == 0:
            mean_reward = rewards.mean().item()
            price_history.append(mean_reward)
            print(f"AC Episode {episode+1}/{config.n_episodes}, "
                  f"Mean PnL: {mean_reward:.6f}, "
                  f"Price (bp): {mean_reward * 1e4:.2f}, "
                  f"Critic loss: {critic_loss.item():.6f}")
    
    return policy, value_fn, price_history


# ============================================================================
# Algorithm 3: Actor-Critic (Online)
# ============================================================================

def train_ac_online(config: Config, verbose: bool = True):
    """
    Algorithm 3 from the paper: Actor-Critic (online).
    
    Same as offline but updates parameters at each time step within
    a trajectory, not after the full rollout.
    """
    device = config.device
    env = ShareRepurchaseEnv(config)
    policy = PolicyNetwork(config).to(device)
    value_fn = ValueNetwork(config).to(device)
    
    optimizer_policy = optim.Adam(policy.parameters(), lr=config.lr_policy)
    optimizer_value = optim.Adam(value_fn.parameters(), lr=config.lr_value)
    
    cfg = config
    price_history = []
    
    for episode in range(config.n_episodes):
        # Initialize state
        batch = config.batch_size
        s = torch.full((batch,), cfg.S0, device=device)
        v = torch.full((batch,), cfg.S0, device=device)
        q = torch.zeros(batch, device=device)
        c = torch.zeros(batch, device=device)
        alive = torch.ones(batch, dtype=torch.bool, device=device)
        
        episode_rewards = torch.zeros(batch, device=device)
        
        for i in range(cfg.N):
            if not alive.any():
                break
            
            t_i = torch.full((batch,), i * cfg.dt, device=device)
            alive_float = alive.float()
            
            # Sample action
            with torch.no_grad():
                a = policy.sample_action(t_i, s, v, q, c)
            
            # Compute log prob (with gradients)
            log_p = policy.log_prob(t_i, s, v, q, c, a)
            
            # Current value
            v_curr = value_fn(t_i, s, v, q, c)
            
            # Simulate step
            dW = torch.randn(batch, device=device) * np.sqrt(cfg.dt)
            s_new, v_new, q_new, c_new = env.simulate_step(
                t_i, s, v, q, c, a, dW)
            
            t_next = torch.full((batch,), (i + 1) * cfg.dt, device=device)
            
            # Check exit
            just_exited = alive & (q_new >= cfg.B)
            terminal = alive & (i == cfg.N - 1) & ~just_exited
            
            # Next value
            with torch.no_grad():
                v_next_val = value_fn(t_next, s_new, v_new, q_new, c_new)
                reward_exit = env.reward(s_new, v_new, q_new, c_new)
            
            v_next_target = torch.where(
                just_exited | terminal,
                reward_exit,
                v_next_val
            )
            
            td_error = (v_next_target - v_curr.detach()) * alive_float
            
            # Actor update
            optimizer_policy.zero_grad()
            actor_loss = -(td_error.detach() * log_p * alive_float).mean()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer_policy.step()
            
            # Critic update
            optimizer_value.zero_grad()
            v_curr_for_critic = value_fn(t_i, s, v, q, c)
            critic_loss = ((v_next_target - v_curr_for_critic) ** 2 * alive_float).mean()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_fn.parameters(), max_norm=1.0)
            optimizer_value.step()
            
            # Track rewards for exited paths
            episode_rewards = torch.where(just_exited | terminal,
                                         reward_exit, episode_rewards)
            
            # Update state
            alive = alive & ~just_exited & ~terminal
            s = s_new
            v = v_new
            q = q_new
            c = c_new
        
        if verbose and (episode + 1) % 100 == 0:
            mean_reward = episode_rewards.mean().item()
            price_history.append(mean_reward)
            print(f"AC-Online Episode {episode+1}/{config.n_episodes}, "
                  f"Mean PnL: {mean_reward:.6f}, "
                  f"Price (bp): {mean_reward * 1e4:.2f}")
    
    return policy, value_fn, price_history


# ============================================================================
# Evaluation: Monte Carlo price computation
# ============================================================================

@torch.no_grad()
def evaluate_price(policy: PolicyNetwork, config: Config, 
                   n_mc: Optional[int] = None) -> Tuple[float, float]:
    """
    Evaluate the price P_BV using Monte Carlo under the learned policy.
    
    Returns:
        mean_price: E[PnL] (in raw units)
        std_price: standard deviation / sqrt(n_mc)
    """
    if n_mc is None:
        n_mc = config.n_mc
    
    env = ShareRepurchaseEnv(config)
    policy.eval()
    
    # Process in chunks to manage memory
    chunk_size = min(10000, n_mc)
    all_rewards = []
    
    for start in range(0, n_mc, chunk_size):
        end = min(start + chunk_size, n_mc)
        batch = end - start
        
        s = torch.full((batch,), config.S0, device=config.device)
        v = torch.full((batch,), config.S0, device=config.device)
        q = torch.zeros(batch, device=config.device)
        c = torch.zeros(batch, device=config.device)
        alive = torch.ones(batch, dtype=torch.bool, device=config.device)
        
        s_exit = torch.zeros(batch, device=config.device)
        v_exit = torch.zeros(batch, device=config.device)
        q_exit = torch.zeros(batch, device=config.device)
        c_exit = torch.zeros(batch, device=config.device)
        
        for i in range(config.N):
            t_i = torch.full((batch,), i * config.dt, device=config.device)
            
            # Use greedy policy (deterministic: pick action with higher prob)
            prob_buy = policy(t_i, s, v, q, c)
            a = (prob_buy > 0.5).float() * config.a_bar
            
            dW = torch.randn(batch, device=config.device) * np.sqrt(config.dt)
            s_new, v_new, q_new, c_new = env.simulate_step(
                t_i, s, v, q, c, a, dW)
            
            just_exited = alive & (q_new >= config.B)
            s_exit = torch.where(just_exited, s_new, s_exit)
            v_exit = torch.where(just_exited, v_new, v_exit)
            q_exit = torch.where(just_exited, q_new, q_exit)
            c_exit = torch.where(just_exited, c_new, c_exit)
            
            alive = alive & ~just_exited
            s, v, q, c = s_new, v_new, q_new, c_new
        
        # Terminal
        s_exit = torch.where(alive, s, s_exit)
        v_exit = torch.where(alive, v, v_exit)
        q_exit = torch.where(alive, q, q_exit)
        c_exit = torch.where(alive, c, c_exit)
        
        rewards = env.reward(s_exit, v_exit, q_exit, c_exit)
        all_rewards.append(rewards)
    
    all_rewards = torch.cat(all_rewards)
    mean_price = all_rewards.mean().item()
    std_price = all_rewards.std().item() / np.sqrt(n_mc)
    
    policy.train()
    return mean_price, std_price


# ============================================================================
# Experiment: Reproduce Figure 1 (price vs a_bar)
# ============================================================================

def experiment_price_vs_abar():
    """
    Reproduce Figure 1: P_BV × 10^4 for different values of a_bar.
    Without market impact (gamma=0) and transaction costs (beta=0).
    """
    a_bar_values = [5.04, 7.56, 10.08, 12.6, 15.12, 17.64, 20.16, 22.68, 25.2]
    
    sgp_prices = []
    sgp_stds = []
    ac_prices = []
    ac_stds = []
    
    for a_bar in a_bar_values:
        print(f"\n{'='*60}")
        print(f"Training with a_bar = {a_bar:.2f}")
        print(f"{'='*60}")
        
        config = Config(
            a_bar=a_bar,
            gamma=0.0,
            beta=0.0,
            sigma=0.2,
            n_episodes=3000,
            batch_size=64,
            lr_policy=1e-3,
        )
        
        # SGP
        print("\n--- SGP ---")
        policy_sgp, _ = train_sgp(config, verbose=True)
        price_sgp, std_sgp = evaluate_price(policy_sgp, config)
        sgp_prices.append(price_sgp)
        sgp_stds.append(std_sgp)
        print(f"SGP Price: {price_sgp * 1e4:.2f} bp ± {std_sgp * 1e4:.2f}")
        
        # AC Offline
        print("\n--- AC Offline ---")
        config_ac = Config(
            a_bar=a_bar,
            gamma=0.0,
            beta=0.0,
            sigma=0.2,
            n_episodes=3000,
            batch_size=32,
            lr_policy=1e-3,
            lr_value=1e-3,
        )
        policy_ac, _, _ = train_ac_offline(config_ac, verbose=True)
        price_ac, std_ac = evaluate_price(policy_ac, config_ac)
        ac_prices.append(price_ac)
        ac_stds.append(std_ac)
        print(f"AC Price: {price_ac * 1e4:.2f} bp ± {std_ac * 1e4:.2f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    a_bars = np.array(a_bar_values)
    sgp_p = np.array(sgp_prices) * 1e4
    sgp_s = np.array(sgp_stds) * 1e4
    ac_p = np.array(ac_prices) * 1e4
    ac_s = np.array(ac_stds) * 1e4
    
    ax.plot(a_bars, sgp_p, 'ro-', label='SGP', markersize=8)
    ax.fill_between(a_bars, sgp_p - sgp_s, sgp_p + sgp_s, alpha=0.2, color='red')
    ax.plot(a_bars, ac_p, 'gs-', label='AC Offline', markersize=8)
    ax.fill_between(a_bars, ac_p - ac_s, ac_p + ac_s, alpha=0.2, color='green')
    
    ax.set_xlabel(r'$\bar{a}$', fontsize=14)
    ax.set_ylabel(r'$P \times 10^4$', fontsize=14)
    ax.set_title(r'$P_{BV} \times 10^4$ vs $\bar{a}$ (no market impact)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure1_price_vs_abar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved figure1_price_vs_abar.png")
    
    return a_bar_values, sgp_prices, ac_prices


# ============================================================================
# Experiment: Reproduce Figure 2 (convergence)
# ============================================================================

def experiment_convergence():
    """
    Reproduce Figure 2: Convergence of P_BV × 10^4 as a function of episodes.
    """
    configs = [
        {"a_bar": 25.2, "sigma": 0.2, "label": r"$\bar{a}=25.2, \sigma=0.2$"},
        {"a_bar": 9.0, "sigma": 0.25, "label": r"$\bar{a}=9, \sigma=0.25$"},
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, cfg_dict in enumerate(configs):
        config = Config(
            a_bar=cfg_dict["a_bar"],
            sigma=cfg_dict["sigma"],
            gamma=0.0,
            beta=0.0,
            n_episodes=5000,
            batch_size=64,
            lr_policy=1e-3,
            lr_value=1e-3,
        )
        
        # Track prices during training
        eval_interval = 50
        
        # SGP
        print(f"\n--- SGP for {cfg_dict['label']} ---")
        env = ShareRepurchaseEnv(config)
        policy_sgp = PolicyNetwork(config).to(config.device)
        optimizer = optim.Adam(policy_sgp.parameters(), lr=config.lr_policy)
        
        sgp_episodes = []
        sgp_prices = []
        
        for ep in range(config.n_episodes):
            optimizer.zero_grad()
            data = env.rollout(policy_sgp, config.batch_size)
            rewards = data['rewards']
            log_probs = data['log_probs_list']
            alive_masks = data['alive_masks']
            
            weighted_lp = torch.zeros(config.batch_size, device=config.device)
            for i in range(config.N):
                weighted_lp += log_probs[i] * alive_masks[i]
            
            loss = -(rewards.detach() * weighted_lp).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_sgp.parameters(), 1.0)
            optimizer.step()
            
            if (ep + 1) % eval_interval == 0:
                price, _ = evaluate_price(policy_sgp, config, n_mc=10000)
                sgp_episodes.append(ep + 1)
                sgp_prices.append(price * 1e4)
                print(f"  EP {ep+1}: {price*1e4:.2f} bp")
        
        # AC Offline
        print(f"\n--- AC Offline for {cfg_dict['label']} ---")
        config_ac = Config(
            a_bar=cfg_dict["a_bar"],
            sigma=cfg_dict["sigma"],
            gamma=0.0,
            beta=0.0,
            n_episodes=5000,
            batch_size=32,
            lr_policy=1e-3,
            lr_value=1e-3,
        )
        
        policy_ac, value_fn, ac_hist = train_ac_offline(config_ac, verbose=False)
        
        ac_episodes = []
        ac_prices_list = []
        
        # Re-train with tracking
        policy_ac2 = PolicyNetwork(config_ac).to(config_ac.device)
        value_fn2 = ValueNetwork(config_ac).to(config_ac.device)
        opt_p = optim.Adam(policy_ac2.parameters(), lr=config_ac.lr_policy)
        opt_v = optim.Adam(value_fn2.parameters(), lr=config_ac.lr_value)
        env_ac = ShareRepurchaseEnv(config_ac)
        
        for ep in range(config_ac.n_episodes):
            data = env_ac.rollout(policy_ac2, config_ac.batch_size)
            rewards = data['rewards']
            log_probs = data['log_probs_list']
            alive_masks = data['alive_masks']
            states = data['states_list']
            states_next = data['states_next_list']
            
            td_errors = []
            v_losses = []
            for i in range(config_ac.N):
                t_i, s_i, v_i, q_i, c_i = states[i]
                t_n, s_n, v_n, q_n, c_n = states_next[i]
                vc = value_fn2(t_i, s_i, v_i, q_i, c_i)
                
                if i < config_ac.N - 1:
                    an = alive_masks[i + 1]
                else:
                    an = torch.zeros_like(alive_masks[0])
                je = alive_masks[i] * (1 - an)
                sa = alive_masks[i] * an
                
                vn = value_fn2(t_n, s_n, v_n, q_n, c_n)
                re = env_ac.reward(s_n, v_n, q_n, c_n)
                vnt = sa * vn + je * re
                if i == config_ac.N - 1:
                    vnt = alive_masks[i] * re
                
                td = (vnt.detach() - vc) * alive_masks[i]
                td_errors.append(td)
                v_losses.append((td ** 2).mean())
            
            opt_p.zero_grad()
            al = torch.tensor(0.0, device=config_ac.device)
            for i in range(config_ac.N):
                al -= (td_errors[i].detach() * log_probs[i] * alive_masks[i]).mean()
            al.backward()
            torch.nn.utils.clip_grad_norm_(policy_ac2.parameters(), 1.0)
            opt_p.step()
            
            opt_v.zero_grad()
            cl = sum(v_losses) / config_ac.N
            cl.backward()
            torch.nn.utils.clip_grad_norm_(value_fn2.parameters(), 1.0)
            opt_v.step()
            
            if (ep + 1) % eval_interval == 0:
                price, _ = evaluate_price(policy_ac2, config_ac, n_mc=10000)
                ac_episodes.append(ep + 1)
                ac_prices_list.append(price * 1e4)
                print(f"  AC EP {ep+1}: {price*1e4:.2f} bp")
        
        ax = axes[idx]
        ax.plot(sgp_episodes, sgp_prices, 'r-', label='SGP', alpha=0.8)
        ax.plot(ac_episodes, ac_prices_list, 'b-', label='AC Offline', alpha=0.8)
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel(r'$P \times 10^4$', fontsize=12)
        ax.set_title(cfg_dict['label'], fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure2_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved figure2_convergence.png")


# ============================================================================
# Experiment: Visualize optimal policy surface (Figures 3 & 4)
# ============================================================================

def experiment_policy_surface(policy: PolicyNetwork, config: Config, 
                              title_suffix: str = ""):
    """
    Visualize the optimal policy ρ_θ(t, x, a_bar) as a function of (V, Q)
    for fixed S and two values of t.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    device = config.device
    
    V_range = np.linspace(0.1, 2.0, 40)
    Q_range = np.linspace(0.1, 0.9, 40)
    V_grid, Q_grid = np.meshgrid(V_range, Q_range)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), subplot_kw={'projection': '3d'})
    
    for idx, t_val in enumerate([config.T - config.dt, config.T / 2]):
        n = V_grid.size
        t_tensor = torch.full((n,), t_val, device=device)
        s_tensor = torch.full((n,), config.S0, device=device)
        v_tensor = torch.tensor(V_grid.flatten(), dtype=torch.float32, device=device)
        q_tensor = torch.tensor(Q_grid.flatten(), dtype=torch.float32, device=device)
        c_tensor = torch.zeros(n, device=device)
        
        with torch.no_grad():
            prob = policy(t_tensor, s_tensor, v_tensor, q_tensor, c_tensor)
        
        prob_grid = prob.cpu().numpy().reshape(V_grid.shape)
        
        ax = axes[idx]
        ax.plot_surface(V_grid, Q_grid, prob_grid, cmap='RdYlGn', alpha=0.8)
        ax.set_xlabel('V')
        ax.set_ylabel('Q')
        ax.set_zlabel(r'$\rho_\theta$')
        t_label = "t = T - dt" if idx == 0 else "t = T/2"
        ax.set_title(f'{t_label} {title_suffix}')
    
    plt.tight_layout()
    fname = f'policy_surface{title_suffix.replace(" ", "_")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")


# ============================================================================
# Experiment: Visualize optimal inventory trajectories (Figures 5 & 6)
# ============================================================================

@torch.no_grad()
def experiment_trajectories(policy: PolicyNetwork, config: Config,
                           title_suffix: str = ""):
    """
    Plot optimal repurchase strategy evolution for sample price realizations.
    """
    env = ShareRepurchaseEnv(config)
    device = config.device
    n_paths = 2
    
    fig, axes = plt.subplots(2, n_paths, figsize=(12, 8))
    
    for path_idx in range(n_paths):
        torch.manual_seed(42 + path_idx * 7)
        
        s_hist = [config.S0]
        v_hist = [config.S0]
        q_hist = [0.0]
        
        s = torch.tensor([config.S0], device=device)
        v = torch.tensor([config.S0], device=device)
        q = torch.tensor([0.0], device=device)
        c = torch.tensor([0.0], device=device)
        done = False
        
        for i in range(config.N):
            if done:
                s_hist.append(s_hist[-1])
                v_hist.append(v_hist[-1])
                q_hist.append(q_hist[-1])
                continue
            
            t_i = torch.tensor([i * config.dt], device=device)
            prob = policy(t_i, s, v, q, c)
            a = (prob > 0.5).float() * config.a_bar
            
            dW = torch.randn(1, device=device) * np.sqrt(config.dt)
            s, v, q, c = env.simulate_step(t_i, s, v, q, c, a, dW)
            
            s_hist.append(s.item())
            v_hist.append(v.item())
            q_hist.append(q.item())
            
            if q.item() >= config.B:
                done = True
        
        times = np.arange(config.N + 1)
        
        # Inventory plot
        ax_q = axes[0, path_idx]
        ax_q.plot(times, q_hist, 'g-', linewidth=2)
        ax_q.axhline(y=config.B, color='k', linestyle='--', alpha=0.5)
        ax_q.set_ylabel('Inventory')
        ax_q.set_title(f'Path {path_idx + 1} {title_suffix}')
        ax_q.grid(True, alpha=0.3)
        
        # Spot/VWAP plot
        ax_s = axes[1, path_idx]
        ax_s.plot(times, s_hist, 'b-', label='S', linewidth=1.5)
        ax_s.plot(times, v_hist, 'r-', label='V', linewidth=1.5)
        ax_s.set_xlabel('Time (days)')
        ax_s.set_ylabel('Spot')
        ax_s.legend()
        ax_s.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fname = f'trajectories{title_suffix.replace(" ", "_")}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")


# ============================================================================
# Quick demo: train and evaluate a single configuration
# ============================================================================

def quick_demo():
    """
    Quick demo: train SGP and AC for a single (a_bar, sigma) and report price.
    """
    print("=" * 70)
    print("QUICK DEMO: Share Repurchase Pricing via Policy Gradient")
    print("=" * 70)
    
    config = Config(
        a_bar=25.2,
        sigma=0.2,
        gamma=0.0,
        beta=0.0,
        n_episodes=2000,
        batch_size=64,
        lr_policy=1e-3,
    )
    
    print(f"\nParameters: a_bar={config.a_bar}, sigma={config.sigma}, "
          f"gamma={config.gamma}, T={config.T_days} days, B={config.B}")
    print(f"Device: {config.device}")
    
    # --- SGP ---
    print("\n" + "=" * 40)
    print("Training SGP (Algorithm 1)")
    print("=" * 40)
    t0 = time.time()
    policy_sgp, hist_sgp = train_sgp(config)
    t_sgp = time.time() - t0
    
    price_sgp, std_sgp = evaluate_price(policy_sgp, config)
    print(f"\nSGP Final Price: {price_sgp * 1e4:.2f} ± {std_sgp * 1e4:.2f} bp")
    print(f"SGP Training time: {t_sgp:.1f}s")
    
    # --- AC Offline ---
    print("\n" + "=" * 40)
    print("Training AC Offline (Algorithm 2)")
    print("=" * 40)
    config_ac = Config(
        a_bar=25.2,
        sigma=0.2,
        gamma=0.0,
        beta=0.0,
        n_episodes=2000,
        batch_size=32,
        lr_policy=1e-3,
        lr_value=1e-3,
    )
    t0 = time.time()
    policy_ac, value_fn, hist_ac = train_ac_offline(config_ac)
    t_ac = time.time() - t0
    
    price_ac, std_ac = evaluate_price(policy_ac, config_ac)
    print(f"\nAC Final Price: {price_ac * 1e4:.2f} ± {std_ac * 1e4:.2f} bp")
    print(f"AC Training time: {t_ac:.1f}s")
    
    # --- Visualizations ---
    print("\nGenerating visualizations...")
    experiment_policy_surface(policy_sgp, config, title_suffix="(no impact)")
    experiment_trajectories(policy_sgp, config, title_suffix="(no impact)")
    
    # --- With market impact ---
    print("\n" + "=" * 40)
    print("Training SGP with market impact (gamma=0.1)")
    print("=" * 40)
    config_impact = Config(
        a_bar=25.2,
        sigma=0.2,
        gamma=0.1,
        beta=0.0,
        n_episodes=2000,
        batch_size=64,
        lr_policy=1e-3,
    )
    policy_impact, _ = train_sgp(config_impact)
    price_impact, std_impact = evaluate_price(policy_impact, config_impact)
    print(f"\nSGP (impact) Price: {price_impact * 1e4:.2f} ± {std_impact * 1e4:.2f} bp")
    
    experiment_policy_surface(policy_impact, config_impact, 
                             title_suffix="(gamma=0.1)")
    experiment_trajectories(policy_impact, config_impact,
                           title_suffix="(gamma=0.1)")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - policy_surface(no_impact).png")
    print("  - trajectories(no_impact).png")
    print("  - policy_surface(gamma=0.1).png")
    print("  - trajectories(gamma=0.1).png")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "demo":
            quick_demo()
        elif mode == "figure1":
            experiment_price_vs_abar()
        elif mode == "figure2":
            experiment_convergence()
        elif mode == "full":
            quick_demo()
            experiment_price_vs_abar()
            experiment_convergence()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python share_repurchase_pg.py [demo|figure1|figure2|full]")
    else:
        quick_demo()
