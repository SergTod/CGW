"""
Router: State-dependent routing via Gumbel-Softmax gating.

Critical requirements:
- Routing decision is a function of current workspace state S_t
- Different workspace states → may produce different routing decisions
- Same workspace state (deterministically) → same routing decision
- Routing must NOT be: uniform random, round-robin, or hardcoded sequence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def gumbel_softmax(
    logits: torch.Tensor, 
    tau: float = 1.0, 
    hard: bool = True,
    dim: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gumbel-Softmax with optional straight-through estimator.
    
    Args:
        logits: Unnormalized log probabilities
        tau: Temperature parameter (lower = more discrete)
        hard: If True, use straight-through estimator for hard selection
        dim: Dimension to apply softmax
    
    Returns:
        output: Hard or soft sample (used for selection)
        soft_probs: Soft probabilities (used for entropy calculation)
    """
    # Sample Gumbel noise
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / tau
    
    y_soft = F.softmax(gumbels, dim=dim)
    
    if hard:
        # Straight-through: take argmax but keep gradients from soft
        index = y_soft.argmax(dim=dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return y_hard - y_soft.detach() + y_soft, y_soft
    
    return y_soft, y_soft


class Router(nn.Module):
    """
    State-dependent router using Gumbel-Softmax gating.
    
    Routes to one of 4 specialists based on workspace state:
    - 0: io (input/output encoding)
    - 1: logic (reasoning)
    - 2: mem (memory operations)
    - 3: sup (supervision/halting)
    """
    
    NUM_SPECIALISTS = 4
    
    def __init__(self, workspace_dim: int, tau: float = 0.5):
        """
        Initialize router.
        
        Args:
            workspace_dim: Dimension of workspace state
            tau: Gumbel-Softmax temperature (lower = more discrete)
        """
        super().__init__()
        self.workspace_dim = workspace_dim
        self.tau = tau
        
        # Routing weight matrix: W_route @ S_t -> routing_logits
        self.W_route = nn.Linear(workspace_dim, self.NUM_SPECIALISTS, bias=True)
        
        # Initialize with larger weights for more varied initial routing
        nn.init.xavier_uniform_(self.W_route.weight, gain=2.0)
        # Initialize biases to encourage different specialists initially
        nn.init.constant_(self.W_route.bias, 0.0)
    
    def forward(
        self, 
        S_t: torch.Tensor, 
        hard: bool = True,
        tau: Optional[float] = None
    ) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """
        Compute routing decision based on workspace state.
        
        Args:
            S_t: Current workspace state [batch, workspace_dim]
            hard: If True, return hard one-hot selection
            tau: Override temperature (uses self.tau if None)
        
        Returns:
            routing_probs: Hard probability distribution (one-hot for selection) [batch, 4]
            selected: Index of selected specialist (int)
            routing_logits: Raw logits before softmax [batch, 4]
            soft_probs: Soft probabilities for entropy calculation [batch, 4]
        """
        tau = tau if tau is not None else self.tau
        
        # Compute routing logits from workspace state
        routing_logits = self.W_route(S_t)  # [batch, 4]
        
        # Apply Gumbel-Softmax - returns (hard_probs, soft_probs)
        routing_probs, soft_probs = gumbel_softmax(routing_logits, tau=tau, hard=hard)
        
        # Get selected specialist (argmax)
        selected = int(routing_probs[0].argmax().item())
        
        return routing_probs, selected, routing_logits, soft_probs
    
    def get_entropy(self, routing_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute routing entropy in bits from logits.
        
        Low entropy (< 0.5 bits): Routing collapsed (one specialist dominates)
        High entropy (> 1.9 bits): Routing uniform (no specialization)
        Healthy range: 0.5 - 1.9 bits (max possible = log₂(4) = 2.0 bits)
        
        Args:
            routing_logits: Raw logits [batch, 4]
        
        Returns:
            Entropy in bits [batch]
        """
        probs = F.softmax(routing_logits, dim=-1)
        return self.get_entropy_from_probs(probs)
    
    def get_entropy_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy from probability distribution.
        
        Args:
            probs: Probability distribution [batch, 4]
        
        Returns:
            Entropy in bits [batch]
        """
        # Add small epsilon to avoid log(0)
        log_probs = torch.log2(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy
    
    def set_tau(self, tau: float) -> None:
        """Update temperature parameter."""
        self.tau = tau
