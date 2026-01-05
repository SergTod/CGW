"""
CGW Model: Cognitive Global Workspace with recurrent reasoning loop.

Core semantics:
- Workspace S_t is persistent across steps
- Updated based on selected specialist output
- State updates are NOT identity functions, pure noise, or single-pass
"""

import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass, field

from fastmem import FastMem
from router import Router
from specialists import Specialists


@dataclass
class StepInfo:
    """Per-step logging information."""
    step_index: int
    selected: int
    selected_name: str
    routing_probs: torch.Tensor
    routing_entropy: float
    reads: int
    writes: int
    workspace_norm: float
    halt_logit: Optional[float] = None


@dataclass 
class RunInfo:
    """Per-run logging information."""
    success: bool
    trajectory_length: int
    specialist_usage: dict[str, int] = field(default_factory=dict)
    step_infos: list[StepInfo] = field(default_factory=list)
    fastmem_stats: dict = field(default_factory=dict)
    mean_entropy: float = 0.0
    std_entropy: float = 0.0
    final_output: Optional[torch.Tensor] = None
    # (B, steps) tensor of per-step halting logits (detached, cpu)
    halt_logits: Optional[torch.Tensor] = None


class CGW(nn.Module):
    """
    Cognitive Global Workspace model.
    
    Executes a recurrent reasoning loop:
    1. Router selects specialist based on current workspace state
    2. Selected specialist produces workspace delta
    3. Workspace is updated: S_{t+1} = S_t + delta_S
    4. Repeat until halting or max steps
    """
    
    def __init__(
        self,
        input_dim: int,
        workspace_dim: int,
        hidden_dim: int = 128,
        tau: float = 1.0,
        max_steps: int = 20,
    ):
        """
        Initialize CGW.
        
        Args:
            input_dim: Dimension of input observations
            workspace_dim: Dimension of workspace state vector
            hidden_dim: Hidden dimension for specialist MLPs
            tau: Gumbel-Softmax temperature
            max_steps: Maximum reasoning steps per sample
        """
        super().__init__()
        self.input_dim = input_dim
        self.workspace_dim = workspace_dim
        self.max_steps = max_steps
        
        # Core components
        self.router = Router(workspace_dim, tau=tau)
        self.specialists = Specialists(input_dim, workspace_dim, hidden_dim)
        
        # Output projection
        self.output_head = nn.Linear(workspace_dim, input_dim)
        
        # Learnable initial workspace state
        self.S_init = nn.Parameter(torch.randn(workspace_dim) * 0.1)
    
    def cgw_step(
        self,
        S_t: torch.Tensor,
        mem: FastMem,
        x: torch.Tensor,
        step_idx: int,
    ) -> tuple[torch.Tensor, StepInfo, torch.Tensor]:
        """
        Execute one CGW reasoning step.
        
        This is the canonical workspace update loop from prompt.md.
        
        Args:
            S_t: Current workspace state [batch, workspace_dim]
            mem: FastMem instance
            x: Input observation [batch, input_dim]
            step_idx: Current step index
        
        Returns:
            S_next: Updated workspace state
            step_info: Logging information
            halt_logit: Halting logit for this step [batch, 1]
        """
        # Route based on current workspace state
        routing_probs, selected, routing_logits, soft_probs = self.router(S_t)
        
        # Compute entropy from SOFT probabilities (not logits!) for accurate measurement
        entropy = self.router.get_entropy_from_probs(soft_probs)
        entropy_bits = float(entropy.mean().item())
        
        # Track memory deltas before specialist execution
        # (delta tracking is done inside FastMem)
        
        # Execute selected specialist to update the workspace
        delta_S, _ = self.specialists(
            selected=selected,
            S_t=S_t,
            x=x,
            mem=mem,
        )
        
        # Update workspace: S_{t+1} = S_t + delta_S
        S_next = S_t + delta_S
        
        # Get memory operation counts for this step
        reads = mem.read_count_delta()
        writes = mem.write_count_delta()
        
        # Compute halting logit every step (independent of routing selection)
        # This is used for evaluation-time adaptive step accounting.
        _, halt_logit = self.specialists.sup(S_next)

        # Build step info
        step_info = StepInfo(
            step_index=step_idx,
            selected=selected,
            selected_name=self.specialists.get_name(selected),
            routing_probs=routing_probs.detach().cpu(),
            routing_entropy=entropy_bits,
            reads=reads,
            writes=writes,
            workspace_norm=float(torch.norm(S_next).item()),
            halt_logit=float(halt_logit.mean().item()),
        )

        return S_next, step_info, halt_logit
    
    def forward(
        self,
        x: torch.Tensor,
        halt_threshold: float = 0.5,
        min_steps: int = 3,
        return_trajectory: bool = False,
    ) -> tuple[torch.Tensor, RunInfo, torch.Tensor]:
        """
        Run full CGW reasoning trajectory.
        
        Args:
            x: Input observation [batch, input_dim]
            halt_threshold: Threshold for halting (sigmoid of halt_logit)
            min_steps: Minimum steps before halting is allowed
            return_trajectory: If True, store all workspace states
        
        Returns:
            output: Model output [batch, input_dim]
            run_info: Per-run logging information
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize workspace state
        S_t = self.S_init.unsqueeze(0).expand(batch_size, -1).clone()
        S_t = S_t.to(device)
        
        # Initialize fresh memory for this run
        mem = FastMem()
        
        # Track trajectory
        step_infos: list[StepInfo] = []
        specialist_usage = {name: 0 for name in self.specialists.SPECIALIST_NAMES}
        trajectory: list[torch.Tensor] = []
        
        # Reasoning loop
        halt_logits: list[torch.Tensor] = []  # list of (B,)
        for t in range(self.max_steps):
            if return_trajectory:
                trajectory.append(S_t.detach().clone())
            
            S_t, step_info, halt_logit = self.cgw_step(S_t, mem, x, t)
            step_infos.append(step_info)
            specialist_usage[step_info.selected_name] += 1

            # Store per-step halting logits for evaluation
            halt_logits.append(halt_logit.squeeze(-1))
            
            # NOTE: We intentionally do *not* early-break here.
            # Phase 0.5 evaluation measures adaptive step usage from halt_logits.
        
        # Compute output from final workspace state
        output = self.output_head(S_t)
        
        # Compute entropy statistics
        entropies = [si.routing_entropy for si in step_infos]
        mean_entropy = sum(entropies) / len(entropies) if entropies else 0.0
        std_entropy = (
            (sum((e - mean_entropy) ** 2 for e in entropies) / len(entropies)) ** 0.5
            if len(entropies) > 1 else 0.0
        )
        
        # Build run info
        run_info = RunInfo(
            success=False,  # Will be updated by caller based on task
            trajectory_length=len(step_infos),
            specialist_usage=specialist_usage,
            step_infos=step_infos,
            fastmem_stats=mem.get_stats(),
            mean_entropy=mean_entropy,
            std_entropy=std_entropy,
            final_output=output.detach().cpu(),
            halt_logits=None,
        )

        halt_logits_t = torch.stack(halt_logits, dim=1)  # (B, steps)
        run_info.halt_logits = halt_logits_t.detach().cpu()

        return output, run_info, halt_logits_t
    
    def get_specialist_diversity(self, run_info: RunInfo) -> tuple[int, list[str]]:
        """
        Check how many specialists were used at least 10% of steps.
        
        Returns:
            count: Number of specialists meeting threshold
            names: Names of those specialists
        """
        total_steps = run_info.trajectory_length
        threshold = max(1, int(0.1 * total_steps))
        
        active = []
        for name, count in run_info.specialist_usage.items():
            if count >= threshold:
                active.append(name)
        
        return len(active), active
