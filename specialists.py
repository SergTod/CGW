"""
Specialists: Four distinct modules with specific responsibilities.

Each specialist:
- Reads workspace state S_t
- Produces output that meaningfully changes S_{t+1}
"""

import torch
import torch.nn as nn
from typing import Optional

from fastmem import FastMem


class IOSpecialist(nn.Module):
    """
    Encodes input tokens/observations into workspace updates.
    
    Responsibility: Transform external input into workspace-compatible representation.
    """
    
    def __init__(self, input_dim: int, workspace_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, workspace_dim),
        )
    
    def forward(self, x: torch.Tensor, S_t: torch.Tensor) -> torch.Tensor:
        """
        Encode input and produce workspace delta.
        
        Args:
            x: Input observation [batch, input_dim]
            S_t: Current workspace state [batch, workspace_dim]
        
        Returns:
            delta_S: Workspace update [batch, workspace_dim]
        """
        encoded = self.encoder(x)
        return encoded


class LogicSpecialist(nn.Module):
    """
    Applies learned transformations to workspace state.
    
    Responsibility: Perform reasoning/computation on current state.
    """
    
    def __init__(self, workspace_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(workspace_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, workspace_dim),
        )
    
    def forward(self, S_t: torch.Tensor) -> torch.Tensor:
        """
        Apply learned transformation to workspace.
        
        Args:
            S_t: Current workspace state [batch, workspace_dim]
        
        Returns:
            delta_S: Workspace update [batch, workspace_dim]
        """
        return self.mlp(S_t)


class MemSpecialist(nn.Module):
    """
    Reads from and writes to FastMem.
    
    Responsibility: Manage memory operations - store intermediate results,
    retrieve previously computed values. Values read MUST influence computation.
    """
    
    def __init__(self, workspace_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Project workspace to key space
        self.key_proj = nn.Linear(workspace_dim, workspace_dim)
        # Project workspace to value space
        self.val_proj = nn.Linear(workspace_dim, workspace_dim)
        # Merge read value with workspace
        self.merge = nn.Sequential(
            nn.Linear(workspace_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, workspace_dim),
        )
    
    def forward(
        self, 
        S_t: torch.Tensor, 
        mem: FastMem,
        write_mode: bool = True
    ) -> torch.Tensor:
        """
        Perform memory read/write and produce workspace delta.
        
        Args:
            S_t: Current workspace state [batch, workspace_dim]
            mem: FastMem instance
            write_mode: If True, also writes to memory
        
        Returns:
            delta_S: Workspace update [batch, workspace_dim]
        """
        batch_size = S_t.shape[0]
        workspace_dim = S_t.shape[1]
        device = S_t.device
        
        # Process each item in batch
        deltas = []
        for i in range(batch_size):
            s_i = S_t[i:i+1]  # Keep batch dim [1, workspace_dim]
            
            # Generate key from current state
            key = self.key_proj(s_i)
            
            # Read from memory
            read_value = mem.read(key.squeeze(0))
            
            if read_value is None:
                read_value = torch.zeros(workspace_dim, device=device)
            else:
                read_value = read_value.to(device)
            
            read_value = read_value.unsqueeze(0)  # [1, workspace_dim]
            
            # Write to memory if in write mode
            if write_mode:
                write_value = self.val_proj(s_i)
                mem.write(key.squeeze(0), write_value.squeeze(0))
            
            # Merge read value with workspace to produce delta
            merged_input = torch.cat([s_i, read_value], dim=-1)
            delta = self.merge(merged_input)
            deltas.append(delta)
        
        return torch.cat(deltas, dim=0)


class SupSpecialist(nn.Module):
    """
    Emits halting signal or confidence score.
    
    Responsibility: Decide when reasoning is complete, provide confidence
    in current workspace state.
    """
    
    def __init__(self, workspace_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(workspace_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, workspace_dim),
        )
        self.halt_head = nn.Linear(workspace_dim, 1)
    
    def forward(self, S_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute workspace delta and halting logit.
        
        Args:
            S_t: Current workspace state [batch, workspace_dim]
        
        Returns:
            delta_S: Workspace update [batch, workspace_dim]
            halt_logit: Halting decision logit [batch, 1]
        """
        delta_S = self.mlp(S_t)
        halt_logit = self.halt_head(S_t)
        return delta_S, halt_logit


class Specialists(nn.Module):
    """
    Container for all four specialists with unified interface.
    """
    
    SPECIALIST_NAMES = ["io", "logic", "mem", "sup"]
    
    def __init__(self, input_dim: int, workspace_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.io = IOSpecialist(input_dim, workspace_dim, hidden_dim)
        self.logic = LogicSpecialist(workspace_dim, hidden_dim)
        self.mem = MemSpecialist(workspace_dim, hidden_dim)
        self.sup = SupSpecialist(workspace_dim, hidden_dim)
        
        self._specialists = [self.io, self.logic, self.mem, self.sup]
    
    def forward(
        self,
        selected: int,
        S_t: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        mem: Optional[FastMem] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Execute selected specialist.
        
        Args:
            selected: Index of specialist to run (0=io, 1=logic, 2=mem, 3=sup)
            S_t: Current workspace state
            x: Input observation (required for io specialist)
            mem: FastMem instance (required for mem specialist)
        
        Returns:
            delta_S: Workspace update
            halt_logit: Halting logit (only for sup, else None)
        """
        halt_logit = None
        
        if selected == 0:  # io
            if x is None:
                raise ValueError("IOSpecialist requires input x")
            delta_S = self.io(x, S_t)
        
        elif selected == 1:  # logic
            delta_S = self.logic(S_t)
        
        elif selected == 2:  # mem
            if mem is None:
                raise ValueError("MemSpecialist requires FastMem instance")
            delta_S = self.mem(S_t, mem)
        
        elif selected == 3:  # sup
            delta_S, halt_logit = self.sup(S_t)
        
        else:
            raise ValueError(f"Invalid specialist index: {selected}")
        
        return delta_S, halt_logit
    
    def get_name(self, index: int) -> str:
        """Get specialist name by index."""
        return self.SPECIALIST_NAMES[index]
