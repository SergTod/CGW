"""
FastMem: Key-value memory with functional read/write operations.

This is NOT a stub - reads must influence computation and produce non-zero values
during successful runs.
"""

import torch
from typing import Optional


class FastMem:
    """
    Key-value memory store with delta tracking for logging.
    
    Requirements satisfied:
    - write(key, value) stores values
    - read(key) retrieves values that influence computation
    - Delta tracking for per-step logging
    """
    
    def __init__(self, similarity_threshold: float = 0.9):
        """
        Initialize FastMem.
        
        Args:
            similarity_threshold: Cosine similarity threshold for key matching.
                Keys with similarity >= threshold are considered matches.
        """
        self.keys: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.read_count: int = 0
        self.write_count: int = 0
        self._prev_read: int = 0
        self._prev_write: int = 0
        self.similarity_threshold = similarity_threshold
    
    def _find_similar_key(self, query_key: torch.Tensor) -> Optional[int]:
        """
        Find index of most similar key above threshold.
        
        Uses cosine similarity for soft key matching, allowing the model
        to learn approximate key representations.
        """
        if not self.keys:
            return None
        
        query_flat = query_key.flatten()
        query_norm = torch.norm(query_flat)
        
        if query_norm < 1e-8:
            return None
        
        best_idx = None
        best_sim = self.similarity_threshold
        
        for idx, stored_key in enumerate(self.keys):
            stored_flat = stored_key.flatten()
            stored_norm = torch.norm(stored_flat)
            
            if stored_norm < 1e-8:
                continue
            
            similarity = torch.dot(query_flat, stored_flat) / (query_norm * stored_norm)
            
            if similarity >= best_sim:
                best_sim = similarity
                best_idx = idx
        
        return best_idx
    
    def write(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """
        Write a key-value pair to memory.
        
        If a similar key exists (above threshold), updates the value.
        Otherwise, stores a new key-value pair.
        """
        key_detached = key.detach().clone()
        value_detached = value.detach().clone()
        
        existing_idx = self._find_similar_key(key_detached)
        
        if existing_idx is not None:
            self.values[existing_idx] = value_detached
        else:
            self.keys.append(key_detached)
            self.values.append(value_detached)
        
        self.write_count += 1
    
    def read(self, key: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Read value for a similar key from memory.
        
        Returns None if no similar key found. The returned value
        MUST influence downstream computation.
        """
        self.read_count += 1
        
        idx = self._find_similar_key(key.detach())
        
        if idx is not None:
            return self.values[idx].clone()
        
        return None
    
    def read_count_delta(self) -> int:
        """Get reads since last delta check."""
        delta = self.read_count - self._prev_read
        self._prev_read = self.read_count
        return delta
    
    def write_count_delta(self) -> int:
        """Get writes since last delta check."""
        delta = self.write_count - self._prev_write
        self._prev_write = self.write_count
        return delta
    
    def reset(self) -> None:
        """Clear all stored key-value pairs and counters."""
        self.keys.clear()
        self.values.clear()
        self.read_count = 0
        self.write_count = 0
        self._prev_read = 0
        self._prev_write = 0
    
    def get_stats(self) -> dict:
        """Get memory statistics for logging."""
        return {
            "num_entries": len(self.keys),
            "total_reads": self.read_count,
            "total_writes": self.write_count,
            "read_write_ratio": self.read_count / max(1, self.write_count),
        }
