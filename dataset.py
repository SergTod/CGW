"""
Toy Reasoning Dataset for Phase 0.

Task: 3-digit addition (e.g., 123 + 456 = 579)

Difficulty distribution:
- Easy (60%): Small numbers, no carries
- Medium (30%): Some carries
- Hard (10%): Large numbers, multiple carries
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Literal
from dataclasses import dataclass
import random


@dataclass
class Sample:
    """A single dataset sample."""
    a: int
    b: int
    result: int
    difficulty: Literal["easy", "medium", "hard"]
    input_tensor: torch.Tensor
    target_tensor: torch.Tensor


class AdditionDataset(Dataset):
    """
    3-digit addition dataset with difficulty labels.
    
    Encoding: Each number is encoded as a sequence of digits,
    normalized to [0, 1] range. Input is concatenation of a and b,
    target is the result.
    """
    
    def __init__(
        self, 
        size: int = 1000, 
        seed: int = 42,
        max_digits: int = 3,
    ):
        """
        Initialize dataset.
        
        Args:
            size: Number of samples
            seed: Random seed for reproducibility
            max_digits: Maximum number of digits per operand
        """
        self.size = size
        self.seed = seed
        self.max_digits = max_digits
        self.samples: list[Sample] = []
        
        self._generate_samples()
    
    def _count_carries(self, a: int, b: int) -> int:
        """Count number of carries in addition."""
        carries = 0
        carry = 0
        while a > 0 or b > 0:
            digit_sum = (a % 10) + (b % 10) + carry
            if digit_sum >= 10:
                carries += 1
                carry = 1
            else:
                carry = 0
            a //= 10
            b //= 10
        return carries
    
    def _classify_difficulty(self, a: int, b: int) -> Literal["easy", "medium", "hard"]:
        """Classify sample difficulty based on operands and carries."""
        carries = self._count_carries(a, b)
        result = a + b
        
        # Easy: no carries, small numbers
        if carries == 0 and result < 500:
            return "easy"
        # Hard: multiple carries or large result
        elif carries >= 2 or result >= 1500:
            return "hard"
        # Medium: some carries
        else:
            return "medium"
    
    def _encode_number(self, n: int, num_digits: int) -> torch.Tensor:
        """Encode number as normalized digit tensor."""
        digits = []
        for _ in range(num_digits):
            digits.append(n % 10)
            n //= 10
        # Reverse to get most significant digit first
        digits = digits[::-1]
        # Normalize to [0, 1]
        return torch.tensor(digits, dtype=torch.float32) / 9.0
    
    def _generate_samples(self) -> None:
        """Generate dataset samples with target difficulty distribution."""
        rng = random.Random(self.seed)
        
        # Target distribution
        target_easy = int(0.6 * self.size)
        target_medium = int(0.3 * self.size)
        target_hard = self.size - target_easy - target_medium
        
        counts = {"easy": 0, "medium": 0, "hard": 0}
        targets = {"easy": target_easy, "medium": target_medium, "hard": target_hard}
        
        max_val = 10 ** self.max_digits - 1
        attempts = 0
        max_attempts = self.size * 100
        
        while len(self.samples) < self.size and attempts < max_attempts:
            attempts += 1
            
            # Generate random operands
            a = rng.randint(0, max_val)
            b = rng.randint(0, max_val)
            
            difficulty = self._classify_difficulty(a, b)
            
            # Check if we need more samples of this difficulty
            if counts[difficulty] >= targets[difficulty]:
                continue
            
            result = a + b
            
            # Encode input: [a_digits, b_digits]
            a_encoded = self._encode_number(a, self.max_digits)
            b_encoded = self._encode_number(b, self.max_digits)
            input_tensor = torch.cat([a_encoded, b_encoded])
            
            # Encode target: result digits (may need extra digit for overflow)
            result_digits = self.max_digits + 1  # Allow for carry overflow
            target_tensor = self._encode_number(result, result_digits)
            
            sample = Sample(
                a=a,
                b=b,
                result=result,
                difficulty=difficulty,
                input_tensor=input_tensor,
                target_tensor=target_tensor,
            )
            
            self.samples.append(sample)
            counts[difficulty] += 1
        
        # If we couldn't hit targets exactly, fill with any samples
        while len(self.samples) < self.size:
            a = rng.randint(0, max_val)
            b = rng.randint(0, max_val)
            result = a + b
            difficulty = self._classify_difficulty(a, b)
            
            a_encoded = self._encode_number(a, self.max_digits)
            b_encoded = self._encode_number(b, self.max_digits)
            input_tensor = torch.cat([a_encoded, b_encoded])
            target_tensor = self._encode_number(result, self.max_digits + 1)
            
            sample = Sample(
                a=a, b=b, result=result, difficulty=difficulty,
                input_tensor=input_tensor, target_tensor=target_tensor,
            )
            self.samples.append(sample)
        
        # Shuffle samples
        rng.shuffle(self.samples)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get a sample.
        
        Returns:
            input_tensor: Encoded operands [2 * max_digits]
            target_tensor: Encoded result [max_digits + 1]
            difficulty: Difficulty label
        """
        sample = self.samples[idx]
        return sample.input_tensor, sample.target_tensor, sample.difficulty
    
    def get_sample(self, idx: int) -> Sample:
        """Get full sample object."""
        return self.samples[idx]
    
    def get_difficulty_distribution(self) -> dict[str, int]:
        """Get count of samples per difficulty."""
        dist = {"easy": 0, "medium": 0, "hard": 0}
        for sample in self.samples:
            dist[sample.difficulty] += 1
        return dist


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, str]]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Collate function for DataLoader."""
    inputs, targets, difficulties = zip(*batch)
    return torch.stack(inputs), torch.stack(targets), list(difficulties)


def create_dataloaders(
    train_size: int = 800,
    val_size: int = 100,
    test_size: int = 100,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = AdditionDataset(size=train_size, seed=seed)
    val_dataset = AdditionDataset(size=val_size, seed=seed + 1)
    test_dataset = AdditionDataset(size=test_size, seed=seed + 2)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader


def decode_number(tensor: torch.Tensor) -> int:
    """Decode normalized digit tensor back to integer."""
    # Denormalize: [0, 1] -> [0, 9]
    digits = (tensor * 9.0).round().long().tolist()
    # Convert digit list to number
    result = 0
    for d in digits:
        result = result * 10 + max(0, min(9, d))
    return result
