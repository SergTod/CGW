"""
Training and evaluation loop for CGW Phase 0.

Semantic competence validation, not optimization exercise.
Uses simple, stable defaults.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
import time
import json

from cgw import CGW, RunInfo
from dataset import create_dataloaders, decode_number, AdditionDataset
from logger import CGWLogger


def compute_steps_used(
    halt_logits: torch.Tensor,  # (B, steps)
    min_steps: int,
    halt_threshold: float,
) -> torch.Tensor:
    """Compute per-sample steps_used from per-step halt logits.

    If a sample never crosses threshold, steps_used = steps (max_steps).
    """
    if halt_logits.ndim != 2:
        raise ValueError(f"halt_logits must have shape (B, steps); got {tuple(halt_logits.shape)}")

    B, steps = halt_logits.shape
    if steps == 0:
        return torch.zeros((B,), device=halt_logits.device, dtype=torch.long)

    halt_prob = torch.sigmoid(halt_logits)

    allowed = torch.zeros_like(halt_prob, dtype=torch.bool)
    if min_steps <= 1:
        allowed[:] = True
    else:
        allowed[:, min_steps - 1 :] = True

    will_halt = (halt_prob > halt_threshold) & allowed

    idx = torch.argmax(will_halt.int(), dim=1)
    has_any = will_halt.any(dim=1)
    idx = torch.where(has_any, idx, torch.full_like(idx, steps - 1))

    return idx + 1


@dataclass
class TrainConfig:
    """Training configuration with simple defaults."""
    # Model
    input_dim: int = 6  # 2 x 3 digits
    output_dim: int = 4  # 4 digits for result
    workspace_dim: int = 128  # Increased for more capacity
    hidden_dim: int = 128
    tau: float = 0.5  # Lower for more decisive routing
    max_steps: int = 10
    
    # Training
    epochs: int = 60
    batch_size: int = 32
    lr: float = 2e-4  # Lower for fine-tuning
    
    # Data
    train_size: int = 1200  # More training data
    val_size: int = 150
    test_size: int = 150
    
    # Halting head thresholds (used during training for shaping behaviour)
    halt_threshold: float = 0.7
    min_steps: int = 4

    # Evaluation-only adaptive step accounting
    adaptive_eval: bool = True
    eval_halt_threshold: float = 0.7
    eval_min_steps: int = 2
    
    # Logging
    log_dir: str = "logs"
    debug_mode: bool = False
    
    # Device
    device: str = "auto"
    
    # Resource guardrails
    max_sample_time: float = 60.0  # seconds
    max_memory_gb: float = 4.0


def get_device(config_device: str) -> torch.device:
    """Determine device based on config and availability."""
    if config_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(config_device)


def compute_accuracy(
    output: torch.Tensor, 
    target: torch.Tensor,
    tolerance: float = 0.5
) -> tuple[float, int, int, float]:
    """
    Compute accuracy for addition task.
    
    Args:
        output: Model output [batch, output_dim]
        target: Target [batch, output_dim]
        tolerance: Per-digit tolerance for correct prediction
    
    Returns:
        accuracy: Fraction of exactly correct predictions
        correct: Number of correct predictions
        total: Total predictions
        digit_accuracy: Fraction of correct individual digits
    """
    # Decode predictions and targets
    batch_size = output.shape[0]
    output_dim = output.shape[1]
    correct = 0
    digit_correct = 0
    total_digits = batch_size * output_dim
    
    for i in range(batch_size):
        pred = decode_number(output[i])
        true = decode_number(target[i])
        if pred == true:
            correct += 1
        
        # Also count per-digit accuracy
        pred_digits = [(output[i][j] * 9).round().long().item() for j in range(output_dim)]
        true_digits = [(target[i][j] * 9).round().long().item() for j in range(output_dim)]
        digit_correct += sum(1 for p, t in zip(pred_digits, true_digits) if p == t)
    
    digit_accuracy = digit_correct / total_digits
    return correct / batch_size, correct, batch_size, digit_accuracy


class Trainer:
    """
    CGW Trainer for Phase 0.
    
    Handles training loop, evaluation, and logging.
    """
    
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = get_device(config.device)
        self.logger = CGWLogger(log_dir=config.log_dir, debug_mode=config.debug_mode)
        
        # Create model
        self.model = CGW(
            input_dim=config.input_dim,
            workspace_dim=config.workspace_dim,
            hidden_dim=config.hidden_dim,
            tau=config.tau,
            max_steps=config.max_steps,
        ).to(self.device)
        
        # Update output head to match target dimension
        self.model.output_head = nn.Linear(config.workspace_dim, config.output_dim).to(self.device)
        
        # Optimizer (simple default)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            train_size=config.train_size,
            val_size=config.val_size,
            test_size=config.test_size,
            batch_size=config.batch_size,
        )
    
    def check_guardrails(self, start_time: float, sample_idx: int) -> bool:
        """
        Check resource guardrails.
        
        Returns:
            True if should abort
        """
        # Time check
        elapsed = time.time() - start_time
        if elapsed > self.config.max_sample_time:
            print(f"ABORT: Sample {sample_idx} exceeded {self.config.max_sample_time}s")
            return True
        
        # Memory check (approximate)
        if torch.cuda.is_available():
            memory_gb = torch.cuda.max_memory_allocated() / 1e9
            if memory_gb > self.config.max_memory_gb:
                print(f"ABORT: Memory usage {memory_gb:.1f}GB exceeds limit")
                return True
        
        return False
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets, difficulties) in enumerate(self.train_loader):
            start_time = time.time()
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs, run_info, _ = self.model(
                inputs, 
                halt_threshold=self.config.halt_threshold,
                min_steps=self.config.min_steps
            )
            
            # Compute loss
            loss = self.criterion(outputs, targets)
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"ABORT: NaN/Inf in loss at batch {batch_idx}")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Check guardrails
            if self.check_guardrails(start_time, batch_idx):
                break
        
        return total_loss / max(1, num_batches)
    
    def evaluate(
        self, 
        loader, 
        log_runs: bool = False,
        prefix: str = ""
    ) -> dict:
        """
        Evaluate model on a data loader.
        
        Returns:
            Dictionary with metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_digit_correct = 0
        total_digits = 0
        difficulty_correct = {"easy": 0, "medium": 0, "hard": 0}
        difficulty_total = {"easy": 0, "medium": 0, "hard": 0}
        
        with torch.no_grad():
            for inputs, targets, difficulties in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs, run_info, halt_logits = self.model(
                    inputs, 
                    halt_threshold=self.config.halt_threshold,
                    min_steps=self.config.min_steps
                )

                if self.config.adaptive_eval:
                    # Evaluation-only adaptive step accounting from halting logits
                    steps_used = compute_steps_used(
                        halt_logits=halt_logits,
                        min_steps=max(1, self.config.eval_min_steps),
                        halt_threshold=float(self.config.eval_halt_threshold),
                    )
                    # Use mean steps as the "trajectory length" for logging/thresholds
                    run_info.trajectory_length = int(round(steps_used.float().mean().item()))
                    if log_runs and self.config.debug_mode:
                        steps_mean = steps_used.float().mean().item()
                        steps_std = steps_used.float().std(unbiased=False).item()
                        print(f"Adaptive steps_used: mean={steps_mean:.2f}, std={steps_std:.2f}")
                else:
                    # Fixed compute: everyone uses max_steps
                    run_info.trajectory_length = int(self.config.max_steps)
                
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * inputs.shape[0]
                
                # Compute accuracy (exact match + per-digit)
                acc, correct, total, digit_acc = compute_accuracy(outputs.cpu(), targets.cpu())
                total_correct += correct
                total_samples += total
                
                # Track per-digit accuracy
                batch_size = outputs.shape[0]
                output_dim = outputs.shape[1]
                for i in range(batch_size):
                    pred_digits = [(outputs[i][j].cpu() * 9).round().long().item() for j in range(output_dim)]
                    true_digits = [(targets[i][j].cpu() * 9).round().long().item() for j in range(output_dim)]
                    total_digit_correct += sum(1 for p, t in zip(pred_digits, true_digits) if p == t)
                    total_digits += output_dim
                
                # Track per-difficulty
                for i, diff in enumerate(difficulties):
                    difficulty_total[diff] += 1
                    pred = decode_number(outputs[i].cpu())
                    true = decode_number(targets[i].cpu())
                    if pred == true:
                        difficulty_correct[diff] += 1
                        run_info.success = True
                    else:
                        run_info.success = False
                
                if log_runs:
                    self.logger.log_run(run_info)
        
        # Compute metrics
        accuracy = total_correct / max(1, total_samples)
        digit_accuracy = total_digit_correct / max(1, total_digits)
        avg_loss = total_loss / max(1, total_samples)
        
        difficulty_success = {}
        for diff in ["easy", "medium", "hard"]:
            if difficulty_total[diff] > 0:
                difficulty_success[diff] = difficulty_correct[diff] / difficulty_total[diff]
            else:
                difficulty_success[diff] = 0.0
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "digit_accuracy": digit_accuracy,
            "difficulty_success": difficulty_success,
            "total_samples": total_samples,
        }
    
    def train(self) -> dict:
        """
        Full training loop.
        
        Returns:
            Training summary
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_acc = 0.0
        train_losses = []
        val_metrics = []
        
        for epoch in range(self.config.epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            train_losses.append(train_loss)
            
            # Validate
            val_result = self.evaluate(self.val_loader, log_runs=False)
            val_metrics.append(val_result)
            
            print(f"Epoch {epoch+1}/{self.config.epochs}: "
                  f"train_loss={train_loss:.4f}, "
                  f"val_loss={val_result['loss']:.4f}, "
                  f"exact_acc={val_result['accuracy']:.1%}, "
                  f"digit_acc={val_result['digit_accuracy']:.1%}")
            
            if val_result['accuracy'] > best_val_acc:
                best_val_acc = val_result['accuracy']
                # Save best model
                torch.save(self.model.state_dict(), Path(self.config.log_dir) / "best_model.pt")
        
        # Final evaluation on test set with logging
        self.logger.reset()
        test_result = self.evaluate(self.test_loader, log_runs=True, prefix="test")
        
        # Run threshold checks
        threshold_results = self.logger.check_thresholds(test_result['difficulty_success'])
        
        # Generate report
        report = self.logger.generate_report(
            output_path=str(Path(self.config.log_dir) / "phase0_report.md")
        )
        
        # Save trajectories for debug
        self.logger.save_trajectories(self.logger.run_infos)
        
        print("\n" + "=" * 50)
        print("FINAL TEST RESULTS")
        print("=" * 50)
        print(f"Exact Match Accuracy: {test_result['accuracy']:.1%}")
        print(f"Per-Digit Accuracy: {test_result['digit_accuracy']:.1%}")
        print(f"\nBy Difficulty (Exact Match):")
        print(f"  Easy: {test_result['difficulty_success']['easy']:.1%}")
        print(f"  Medium: {test_result['difficulty_success']['medium']:.1%}")
        print(f"  Hard: {test_result['difficulty_success']['hard']:.1%}")
        print("\nThreshold Checks:")
        for check in threshold_results:
            status = "✅" if check.passed else "❌"
            print(f"  {status} {check.message}")
        
        return {
            "train_losses": train_losses,
            "val_metrics": val_metrics,
            "test_result": test_result,
            "threshold_results": [
                {"name": c.name, "value": c.value, "passed": c.passed, "message": c.message}
                for c in threshold_results
            ],
            "all_passed": all(c.passed for c in threshold_results),
        }


def train_cgw(config: Optional[TrainConfig] = None, **kwargs) -> dict:
    """
    Convenience function to train CGW.
    
    Args:
        config: Training configuration
        **kwargs: Override config values
    
    Returns:
        Training summary
    """
    if config is None:
        config = TrainConfig()
    
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    trainer = Trainer(config)
    return trainer.train()


if __name__ == "__main__":
    # Quick test
    result = train_cgw(epochs=5, debug_mode=True)
    print(f"\nAll checks passed: {result['all_passed']}")
