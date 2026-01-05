"""
Logging infrastructure for CGW Phase 0.

Provides per-step and per-run logging with automated threshold checks.
"""

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import torch

from cgw import StepInfo, RunInfo


@dataclass
class ThresholdCheck:
    """Result of a threshold check."""
    name: str
    value: float
    threshold_low: Optional[float]
    threshold_high: Optional[float]
    passed: bool
    message: str


class CGWLogger:
    """
    Logger for CGW training and evaluation.
    
    Handles:
    - Per-step logging
    - Per-run logging
    - Automated threshold checks
    - Report generation
    """
    
    # Phase 0 thresholds
    ENTROPY_LOW = 0.5
    ENTROPY_HIGH = 1.9
    FASTMEM_READ_WRITE_MIN = 0.1
    EASY_SUCCESS_MIN = 0.2
    SPECIALIST_DIVERSITY_MIN = 2
    
    def __init__(self, log_dir: str = "logs", debug_mode: bool = False):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory for log files
            debug_mode: If True, print detailed per-step info
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.debug_mode = debug_mode
        
        # Aggregate statistics
        self.run_infos: list[RunInfo] = []
        self.threshold_results: list[ThresholdCheck] = []
    
    def log_step(self, step_info: StepInfo) -> None:
        """Log a single step (debug mode only)."""
        if self.debug_mode:
            print(f"  Step {step_info.step_index}: "
                  f"selected={step_info.selected_name}, "
                  f"entropy={step_info.routing_entropy:.3f} bits, "
                  f"reads={step_info.reads}, writes={step_info.writes}, "
                  f"workspace_norm={step_info.workspace_norm:.3f}"
                  + (f", halt_logit={step_info.halt_logit:.3f}" if step_info.halt_logit is not None else ""))
    
    def log_run(self, run_info: RunInfo, sample_idx: Optional[int] = None) -> None:
        """Log a complete run."""
        self.run_infos.append(run_info)
        
        if self.debug_mode:
            prefix = f"[Sample {sample_idx}] " if sample_idx is not None else ""
            print(f"{prefix}Run completed: "
                  f"success={run_info.success}, "
                  f"steps={run_info.trajectory_length}, "
                  f"entropy={run_info.mean_entropy:.3f}±{run_info.std_entropy:.3f} bits")
            print(f"  Specialist usage: {run_info.specialist_usage}")
            print(f"  FastMem: {run_info.fastmem_stats}")
    
    def check_thresholds(self, difficulty_success: dict[str, float]) -> list[ThresholdCheck]:
        """
        Run automated threshold checks.
        
        Args:
            difficulty_success: Success rate per difficulty level
        
        Returns:
            List of threshold check results
        """
        results = []
        
        # 1. Routing entropy check
        if self.run_infos:
            mean_entropy = sum(r.mean_entropy for r in self.run_infos) / len(self.run_infos)
            passed = self.ENTROPY_LOW <= mean_entropy <= self.ENTROPY_HIGH
            results.append(ThresholdCheck(
                name="routing_entropy",
                value=mean_entropy,
                threshold_low=self.ENTROPY_LOW,
                threshold_high=self.ENTROPY_HIGH,
                passed=passed,
                message=f"Entropy {mean_entropy:.3f} bits "
                        f"({'OK' if passed else 'RED FLAG: collapsed' if mean_entropy < self.ENTROPY_LOW else 'RED FLAG: uniform'})"
            ))
        
        # 2. FastMem read/write ratio
        total_reads = sum(r.fastmem_stats.get("total_reads", 0) for r in self.run_infos)
        total_writes = sum(r.fastmem_stats.get("total_writes", 0) for r in self.run_infos)
        ratio = total_reads / max(1, total_writes)
        passed = ratio >= self.FASTMEM_READ_WRITE_MIN
        results.append(ThresholdCheck(
            name="fastmem_read_write_ratio",
            value=ratio,
            threshold_low=self.FASTMEM_READ_WRITE_MIN,
            threshold_high=None,
            passed=passed,
            message=f"Read/Write ratio {ratio:.3f} "
                    f"({'OK' if passed else 'RED FLAG: memory write-only'})"
        ))
        
        # 3. Easy task success rate
        easy_success = difficulty_success.get("easy", 0.0)
        passed = easy_success >= self.EASY_SUCCESS_MIN
        results.append(ThresholdCheck(
            name="easy_success_rate",
            value=easy_success,
            threshold_low=self.EASY_SUCCESS_MIN,
            threshold_high=None,
            passed=passed,
            message=f"Easy success rate {easy_success:.1%} "
                    f"({'OK' if passed else 'RED FLAG: core model broken'})"
        ))
        
        # 4. Specialist diversity
        if self.run_infos:
            # Aggregate usage across all runs
            total_usage = {name: 0 for name in ["io", "logic", "mem", "sup"]}
            total_steps = 0
            for run in self.run_infos:
                for name, count in run.specialist_usage.items():
                    total_usage[name] += count
                total_steps += run.trajectory_length
            
            threshold = max(1, int(0.1 * total_steps))
            active_specialists = sum(1 for count in total_usage.values() if count >= threshold)
            passed = active_specialists >= self.SPECIALIST_DIVERSITY_MIN
            results.append(ThresholdCheck(
                name="specialist_diversity",
                value=float(active_specialists),
                threshold_low=float(self.SPECIALIST_DIVERSITY_MIN),
                threshold_high=None,
                passed=passed,
                message=f"{active_specialists} specialists at ≥10% usage "
                        f"({'OK' if passed else 'RED FLAG: routing collapsed'})"
            ))
        
        # 5. Trajectory length variance
        if len(self.run_infos) > 1:
            lengths = [r.trajectory_length for r in self.run_infos]
            mean_len = sum(lengths) / len(lengths)
            std_len = (sum((l - mean_len) ** 2 for l in lengths) / len(lengths)) ** 0.5
            ratio = std_len / mean_len if mean_len > 0 else 0
            passed = ratio >= 0.5
            results.append(ThresholdCheck(
                name="trajectory_variance",
                value=ratio,
                threshold_low=0.5,
                threshold_high=None,
                passed=passed,
                message=f"Trajectory std/mean = {ratio:.3f} "
                        f"({'OK' if passed else 'WARNING: no adaptation'})"
            ))
        
        self.threshold_results = results
        return results
    
    def save_trajectories(self, run_infos: list[RunInfo], filename: str = "trajectories.json") -> None:
        """Save first N trajectories for debug inspection."""
        trajectories = []
        for i, run in enumerate(run_infos[:5]):  # First 5
            traj = {
                "run_idx": i,
                "success": run.success,
                "length": run.trajectory_length,
                "specialist_usage": run.specialist_usage,
                "steps": [
                    {
                        "step": s.step_index,
                        "specialist": s.selected_name,
                        "entropy": s.routing_entropy,
                        "reads": s.reads,
                        "writes": s.writes,
                        "workspace_norm": s.workspace_norm,
                        "halt_logit": s.halt_logit,
                    }
                    for s in run.step_infos
                ]
            }
            trajectories.append(traj)
        
        with open(self.log_dir / filename, "w") as f:
            json.dump(trajectories, f, indent=2)
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate Phase 0 report.
        
        Returns:
            Report as markdown string
        """
        report_lines = [
            "# Phase 0 Report",
            "",
            "## Summary",
            f"- Total runs: {len(self.run_infos)}",
        ]
        
        if self.run_infos:
            success_rate = sum(1 for r in self.run_infos if r.success) / len(self.run_infos)
            mean_length = sum(r.trajectory_length for r in self.run_infos) / len(self.run_infos)
            mean_entropy = sum(r.mean_entropy for r in self.run_infos) / len(self.run_infos)
            
            report_lines.extend([
                f"- Overall success rate: {success_rate:.1%}",
                f"- Mean trajectory length: {mean_length:.1f}",
                f"- Mean routing entropy: {mean_entropy:.3f} bits",
                "",
            ])
        
        # Threshold checks
        report_lines.extend([
            "## Threshold Checks",
            "",
            "| Metric | Value | Threshold | Status |",
            "|--------|-------|-----------|--------|",
        ])
        
        for check in self.threshold_results:
            status = "✅ PASS" if check.passed else "❌ FAIL"
            threshold_str = ""
            if check.threshold_low is not None and check.threshold_high is not None:
                threshold_str = f"[{check.threshold_low}, {check.threshold_high}]"
            elif check.threshold_low is not None:
                threshold_str = f"≥ {check.threshold_low}"
            elif check.threshold_high is not None:
                threshold_str = f"≤ {check.threshold_high}"
            
            report_lines.append(f"| {check.name} | {check.value:.3f} | {threshold_str} | {status} |")
        
        report_lines.extend([
            "",
            "## Detailed Messages",
            "",
        ])
        for check in self.threshold_results:
            report_lines.append(f"- **{check.name}**: {check.message}")
        
        report = "\n".join(report_lines)
        
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(report)
        
        return report
    
    def reset(self) -> None:
        """Clear accumulated statistics."""
        self.run_infos.clear()
        self.threshold_results.clear()
