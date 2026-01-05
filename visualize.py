"""
Visualization utilities for CGW Phase 0.

Generates:
- Routing usage plot
- Routing entropy over time
- FastMem read/write heatmap
"""

import json
from pathlib import Path
from typing import Optional
import torch

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from cgw import RunInfo, StepInfo


def plot_routing_usage(
    run_infos: list[RunInfo],
    output_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot specialist usage distribution.
    
    Args:
        run_infos: List of run information
        output_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed, skipping visualization")
        return
    
    # Aggregate usage
    total_usage = {"io": 0, "logic": 0, "mem": 0, "sup": 0}
    for run in run_infos:
        for name, count in run.specialist_usage.items():
            total_usage[name] += count
    
    total = sum(total_usage.values())
    usage_pct = {name: count / max(1, total) * 100 for name, count in total_usage.items()}
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    names = list(usage_pct.keys())
    values = list(usage_pct.values())
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add percentage labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11)
    
    ax.set_ylabel('Usage (%)', fontsize=12)
    ax.set_xlabel('Specialist', fontsize=12)
    ax.set_title('Specialist Routing Usage Distribution', fontsize=14)
    ax.set_ylim(0, max(values) * 1.2)
    
    # Add threshold line at 10%
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='10% threshold')
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_entropy_over_time(
    run_infos: list[RunInfo],
    output_path: Optional[str] = None,
    show: bool = True,
    max_runs: int = 10,
) -> None:
    """
    Plot routing entropy over time for multiple runs.
    
    Args:
        run_infos: List of run information
        output_path: Path to save figure (optional)
        show: Whether to display the plot
        max_runs: Maximum number of runs to plot
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed, skipping visualization")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, run in enumerate(run_infos[:max_runs]):
        steps = [s.step_index for s in run.step_infos]
        entropies = [s.routing_entropy for s in run.step_infos]
        ax.plot(steps, entropies, alpha=0.6, linewidth=1.5, label=f'Run {i+1}')
    
    # Add threshold lines
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Low threshold (0.5)')
    ax.axhline(y=1.9, color='orange', linestyle='--', alpha=0.7, label='High threshold (1.9)')
    ax.axhline(y=2.0, color='gray', linestyle=':', alpha=0.5, label='Max entropy (2.0)')
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Routing Entropy (bits)', fontsize=12)
    ax.set_title('Routing Entropy Over Time', fontsize=14)
    ax.set_ylim(0, 2.2)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_fastmem_heatmap(
    run_infos: list[RunInfo],
    output_path: Optional[str] = None,
    show: bool = True,
    max_runs: int = 20,
) -> None:
    """
    Plot FastMem read/write heatmap across runs and steps.
    
    Args:
        run_infos: List of run information
        output_path: Path to save figure (optional)
        show: Whether to display the plot
        max_runs: Maximum number of runs to show
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed, skipping visualization")
        return
    
    runs_to_plot = run_infos[:max_runs]
    max_steps = max(run.trajectory_length for run in runs_to_plot)
    
    # Create heatmap data: 0=nothing, 1=read, 2=write, 3=both
    heatmap = [[0] * max_steps for _ in range(len(runs_to_plot))]
    
    for i, run in enumerate(runs_to_plot):
        for step_info in run.step_infos:
            step = step_info.step_index
            if step < max_steps:
                val = 0
                if step_info.reads > 0:
                    val += 1
                if step_info.writes > 0:
                    val += 2
                heatmap[i][step] = val
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Custom colormap: white, blue (read), red (write), purple (both)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['white', '#3498db', '#e74c3c', '#9b59b6'])
    
    im = ax.imshow(heatmap, aspect='auto', cmap=cmap, vmin=0, vmax=3)
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Run', fontsize=12)
    ax.set_title('FastMem Operations Heatmap', fontsize=14)
    
    # Legend
    patches = [
        mpatches.Patch(color='white', label='No op', edgecolor='black'),
        mpatches.Patch(color='#3498db', label='Read'),
        mpatches.Patch(color='#e74c3c', label='Write'),
        mpatches.Patch(color='#9b59b6', label='Read+Write'),
    ]
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_trajectory(
    run_info: RunInfo,
    output_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot detailed trajectory for a single run.
    
    Shows specialist selection, entropy, and workspace norm over time.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed, skipping visualization")
        return
    
    steps = [s.step_index for s in run_info.step_infos]
    specialists = [s.selected for s in run_info.step_infos]
    entropies = [s.routing_entropy for s in run_info.step_infos]
    norms = [s.workspace_norm for s in run_info.step_infos]
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Specialist selection
    ax1 = axes[0]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    specialist_colors = [colors[s] for s in specialists]
    ax1.scatter(steps, specialists, c=specialist_colors, s=100, edgecolors='black')
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(['io', 'logic', 'mem', 'sup'])
    ax1.set_ylabel('Specialist')
    ax1.set_title('Trajectory Analysis')
    
    # Entropy
    ax2 = axes[1]
    ax2.plot(steps, entropies, 'b-o', linewidth=2, markersize=6)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    ax2.axhline(y=1.9, color='orange', linestyle='--', alpha=0.7)
    ax2.set_ylabel('Entropy (bits)')
    ax2.set_ylim(0, 2.2)
    
    # Workspace norm
    ax3 = axes[2]
    ax3.plot(steps, norms, 'g-o', linewidth=2, markersize=6)
    ax3.set_ylabel('Workspace Norm')
    ax3.set_xlabel('Step')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def generate_all_visualizations(
    run_infos: list[RunInfo],
    output_dir: str = "logs/viz",
    show: bool = False,
) -> None:
    """
    Generate all visualization plots.
    
    Args:
        run_infos: List of run information
        output_dir: Directory to save plots
        show: Whether to display plots
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed, skipping all visualizations")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Generating visualizations...")
    
    # 1. Routing usage
    plot_routing_usage(
        run_infos, 
        output_path=str(output_path / "routing_usage.png"),
        show=show
    )
    print(f"  Saved: {output_path / 'routing_usage.png'}")
    
    # 2. Entropy over time
    plot_entropy_over_time(
        run_infos,
        output_path=str(output_path / "entropy_over_time.png"),
        show=show
    )
    print(f"  Saved: {output_path / 'entropy_over_time.png'}")
    
    # 3. FastMem heatmap
    plot_fastmem_heatmap(
        run_infos,
        output_path=str(output_path / "fastmem_heatmap.png"),
        show=show
    )
    print(f"  Saved: {output_path / 'fastmem_heatmap.png'}")
    
    # 4. Sample trajectory (first successful run)
    for i, run in enumerate(run_infos):
        if run.success:
            plot_trajectory(
                run,
                output_path=str(output_path / f"trajectory_run{i}.png"),
                show=show
            )
            print(f"  Saved: {output_path / f'trajectory_run{i}.png'}")
            break
    
    print("Visualization complete!")
