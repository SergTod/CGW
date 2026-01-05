#!/usr/bin/env python3
"""
CGW Phase 0 - Main CLI Entry Point

Usage:
    python main.py train [--debug-mode] [--visualize] [--device cpu|cuda|mps]
    python main.py test [--debug-mode]
    python main.py sanity [--device cpu|cuda|mps]
"""

import argparse
import sys
import torch
import torch.nn as nn
from pathlib import Path

from cgw import CGW
from train import TrainConfig, Trainer, train_cgw, get_device
from dataset import AdditionDataset, create_dataloaders
from sanity_tests import run_all_sanity_tests
from logger import CGWLogger
from visualize import generate_all_visualizations, HAS_MATPLOTLIB


def parse_args():
    parser = argparse.ArgumentParser(
        description="CGW Phase 0 - Cognitive Global Workspace"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the CGW model")
    train_parser.add_argument(
        "--debug-mode", 
        action="store_true",
        help="Print routing decisions per step, dump FastMem keys, save trajectories"
    )
    train_parser.add_argument(
        "--visualize",
        action="store_true", 
        help="Generate routing usage plot, entropy over time, FastMem heatmap"
    )
    train_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    train_parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum reasoning steps per sample"
    )
    train_parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for logs and outputs"
    )
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Evaluate a trained model")
    test_parser.add_argument(
        "--model-path",
        type=str,
        default="logs/best_model.pt",
        help="Path to trained model"
    )
    test_parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Print detailed evaluation info"
    )
    test_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use"
    )

    # Adaptive evaluation (Phase 0.5)
    test_parser.add_argument(
        "--adaptive-eval",
        action="store_true",
        help="Compute per-sample steps_used from halt logits during evaluation"
    )
    test_parser.add_argument(
        "--min-steps",
        type=int,
        default=2,
        help="Minimum steps before halting is allowed (evaluation only)"
    )
    test_parser.add_argument(
        "--halt-threshold",
        type=float,
        default=0.7,
        help="Halting probability threshold (evaluation only)"
    )
    
    # Sanity test command
    sanity_parser = subparsers.add_parser("sanity", help="Run sanity tests")
    sanity_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use"
    )
    sanity_parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model (optional, uses random init if not provided)"
    )
    
    return parser.parse_args()


def cmd_train(args):
    """Run training."""
    print("=" * 60)
    print("CGW Phase 0 - Training")
    print("=" * 60)
    
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        log_dir=args.log_dir,
        debug_mode=args.debug_mode,
        device=args.device,
    )
    
    print(f"Configuration:")
    print(f"  Device: {args.device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Debug mode: {args.debug_mode}")
    print(f"  Visualize: {args.visualize}")
    print()
    
    trainer = Trainer(config)
    result = trainer.train()
    
    # Generate visualizations if requested
    if args.visualize:
        if HAS_MATPLOTLIB:
            generate_all_visualizations(
                trainer.logger.run_infos,
                output_dir=str(Path(args.log_dir) / "viz"),
                show=False
            )
        else:
            print("\nWARNING: matplotlib not installed, skipping visualizations")
            print("Install with: pip install matplotlib")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Report saved to: {Path(args.log_dir) / 'phase0_report.md'}")
    print(f"Trajectories saved to: {Path(args.log_dir) / 'trajectories.json'}")
    if args.visualize and HAS_MATPLOTLIB:
        print(f"Visualizations saved to: {Path(args.log_dir) / 'viz/'}")
    
    print(f"\nAll threshold checks passed: {result['all_passed']}")
    
    if result['all_passed']:
        print("\n✅ Phase 0 PASSED - Ready for human review before Phase 1")
    else:
        print("\n❌ Phase 0 FAILED - Review threshold check failures")
    
    return 0 if result['all_passed'] else 1


def cmd_test(args):
    """Run evaluation on trained model."""
    print("=" * 60)
    print("CGW Phase 0 - Evaluation")
    print("=" * 60)
    
    device = get_device(args.device)
    config = TrainConfig(device=args.device, debug_mode=args.debug_mode)
    # Evaluation-only adaptive step accounting flags
    config.adaptive_eval = bool(args.adaptive_eval)
    config.eval_min_steps = int(args.min_steps)
    config.eval_halt_threshold = float(args.halt_threshold)
    
    # Load model
    model = CGW(
        input_dim=config.input_dim,
        workspace_dim=config.workspace_dim,
        hidden_dim=config.hidden_dim,
        tau=config.tau,
        max_steps=config.max_steps,
    ).to(device)
    model.output_head = nn.Linear(config.workspace_dim, config.output_dim).to(device)
    
    if Path(args.model_path).exists():
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"WARNING: Model not found at {args.model_path}, using random initialization")
    
    # Create test loader
    _, _, test_loader = create_dataloaders(
        test_size=config.test_size,
        batch_size=config.batch_size,
    )
    
    # Evaluate
    trainer = Trainer(config)
    trainer.model = model
    result = trainer.evaluate(test_loader, log_runs=True)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {result['accuracy']:.1%}")
    print(f"  Loss: {result['loss']:.4f}")
    print(f"  Easy: {result['difficulty_success']['easy']:.1%}")
    print(f"  Medium: {result['difficulty_success']['medium']:.1%}")
    print(f"  Hard: {result['difficulty_success']['hard']:.1%}")
    
    return 0


def cmd_sanity(args):
    """Run sanity tests."""
    print("=" * 60)
    print("CGW Phase 0 - Sanity Tests")
    print("=" * 60)
    
    device = get_device(args.device)
    config = TrainConfig()
    
    # Create model
    model = CGW(
        input_dim=config.input_dim,
        workspace_dim=config.workspace_dim,
        hidden_dim=config.hidden_dim,
        tau=config.tau,
        max_steps=config.max_steps,
    ).to(device)
    model.output_head = nn.Linear(config.workspace_dim, config.output_dim).to(device)
    
    # Load weights if provided
    if args.model_path and Path(args.model_path).exists():
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    else:
        print("Using randomly initialized model")
    
    # Create dataset
    dataset = AdditionDataset(size=100, seed=42)
    
    # Run tests
    results = run_all_sanity_tests(model, dataset, device)
    
    # Summary
    print("\n" + "=" * 60)
    print("SANITY TEST SUMMARY")
    print("=" * 60)
    
    all_passed = all(r.passed for r in results)
    for r in results:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        print(f"  {status}: {r.name}")
    
    print(f"\nAll tests passed: {all_passed}")
    
    return 0 if all_passed else 1


def main():
    args = parse_args()
    
    if args.command is None:
        print("No command specified. Use --help for usage.")
        print("\nAvailable commands:")
        print("  train  - Train the CGW model")
        print("  test   - Evaluate a trained model")
        print("  sanity - Run sanity tests")
        return 1
    
    if args.command == "train":
        return cmd_train(args)
    elif args.command == "test":
        return cmd_test(args)
    elif args.command == "sanity":
        return cmd_sanity(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
