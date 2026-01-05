#!/usr/bin/env python3
"""
Phase 0 Diagnostic Script

Diagnoses why the model isn't learning:
1. Task representation issues
2. Router gradient flow
3. Workspace state dynamics
4. Specialist differentiation
"""

import torch
import torch.nn.functional as F
from pathlib import Path

from cgw import CGW
from dataset import AdditionDataset, decode_number
from fastmem import FastMem
from train import TrainConfig, get_device


def diagnose_model(model_path: str = "logs/best_model.pt"):
    """Run full diagnostics on trained model."""
    
    config = TrainConfig()
    device = get_device("cpu")  # Use CPU for clearer debugging
    
    # Create model
    model = CGW(
        input_dim=config.input_dim,
        workspace_dim=config.workspace_dim,
        hidden_dim=config.hidden_dim,
        tau=config.tau,
        max_steps=config.max_steps,
    ).to(device)
    model.output_head = torch.nn.Linear(config.workspace_dim, config.output_dim).to(device)
    
    # Load weights if available
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Loaded model from {model_path}")
    else:
        print(f"⚠ No checkpoint found, using random initialization")
    
    # Create dataset
    dataset = AdditionDataset(size=100, seed=42)
    
    print("\n" + "=" * 70)
    print("TASK DESCRIPTION")
    print("=" * 70)
    
    sample = dataset.get_sample(0)
    print(f"Task: 3-digit addition")
    print(f"Example: {sample.a} + {sample.b} = {sample.result}")
    print(f"Input tensor shape: {sample.input_tensor.shape}")
    print(f"Input tensor (normalized digits): {sample.input_tensor.tolist()}")
    print(f"Target tensor shape: {sample.target_tensor.shape}")
    print(f"Target tensor (normalized digits): {sample.target_tensor.tolist()}")
    print(f"Difficulty: {sample.difficulty}")
    print(f"\nDataset distribution: {dataset.get_difficulty_distribution()}")
    
    print("\n" + "=" * 70)
    print("SINGLE EXAMPLE TRAJECTORY")
    print("=" * 70)
    
    model.eval()
    input_tensor = sample.input_tensor.unsqueeze(0).to(device)
    target_tensor = sample.target_tensor.unsqueeze(0).to(device)
    
    # Manual step-by-step execution
    S_t = model.S_init.unsqueeze(0).clone()
    mem = FastMem()
    
    print(f"\nInput: {sample.a} + {sample.b} = {sample.result}")
    print(f"Target: {sample.target_tensor.tolist()}")
    print(f"\nInitial S_0 norm: {S_t.norm().item():.4f}")
    print(f"Initial S_0 mean: {S_t.mean().item():.4f}")
    print(f"Initial S_0 std: {S_t.std().item():.4f}")
    print()
    
    trajectory = []
    for t in range(model.max_steps):
        # Get routing decision
        routing_probs, selected, routing_logits = model.router(S_t)
        entropy = model.router.get_entropy(routing_logits).item()
        
        # Execute specialist
        delta_S, halt_logit = model.specialists(selected, S_t, input_tensor, mem)
        S_next = S_t + delta_S
        
        # Log
        step_info = {
            "step": t,
            "selected": selected,
            "selected_name": model.specialists.get_name(selected),
            "routing_probs": routing_probs[0].detach().tolist(),
            "entropy": entropy,
            "delta_S_norm": delta_S.norm().item(),
            "S_norm": S_next.norm().item(),
            "S_mean": S_next.mean().item(),
            "reads": mem.read_count_delta(),
            "writes": mem.write_count_delta(),
            "halt_logit": halt_logit.item() if halt_logit is not None else None,
        }
        trajectory.append(step_info)
        
        print(f"Step {t}: specialist={step_info['selected_name']:5s} "
              f"probs=[{', '.join(f'{p:.2f}' for p in step_info['routing_probs'])}] "
              f"entropy={step_info['entropy']:.3f} bits")
        print(f"         delta_S_norm={step_info['delta_S_norm']:.4f} "
              f"S_norm={step_info['S_norm']:.4f} "
              f"reads={step_info['reads']} writes={step_info['writes']}")
        
        S_t = S_next
        
        # Check halt
        if halt_logit is not None and torch.sigmoid(halt_logit).item() > 0.5:
            print(f"         HALT (logit={step_info['halt_logit']:.3f})")
            break
    
    # Get output
    output = model.output_head(S_t)
    predicted = decode_number(output[0])
    expected = sample.result
    
    print(f"\nFinal output tensor: {output[0].detach().tolist()}")
    print(f"Predicted (decoded): {predicted}")
    print(f"Expected: {expected}")
    print(f"Correct: {predicted == expected}")
    
    print("\n" + "=" * 70)
    print("ROUTER WEIGHT ANALYSIS")
    print("=" * 70)
    
    W = model.router.W_route.weight.detach()
    b = model.router.W_route.bias.detach()
    
    print(f"W_route weight shape: {W.shape}")
    print(f"W_route weight norm: {W.norm().item():.4f}")
    print(f"W_route weight mean: {W.mean().item():.6f}")
    print(f"W_route weight std: {W.std().item():.6f}")
    print(f"W_route bias: {b.tolist()}")
    
    # Check if weights vary across specialists
    for i in range(4):
        print(f"  Specialist {i} ({model.specialists.get_name(i)}): "
              f"weight_norm={W[i].norm().item():.4f}, bias={b[i].item():.4f}")
    
    print("\n" + "=" * 70)
    print("SPECIALIST DIFFERENTIATION TEST")
    print("=" * 70)
    
    # Test if specialists produce different outputs
    S_test = torch.randn(1, config.workspace_dim)
    x_test = torch.randn(1, config.input_dim)
    mem_test = FastMem()
    
    outputs = []
    for specialist_id in range(4):
        if specialist_id == 2:  # mem specialist needs fresh memory
            mem_test = FastMem()
        out, _ = model.specialists(specialist_id, S_test, x_test, mem_test)
        outputs.append(out[0])
        print(f"Specialist {specialist_id} ({model.specialists.get_name(specialist_id)}): "
              f"output_norm={out.norm().item():.4f}, mean={out.mean().item():.4f}")
    
    print("\nCosine similarities between specialist outputs:")
    for i in range(4):
        for j in range(i + 1, 4):
            sim = F.cosine_similarity(outputs[i].unsqueeze(0), outputs[j].unsqueeze(0)).item()
            print(f"  {model.specialists.get_name(i)} vs {model.specialists.get_name(j)}: {sim:.4f}")
    
    print("\n" + "=" * 70)
    print("WORKSPACE DYNAMICS TEST")
    print("=" * 70)
    
    # Check if workspace changes across different inputs
    S_samples = []
    for i in range(5):
        inp, _, _ = dataset[i]
        inp = inp.unsqueeze(0).to(device)
        _, run_info, _ = model(inp)
        S_samples.append(run_info.final_output)
    
    S_all = torch.stack(S_samples)
    print(f"Workspace variance across 5 samples: {S_all.var().item():.6f}")
    print(f"Workspace mean across 5 samples: {S_all.mean().item():.6f}")
    
    # Check variance within a single trajectory
    print("\nWorkspace evolution within trajectory:")
    S_t = model.S_init.unsqueeze(0).clone()
    mem = FastMem()
    S_history = [S_t.clone()]
    
    for t in range(model.max_steps):
        _, selected, _ = model.router(S_t)
        delta_S, _ = model.specialists(selected, S_t, input_tensor, mem)
        S_t = S_t + delta_S
        S_history.append(S_t.clone())
    
    S_history = torch.stack([s.squeeze(0) for s in S_history])
    print(f"  Trajectory variance: {S_history.var().item():.6f}")
    print(f"  Step-to-step changes:")
    for t in range(len(S_history) - 1):
        diff = (S_history[t + 1] - S_history[t]).norm().item()
        print(f"    Step {t} → {t+1}: delta_norm={diff:.4f}")
    
    print("\n" + "=" * 70)
    print("LOSS FUNCTION ANALYSIS")  
    print("=" * 70)
    
    # Check what loss we're computing
    criterion = torch.nn.MSELoss()
    output, _, _ = model(input_tensor)
    loss = criterion(output, target_tensor)
    
    print(f"Loss type: MSELoss")
    print(f"Single sample loss: {loss.item():.6f}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"Target range: [{target_tensor.min().item():.4f}, {target_tensor.max().item():.4f}]")
    
    # Check if output is in reasonable range
    print(f"\nOutput (denormalized to digits): {[round(v * 9) for v in output[0].tolist()]}")
    print(f"Target (denormalized to digits): {[round(v * 9) for v in target_tensor[0].tolist()]}")
    
    print("\n" + "=" * 70)
    print("RANDOM INITIALIZATION TEST")
    print("=" * 70)
    
    # Check if any random init can solve problems
    print("Testing 5 random seeds on 10 easy problems...")
    easy_samples = [s for s in dataset.samples if s.difficulty == "easy"][:10]
    
    for seed in range(5):
        torch.manual_seed(seed)
        fresh_model = CGW(
            input_dim=config.input_dim,
            workspace_dim=config.workspace_dim,
            hidden_dim=config.hidden_dim,
            tau=config.tau,
            max_steps=config.max_steps,
        ).to(device)
        fresh_model.output_head = torch.nn.Linear(config.workspace_dim, config.output_dim).to(device)
        fresh_model.eval()
        
        correct = 0
        for sample in easy_samples:
            inp = sample.input_tensor.unsqueeze(0).to(device)
            out, _ = fresh_model(inp)
            pred = decode_number(out[0])
            if pred == sample.result:
                correct += 1
        
        print(f"  Seed {seed}: {correct}/10 correct ({correct * 10}%)")
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    
    # Summarize findings
    issues = []
    
    # Check routing uniformity
    avg_entropy = sum(t["entropy"] for t in trajectory) / len(trajectory)
    if avg_entropy > 1.8:
        issues.append(f"ROUTING UNIFORM: entropy={avg_entropy:.2f} bits (near max 2.0)")
    
    # Check workspace dynamics
    if S_all.var().item() < 0.01:
        issues.append(f"WORKSPACE COLLAPSED: variance={S_all.var().item():.6f}")
    
    # Check specialist differentiation
    all_similar = True
    for i in range(4):
        for j in range(i + 1, 4):
            sim = F.cosine_similarity(outputs[i].unsqueeze(0), outputs[j].unsqueeze(0)).item()
            if sim < 0.9:
                all_similar = False
    if all_similar:
        issues.append("SPECIALISTS NOT DIFFERENTIATED: all outputs similar (cosine > 0.9)")
    
    # Check delta_S magnitude
    avg_delta = sum(t["delta_S_norm"] for t in trajectory) / len(trajectory)
    if avg_delta < 0.01:
        issues.append(f"DELTA_S TOO SMALL: avg={avg_delta:.6f}")
    
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("No obvious issues found - check gradient flow manually")
    
    return trajectory, issues


if __name__ == "__main__":
    diagnose_model()
