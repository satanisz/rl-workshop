import torch
import numpy as np
from dataset import KnapsackGenerator, KnapsackDataset, collate_fn
from solvers import DynamicProgrammingSolver, GreedySolver
from model import KnapsackTransformer
from train import train_sl, train_rl
from torch.utils.data import DataLoader
import time

def evaluate_models(n_test_items=20, n_samples=50):
    print(f"\n{'='*20} Starting Evaluation (N={n_test_items}) {'='*20}")
    
    # 1. Train Models (Small training for demo purposes)
    print("Training Supervised Model (SL)...")
    model_sl = KnapsackTransformer(nhead=4, num_layers=2)
    # Train heavily reduces for speed in demo; increase epochs for real results
    train_sl(model_sl, n_epochs=5, batch_size=32, n_train_items=n_test_items)
    model_sl.eval()

    print("\nTraining Reinforcement Learning Model (RL)...")
    model_rl = KnapsackTransformer(nhead=4, num_layers=2)
    train_rl(model_rl, n_epochs=5, batch_size=32, n_train_items=n_test_items)
    model_rl.eval()
    
    # 2. Generate Test Data
    print("\nGenerating Test Data...")
    generator = KnapsackGenerator(min_items=n_test_items, max_items=n_test_items)
    test_instances = generator.generate(n_instances=n_samples)
    
    # 3. Solve with Classical Methods
    dp_solver = DynamicProgrammingSolver()
    greedy_solver = GreedySolver()
    
    results = {
        'DP': {'value': [], 'time': [], 'valid': []},
        'Greedy': {'value': [], 'time': [], 'valid': []},
        'SL': {'value': [], 'time': [], 'valid': []},
        'RL': {'value': [], 'time': [], 'valid': []}
    }
    
    print("Evaluating...")
    
    for i, inst in enumerate(test_instances):
        # --- DP ---
        start = time.time()
        dp_val, _ = dp_solver.solve(inst)
        end = time.time()
        results['DP']['value'].append(dp_val)
        results['DP']['time'].append(end - start)
        results['DP']['valid'].append(1) # Always valid
        
        # --- Greedy ---
        start = time.time()
        g_val, _ = greedy_solver.solve(inst)
        end = time.time()
        results['Greedy']['value'].append(g_val)
        results['Greedy']['time'].append(end - start)
        results['Greedy']['valid'].append(1) # Always valid
        
        # --- Neural Inference Prep ---
        # Prepare single batch
        dataset = KnapsackDataset([inst])
        batch = collate_fn([dataset[0]])
        features = batch['features']
        mask = batch['mask']
        
        # --- SL Inference ---
        start = time.time()
        with torch.no_grad():
            probs = model_sl(features, mask=mask, temperature=0.1) # Low temp for greedy decoding
            # Greedy decoding: > 0.5
            actions = (probs > 0.5).int().squeeze(0).numpy() # (N,)
            
        # Calc constraints
        w_total = (actions * inst.weights).sum()
        v_total = (actions * inst.values).sum()
        
        valid = 1 if w_total <= inst.capacity else 0
        val = v_total if valid else 0 
        
        results['SL']['value'].append(val)
        results['SL']['time'].append(time.time() - start)
        results['SL']['valid'].append(valid)
        
        # --- RL Inference ---
        start = time.time()
        with torch.no_grad():
            probs = model_rl(features, mask=mask, temperature=0.1)
            actions = (probs > 0.5).int().squeeze(0).numpy()
            
        w_total = (actions * inst.weights).sum()
        v_total = (actions * inst.values).sum()
        
        valid = 1 if w_total <= inst.capacity else 0
        val = v_total if valid else 0 
        
        results['RL']['value'].append(val)
        results['RL']['time'].append(time.time() - start)
        results['RL']['valid'].append(valid)

    # 4. Report
    print(f"\nResults (Average over {n_samples} samples):")
    print(f"{'Method':<10} | {'Avg Value':<12} | {'Optimal %':<10} | {'Time (ms)':<10} | {'Valid %':<10}")
    print("-" * 65)
    
    avg_dp_val = np.mean(results['DP']['value'])
    
    for name in ['DP', 'Greedy', 'SL', 'RL']:
        avg_val = np.mean(results[name]['value'])
        avg_time = np.mean(results[name]['time']) * 1000
        valid_pct = np.mean(results[name]['valid']) * 100
        
        # Optimal % is strictly Value / DP_Value
        opt_pct = (avg_val / avg_dp_val) * 100
        
        print(f"{name:<10} | {avg_val:<12.2f} | {opt_pct:<10.2f} | {avg_time:<10.4f} | {valid_pct:<10.1f}")
        
    return results

if __name__ == "__main__":
    evaluate_models(n_test_items=20, n_samples=50)
    # Optional: Test generalization
    # evaluate_models(n_test_items=50, n_samples=50)
