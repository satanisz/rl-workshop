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
    train_sl(model_sl, n_epochs=200, batch_size=32, n_train_items=n_test_items)
    model_sl.eval()

    print("\nTraining Reinforcement Learning Model (RL)...")
    model_rl = KnapsackTransformer(nhead=4, num_layers=2)
    train_rl(model_rl, n_epochs=200, batch_size=32, n_train_items=n_test_items)
    model_rl.eval()
    
    # 2. Generate Test Data
    print("\nGenerating Test Data...")
    generator = KnapsackGenerator(min_items=n_test_items, max_items=n_test_items)
    test_instances = generator.generate(n_instances=n_samples)
    
    # 3. Solve with Classical Methods
    dp_solver = DynamicProgrammingSolver()
    greedy_solver = GreedySolver()
    
    # NEW: Neural Greedy Solvers
    from solvers import NeuralGreedySolver
    sl_constructive_solver = NeuralGreedySolver(model_sl)
    rl_constructive_solver = NeuralGreedySolver(model_rl)
    
    results = {
        'DP': {'value': [], 'time': [], 'valid': []},
        'Greedy': {'value': [], 'time': [], 'valid': []},
        'SL_Raw': {'value': [], 'time': [], 'valid': []},
        'SL_Constr': {'value': [], 'time': [], 'valid': []},
        'RL_Raw': {'value': [], 'time': [], 'valid': []},
        'RL_Constr': {'value': [], 'time': [], 'valid': []}
    }
    
    print("Evaluating...")
    
    for i, inst in enumerate(test_instances):
        # --- DP ---
        start = time.time()
        dp_val, _ = dp_solver.solve(inst)
        results['DP']['value'].append(dp_val)
        results['DP']['time'].append(time.time() - start)
        results['DP']['valid'].append(1) 

        # --- Greedy ---
        start = time.time()
        g_val, _ = greedy_solver.solve(inst)
        results['Greedy']['value'].append(g_val)
        results['Greedy']['time'].append(time.time() - start)
        results['Greedy']['valid'].append(1)
        
        # --- SL Constructive (Neural Greedy) ---
        start = time.time()
        sl_c_val, _ = sl_constructive_solver.solve(inst)
        results['SL_Constr']['value'].append(sl_c_val)
        results['SL_Constr']['time'].append(time.time() - start)
        results['SL_Constr']['valid'].append(1) # Always valid by design

        # --- RL Constructive (Neural Greedy) ---
        start = time.time()
        rl_c_val, _ = rl_constructive_solver.solve(inst)
        results['RL_Constr']['value'].append(rl_c_val)
        results['RL_Constr']['time'].append(time.time() - start)
        results['RL_Constr']['valid'].append(1) # Always valid by design
        
        # --- Raw Neural Inference (Old "In/Out" classification) ---
        dataset = KnapsackDataset([inst])
        batch = collate_fn([dataset[0]])
        features = batch['features']
        mask = batch['mask']
        
        # SL Raw
        with torch.no_grad():
            probs = model_sl(features, mask=mask, temperature=0.1)
            actions = (probs > 0.5).int().squeeze(0).numpy()
        w = (actions * inst.weights).sum()
        v = (actions * inst.values).sum() if w <= inst.capacity else 0
        results['SL_Raw']['value'].append(v)
        results['SL_Raw']['time'].append(0) # Negligible/Included in constr measure mostly
        results['SL_Raw']['valid'].append(1 if w <= inst.capacity else 0)

        # RL Raw
        with torch.no_grad():
            probs = model_rl(features, mask=mask, temperature=0.1)
            actions = (probs > 0.5).int().squeeze(0).numpy()
        w = (actions * inst.weights).sum()
        v = (actions * inst.values).sum() if w <= inst.capacity else 0
        results['RL_Raw']['value'].append(v)
        results['RL_Raw']['time'].append(0)
        results['RL_Raw']['valid'].append(1 if w <= inst.capacity else 0)

    # 4. Report
    print(f"\nResults (Average over {n_samples} samples):")
    print(f"{'Method':<12} | {'Avg Value':<12} | {'Optimal %':<10} | {'Time (ms)':<10} | {'Valid %':<10}")
    print("-" * 65)
    
    avg_dp_val = np.mean(results['DP']['value'])
    
    for name in ['DP', 'Greedy', 'SL_Raw', 'SL_Constr', 'RL_Raw', 'RL_Constr']:
        avg_val = np.mean(results[name]['value'])
        avg_time = np.mean(results[name]['time']) * 1000
        valid_pct = np.mean(results[name]['valid']) * 100
        opt_pct = (avg_val / avg_dp_val) * 100 if avg_dp_val > 0 else 0
        
        print(f"{name:<12} | {avg_val:<12.2f} | {opt_pct:<10.2f} | {avg_time:<10.4f} | {valid_pct:<10.1f}")
        
    return results
        
    return results

if __name__ == "__main__":
    evaluate_models(n_test_items=20, n_samples=500)
    # Optional: Test generalization
    # evaluate_models(n_test_items=50, n_samples=50)
