import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from model import KnapsackTransformer
from dataset import KnapsackDataset, KnapsackGenerator, collate_fn
from solvers import DynamicProgrammingSolver

def solve_batch_dp(instances):
    solver = DynamicProgrammingSolver()
    solutions = []
    for inst in instances:
        _, sol = solver.solve(inst)
        solutions.append(sol)
    return solutions

def train_sl(model, n_epochs=10, batch_size=32, lr=1e-3, n_train_items=20):
    """
    Supervised Learning: Train model to mimic DP solver.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    generator = KnapsackGenerator(min_items=n_train_items, max_items=n_train_items)
    # We generate data on the fly or pre-generate. 
    # For stability, let's pre-generate a fixed dataset for 'epoch' concept.
    
    print("Generating SL training data...")
    train_instances = generator.generate(n_instances=1000) # 1000 samples
    
    # Solve with DP to get labels
    print("Solving with DP...")
    solver = DynamicProgrammingSolver()
    for inst in train_instances:
        _, sol = solver.solve(inst)
        inst.optimal_solution = sol
        
    dataset = KnapsackDataset(train_instances)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for batch in dataloader:
            features = batch['features']
            labels = batch['labels']
            mask = batch['mask']
            
            optimizer.zero_grad()
            probs = model(features, mask=mask)
            
            # Apply mask to loss calculation (only valid items)
            # Flatten
            probs_flat = probs[mask]
            labels_flat = labels[mask]
            
            loss = criterion(probs_flat, labels_flat)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"SL Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    return model

def train_rl(model, n_epochs=10, batch_size=32, lr=1e-4, n_train_items=20):
    """
    Reinforcement Learning (REINFORCE/PG):
    Reward = Total Value if Weight <= Capacity else 0 (or penalty).
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    generator = KnapsackGenerator(min_items=n_train_items, max_items=n_train_items)
    
    # RL usually needs more data, we can generate on fly
    batches_per_epoch = 50
    
    model.train()
    for epoch in range(n_epochs):
        total_reward = 0
        total_loss = 0
        
        for _ in range(batches_per_epoch):
            instances = generator.generate(n_instances=batch_size)
            dataset = KnapsackDataset(instances)
            batch = collate_fn([dataset[i] for i in range(len(dataset))])
            
            features = batch['features']
            mask = batch['mask']
            weights = batch['weights']
            values = batch['values']
            capacities = batch['capacities']
            
            optimizer.zero_grad()
            
            # 1. Forward pass to get probabilities
            probs = model(features, mask=mask) # (B, N)
            
            # 2. Sample actions (Bernoulli)
            m = torch.distributions.Bernoulli(probs)
            actions = m.sample() # (B, N) -> 0 or 1
            
            # 3. Calculate Reward
            # actions is (B, N) float
            # weights is (B, N)
            # values is (B, N)
            
            selected_weights = (actions * weights).sum(dim=1) # (B,)
            selected_values = (actions * values).sum(dim=1)   # (B,)
            
            rewards = []
            for i in range(batch_size):
                if selected_weights[i] <= capacities[i]:
                    # Valid solution
                    r = selected_values[i].item()
                else:
                    # Invalid solution - Constraint violation
                    # Penalty: 0 or negative. 
                    # If we give 0, it might learn to pick nothing.
                    # Let's give a small penalty or just 0, but often 0 is fine if picking nothing is also length 0.
                    # Better: Penalty proportional to violation or just 0.
                    r = 0 # Strict penalty
                rewards.append(r)
            
            rewards = torch.FloatTensor(rewards)
            
            # Baseline (simple moving average or mean of batch)
            baseline = rewards.mean()
            
            # 4. Loss = - (Reward - Baseline) * log_prob(action)
            # We want to maximize Reward, so minimize Negative Reward
            
            log_probs = m.log_prob(actions)
            
            # Mask out padding in log_probs sum
            # log_probs is (B, N). We only care about valid items.
            # But sum(log_probs) over items gives log_prob of the *trajectory* (solution)
            
            # We need to handle the mask.
            # For padded items, probs might be anything, but we shouldn't update based on them.
            # Let's zero out log_probs for padded items
            log_probs = log_probs * mask.float()
            
            # Sum log_probs over items to get log_prob of the configuration
            log_prob_trajectory = log_probs.sum(dim=1) # (B,)
            
            loss = - ((rewards - baseline) * log_prob_trajectory).mean()
            
            loss.backward()
            optimizer.step()
            
            total_reward += rewards.mean().item()
            total_loss += loss.item()
            
        print(f"RL Epoch {epoch+1}/{n_epochs}, Avg Reward: {total_reward/batches_per_epoch:.2f}")
        
    return model

if __name__ == "__main__":
    # Quick test
    print("Testing SL Training...")
    model_sl = KnapsackTransformer()
    train_sl(model_sl, n_epochs=2, batch_size=4, n_train_items=10)
    
    print("\nTesting RL Training...")
    model_rl = KnapsackTransformer()
    train_rl(model_rl, n_epochs=2, batch_size=4, n_train_items=10)
