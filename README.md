# Walkthrough - Knapsack Solver Comparison

I have implemented and compared four approaches to the 0-1 Knapsack Problem:
1.  **Dynamic Programming (DP)**: Exact solution (Ground Truth).
2.  **Greedy**: Heuristic (Value/Weight ratio).
3.  **Transformer (SL)**: Supervised Learning (mimicking DP).
4.  **Transformer (RL)**: Reinforcement Learning (REINFORCE).
5.  **Neural Greedy (Constructive)**: Using the Transformer's output as priority scores for a greedy selection.

## Execution
Run the comparison using `uv`:

```bash
uv run main.py
```

## Final Results (N=20 Items, 50 Samples, 5 Epochs Training)

| Method | Avg Value | Optimal % | Valid % | Time (ms) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DP** | 774.28 | **100.00%** | 100.0% | ~4.45 | Optimal Baseline |
| **Greedy** | 769.38 | 99.37% | 100.0% | **~0.05** | Very hard to beat |
| **SL_Raw** | 428.02 | 55.28% | 62.0% | ~0.00 | Failed to learn constraints well |
| **SL_Constr** | **769.08** | **99.33%** | **100.0%** | ~2.68 | **Matches Greedy Performance!** |
| **RL_Raw** | 87.34 | 11.28% | 100.0% | ~0.00 | Learned to be overly safe |
| **RL_Constr** | 735.68 | 95.01% | 100.0% | ~2.48 | Good heuristic, slightly worse than SL |

## Key Findings
- **Constructive Decoding Solves Validity**: By using the model to *sort* items rather than just classify them, we achieve **100% validity** and near-optimal performance (99.33%).
- **Neural Greedy**: The `SL_Constructive` model effectively learned a heuristic that is statistically indistinguishable from the "Value/Weight" greedy heuristic on this dataset.
- **Comparison**:
    - **DP**: Best result, slowest scaling.
    - **Classic Greedy**: Best trade-off (Instant & 99% Optimal).
    - **Neural Greedy**: Good proof of concept. It matches the classic heuristic without being explicitly programmed with the "Value/Weight" rule—it *learned* to prioritize items similarly.

## Conclusion
The Transformer successfully learned to solve the Knapsack problem when paired with a **Constructive Decoder** (Neural Greedy). While the raw classification output struggled with constraints, the learned rankings were high quality.
