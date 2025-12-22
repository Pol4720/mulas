# Balanced Multi-Bin Packing with Capacity Constraints

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python implementation for solving the **Balanced Multi-Bin Packing with Capacity Constraints** problem - an NP-hard combinatorial optimization problem with applications in logistics and load distribution.

## 📋 Problem Description

Given:
- A set of **n items**, each with weight $w_i$ and value $v_i$
- **k bins** with capacity $C$

Objective:
- Minimize the **maximum difference** in total values between bins
- While respecting **capacity constraints**

### Mathematical Formulation

$$\min z = \max_{j=1}^{k} \sum_{i=1}^{n} v_i \cdot x_{ij} - \min_{j=1}^{k} \sum_{i=1}^{n} v_i \cdot x_{ij}$$

Here, the decision variable $x_{ij}$ indicates whether item `i` is assigned to bin `j`:

$$
x_{ij} = \begin{cases}
1 & \text{if item } i \text{ is assigned to bin } j, \\
0 & \text{otherwise.}
\end{cases}
$$

The variables are binary, i.e. $x_{ij} \in \{0,1\}$.

Subject to:
- Each item assigned to exactly one bin

$$
\forall\; i\in\{1,\dots,n\}:\qquad \sum_{j=1}^{k} x_{ij} = 1
$$

- Capacity constraints: $\sum_i w_i \cdot x_{ij} \leq C$

## 🚀 Features

### Algorithms
- **Greedy**: First Fit Decreasing (FFD), Best Fit Decreasing (BFD), Worst Fit Decreasing (WFD), LPT Balanced
- **Metaheuristics**: Simulated Annealing, Genetic Algorithm, Tabu Search
- **Exact**: Dynamic Programming, Branch and Bound
- **Approximation**: LPT Approximation, Multi-Start Local Search, GRASP

### Visualization
- Interactive Plotly charts
- Manim algorithm animations
- Solution comparison plots
- Convergence analysis

### Tools
- Benchmark runner with statistical analysis
- Instance generators (uniform, bimodal, correlated)
- LaTeX/Markdown export
- Interactive Streamlit dashboard

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/your-repo/mulas.git
cd mulas

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Quick Start

```python
from discrete_logistics.core import Problem, Item
from discrete_logistics.algorithms import SimulatedAnnealing

# Create problem instance
items = [
    Item("item_1", weight=10, value=25),
    Item("item_2", weight=15, value=30),
    Item("item_3", weight=8, value=20),
    Item("item_4", weight=12, value=28),
    Item("item_5", weight=6, value=15),
]

problem = Problem(
    items=items,
    num_bins=2,
    bin_capacity=30,
    name="example"
)

# Solve with Simulated Annealing
solver = SimulatedAnnealing(
    initial_temp=100,
    cooling_rate=0.995,
    max_iterations=5000
)
solution = solver.solve(problem)

# Display results
print(f"Objective (value difference): {solution.objective:.4f}")
for i, bin_obj in enumerate(solution.bins):
    print(f"Bin {i+1}: {[item.id for item in bin_obj.items]}")
```

## 📊 Dashboard

Launch the interactive web dashboard:

```bash
cd discrete_logistics/dashboard
streamlit run app.py
```

Features:
- Problem configuration with presets
- Algorithm selection and parameter tuning
- Real-time visualization
- Results export (JSON, CSV, LaTeX)

## 📁 Project Structure

```
discrete_logistics/
├── core/                   # Data structures
│   ├── problem.py         # Item, Bin, Solution, Problem classes
│   └── instance_generator.py
├── algorithms/             # Algorithm implementations
│   ├── base.py            # Abstract base class
│   ├── greedy.py          # FFD, BFD, WFD, LPT
│   ├── dynamic_programming.py
│   ├── branch_and_bound.py
│   ├── metaheuristics.py  # SA, GA, Tabu Search
│   └── approximation.py
├── visualizations/         # Plotting and animations
│   ├── plots.py
│   ├── animations.py
│   └── interactive.py
├── theory/                 # Mathematical documentation
│   ├── formalization.py
│   ├── complexity.py
│   └── pseudocode.py
├── benchmarks/             # Performance testing
│   ├── runner.py
│   ├── instances.py
│   └── analysis.py
├── dashboard/              # Streamlit app
│   ├── app.py
│   └── components.py
└── utils/                  # Utilities
    ├── validators.py
    ├── exporters.py
    └── helpers.py

Report/                     # LaTeX report
└── main.tex
```

## 🧪 Running Benchmarks

```python
from discrete_logistics.benchmarks import BenchmarkRunner, StandardInstances
from discrete_logistics.algorithms import *

# Create runner
runner = BenchmarkRunner()

# Add algorithms
runner.add_algorithm('FFD', FirstFitDecreasing())
runner.add_algorithm('SA', SimulatedAnnealing())
runner.add_algorithm('GA', GeneticAlgorithm())

# Add test instances
instances = StandardInstances.medium_uniform()
runner.add_problems(instances.instances)

# Run benchmark
results = runner.run()

# Print summary
print(runner.get_comparison_table())

# Export results
runner.export_results('benchmark_results.json')
```

## 📈 Algorithm Comparison

| Algorithm | Time Complexity | Space | Quality | Best For |
|-----------|----------------|-------|---------|----------|
| FFD | O(n log n) | O(n) | ⭐⭐ | Quick baseline |
| LPT | O(n log n) | O(n) | ⭐⭐⭐ | Balanced solutions |
| SA | O(I·n) | O(n) | ⭐⭐⭐⭐ | Medium instances |
| GA | O(G·P·n) | O(P·n) | ⭐⭐⭐⭐ | Diverse search |
| B&B | O(k^n) | O(n) | ⭐⭐⭐⭐⭐ | Small instances (n<20) |

## 📚 Theory

The problem is **NP-hard** (proven by reduction from 3-PARTITION).

Key theoretical results:
- LPT approximation ratio: $(4/3 - 1/(3k))$ for makespan
- Greedy algorithms provide feasible but potentially suboptimal solutions
- Metaheuristics offer good trade-off between quality and runtime

See `Report/main.tex` for complete mathematical formalization and proofs.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 📖 Citation

```bibtex
@software{balanced_bin_packing,
  title = {Balanced Multi-Bin Packing with Capacity Constraints},
  author = {DAA Project},
  year = {2024},
  url = {https://github.com/your-repo/mulas}
}
```
Final DAA project in Computer Science Degree
