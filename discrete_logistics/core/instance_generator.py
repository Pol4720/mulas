"""
Instance Generator for the Balanced Multi-Bin Packing Problem.

Generates various types of test instances:
- Random instances with different distributions
- Structured instances (correlated, clustered)
- Benchmark instances from literature
- Pathological/adversarial instances
"""

import numpy as np
from typing import List, Optional, Tuple, Literal
import json
from .problem import Problem, Item


class InstanceGenerator:
    """
    Generates problem instances for testing and benchmarking.
    
    Supports various distribution types and correlation patterns
    to create diverse test cases.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        self.seed = seed
    
    def set_seed(self, seed: int):
        """Set a new random seed."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    
    def generate_uniform(
        self,
        n_items: int,
        num_bins: int,
        weight_range: Tuple[float, float] = (1, 100),
        value_range: Tuple[float, float] = (1, 100),
        capacity_factor: float = 1.5,
        name: Optional[str] = None
    ) -> Problem:
        """
        Generate instance with uniformly distributed weights and values.
        
        Args:
            n_items: Number of items
            num_bins: Number of bins
            weight_range: (min, max) for uniform weight distribution
            value_range: (min, max) for uniform value distribution
            capacity_factor: Multiply average weight load by this to get capacity
            name: Instance name
            
        Returns:
            Problem instance
        """
        weights = self.rng.uniform(weight_range[0], weight_range[1], n_items)
        values = self.rng.uniform(value_range[0], value_range[1], n_items)
        
        # Calculate capacity to ensure feasibility
        avg_weight_per_bin = weights.sum() / num_bins
        capacity = avg_weight_per_bin * capacity_factor
        capacity = max(capacity, weights.max() + 0.01)  # Must fit largest item
        
        items = [
            Item(id=i, weight=float(w), value=float(v))
            for i, (w, v) in enumerate(zip(weights, values))
        ]
        
        instance_name = name or f"uniform_n{n_items}_k{num_bins}"
        
        return Problem(
            items=items,
            num_bins=num_bins,
            bin_capacity=float(capacity),
            name=instance_name
        )
    
    def generate_normal(
        self,
        n_items: int,
        num_bins: int,
        weight_mean: float = 50,
        weight_std: float = 15,
        value_mean: float = 50,
        value_std: float = 15,
        capacity_factor: float = 1.5,
        name: Optional[str] = None
    ) -> Problem:
        """
        Generate instance with normally distributed weights and values.
        
        Args:
            n_items: Number of items
            num_bins: Number of bins
            weight_mean: Mean weight
            weight_std: Weight standard deviation
            value_mean: Mean value
            value_std: Value standard deviation
            capacity_factor: Capacity multiplier
            name: Instance name
            
        Returns:
            Problem instance
        """
        weights = np.abs(self.rng.normal(weight_mean, weight_std, n_items))
        weights = np.clip(weights, 1, None)  # Ensure positive
        
        values = np.abs(self.rng.normal(value_mean, value_std, n_items))
        values = np.clip(values, 1, None)
        
        avg_weight_per_bin = weights.sum() / num_bins
        capacity = avg_weight_per_bin * capacity_factor
        capacity = max(capacity, weights.max() + 0.01)
        
        items = [
            Item(id=i, weight=float(w), value=float(v))
            for i, (w, v) in enumerate(zip(weights, values))
        ]
        
        instance_name = name or f"normal_n{n_items}_k{num_bins}"
        
        return Problem(
            items=items,
            num_bins=num_bins,
            bin_capacity=float(capacity),
            name=instance_name
        )
    
    def generate_correlated(
        self,
        n_items: int,
        num_bins: int,
        correlation: float = 0.8,
        weight_range: Tuple[float, float] = (1, 100),
        capacity_factor: float = 1.5,
        name: Optional[str] = None
    ) -> Problem:
        """
        Generate instance where value is correlated with weight.
        
        value_i ≈ weight_i * factor + noise
        
        Args:
            n_items: Number of items
            num_bins: Number of bins
            correlation: Target correlation coefficient (-1 to 1)
            weight_range: (min, max) for weight distribution
            capacity_factor: Capacity multiplier
            name: Instance name
            
        Returns:
            Problem instance
        """
        weights = self.rng.uniform(weight_range[0], weight_range[1], n_items)
        
        # Generate correlated values
        noise = self.rng.normal(0, 1, n_items)
        values = correlation * (weights - weights.mean()) / weights.std() + \
                 np.sqrt(1 - correlation**2) * noise
        values = values - values.min() + 1  # Shift to positive
        values = values * (weight_range[1] - weight_range[0]) / (values.max() - values.min() + 1e-6)
        values = np.clip(values, 1, None)
        
        avg_weight_per_bin = weights.sum() / num_bins
        capacity = avg_weight_per_bin * capacity_factor
        capacity = max(capacity, weights.max() + 0.01)
        
        items = [
            Item(id=i, weight=float(w), value=float(v))
            for i, (w, v) in enumerate(zip(weights, values))
        ]
        
        instance_name = name or f"corr{correlation:.1f}_n{n_items}_k{num_bins}"
        
        return Problem(
            items=items,
            num_bins=num_bins,
            bin_capacity=float(capacity),
            name=instance_name
        )
    
    def generate_clustered(
        self,
        n_items: int,
        num_bins: int,
        n_clusters: int = 3,
        weight_range: Tuple[float, float] = (1, 100),
        value_range: Tuple[float, float] = (1, 100),
        cluster_std: float = 5,
        capacity_factor: float = 1.5,
        name: Optional[str] = None
    ) -> Problem:
        """
        Generate instance with clustered item characteristics.
        
        Items are grouped around cluster centers with some noise.
        
        Args:
            n_items: Number of items
            num_bins: Number of bins
            n_clusters: Number of item clusters
            weight_range: (min, max) range for cluster centers
            value_range: (min, max) range for cluster centers
            cluster_std: Standard deviation within clusters
            capacity_factor: Capacity multiplier
            name: Instance name
            
        Returns:
            Problem instance
        """
        items_per_cluster = n_items // n_clusters
        remainder = n_items % n_clusters
        
        weights = []
        values = []
        
        # Generate cluster centers
        weight_centers = self.rng.uniform(
            weight_range[0] + cluster_std * 2,
            weight_range[1] - cluster_std * 2,
            n_clusters
        )
        value_centers = self.rng.uniform(
            value_range[0] + cluster_std * 2,
            value_range[1] - cluster_std * 2,
            n_clusters
        )
        
        for i in range(n_clusters):
            n = items_per_cluster + (1 if i < remainder else 0)
            w = self.rng.normal(weight_centers[i], cluster_std, n)
            v = self.rng.normal(value_centers[i], cluster_std, n)
            weights.extend(w)
            values.extend(v)
        
        weights = np.clip(np.array(weights), 1, weight_range[1])
        values = np.clip(np.array(values), 1, value_range[1])
        
        avg_weight_per_bin = weights.sum() / num_bins
        capacity = avg_weight_per_bin * capacity_factor
        capacity = max(capacity, weights.max() + 0.01)
        
        items = [
            Item(id=i, weight=float(w), value=float(v))
            for i, (w, v) in enumerate(zip(weights, values))
        ]
        
        instance_name = name or f"clustered_n{n_items}_k{num_bins}_c{n_clusters}"
        
        return Problem(
            items=items,
            num_bins=num_bins,
            bin_capacity=float(capacity),
            name=instance_name
        )
    
    def generate_bimodal(
        self,
        n_items: int,
        num_bins: int,
        small_range: Tuple[float, float] = (1, 20),
        large_range: Tuple[float, float] = (80, 100),
        small_fraction: float = 0.7,
        capacity_factor: float = 1.5,
        name: Optional[str] = None
    ) -> Problem:
        """
        Generate instance with bimodal distribution (many small, few large items).
        
        Args:
            n_items: Number of items
            num_bins: Number of bins
            small_range: (min, max) for small items
            large_range: (min, max) for large items
            small_fraction: Fraction of items that are small
            capacity_factor: Capacity multiplier
            name: Instance name
            
        Returns:
            Problem instance
        """
        n_small = int(n_items * small_fraction)
        n_large = n_items - n_small
        
        small_weights = self.rng.uniform(small_range[0], small_range[1], n_small)
        small_values = self.rng.uniform(small_range[0], small_range[1], n_small)
        
        large_weights = self.rng.uniform(large_range[0], large_range[1], n_large)
        large_values = self.rng.uniform(large_range[0], large_range[1], n_large)
        
        weights = np.concatenate([small_weights, large_weights])
        values = np.concatenate([small_values, large_values])
        
        # Shuffle
        indices = self.rng.permutation(n_items)
        weights = weights[indices]
        values = values[indices]
        
        avg_weight_per_bin = weights.sum() / num_bins
        capacity = avg_weight_per_bin * capacity_factor
        capacity = max(capacity, weights.max() + 0.01)
        
        items = [
            Item(id=i, weight=float(w), value=float(v))
            for i, (w, v) in enumerate(zip(weights, values))
        ]
        
        instance_name = name or f"bimodal_n{n_items}_k{num_bins}"
        
        return Problem(
            items=items,
            num_bins=num_bins,
            bin_capacity=float(capacity),
            name=instance_name
        )
    
    def generate_adversarial(
        self,
        n_items: int,
        num_bins: int,
        adversary_type: Literal["greedy_bad", "large_items", "imbalanced"] = "greedy_bad",
        capacity_factor: float = 1.5,
        name: Optional[str] = None
    ) -> Problem:
        """
        Generate adversarial instances that are hard for specific algorithms.
        
        Args:
            n_items: Number of items
            num_bins: Number of bins
            adversary_type: Type of adversarial instance
            capacity_factor: Capacity multiplier
            name: Instance name
            
        Returns:
            Problem instance
        """
        if adversary_type == "greedy_bad":
            # Instance where greedy by value performs poorly
            # Large values but also large weights, interspersed with
            # small value, small weight items that pack better
            weights = []
            values = []
            
            for i in range(n_items):
                if i % 3 == 0:  # Large items
                    weights.append(self.rng.uniform(70, 90))
                    values.append(self.rng.uniform(80, 100))
                else:  # Small items that together beat large ones
                    weights.append(self.rng.uniform(10, 25))
                    values.append(self.rng.uniform(35, 45))
                    
        elif adversary_type == "large_items":
            # Items close to capacity - hard to pack efficiently
            base_capacity = 100
            weights = self.rng.uniform(base_capacity * 0.4, base_capacity * 0.6, n_items)
            values = self.rng.uniform(40, 60, n_items)
            capacity_factor = 1.0  # Tight capacity
            
        elif adversary_type == "imbalanced":
            # Highly imbalanced values
            weights = self.rng.uniform(10, 30, n_items)
            values = np.zeros(n_items)
            # Few items with very high value
            high_value_count = max(2, n_items // 10)
            high_indices = self.rng.choice(n_items, high_value_count, replace=False)
            for idx in high_indices:
                values[idx] = self.rng.uniform(80, 100)
            # Rest have low values
            for i in range(n_items):
                if values[i] == 0:
                    values[i] = self.rng.uniform(1, 10)
        else:
            raise ValueError(f"Unknown adversary type: {adversary_type}")
        
        weights = np.array(weights)
        values = np.array(values)
        
        avg_weight_per_bin = weights.sum() / num_bins
        capacity = avg_weight_per_bin * capacity_factor
        capacity = max(capacity, weights.max() + 0.01)
        
        items = [
            Item(id=i, weight=float(w), value=float(v))
            for i, (w, v) in enumerate(zip(weights, values))
        ]
        
        instance_name = name or f"adversarial_{adversary_type}_n{n_items}_k{num_bins}"
        
        return Problem(
            items=items,
            num_bins=num_bins,
            bin_capacity=float(capacity),
            name=instance_name
        )
    
    def generate_equal_partition(
        self,
        n_items: int,
        value_range: Tuple[float, float] = (1, 100),
        capacity_factor: float = 2.0,
        name: Optional[str] = None
    ) -> Problem:
        """
        Generate a 2-partition problem instance (k=2).
        
        This is related to the classic PARTITION problem.
        
        Args:
            n_items: Number of items
            value_range: (min, max) for values
            capacity_factor: Capacity multiplier
            name: Instance name
            
        Returns:
            Problem instance with 2 bins
        """
        values = self.rng.uniform(value_range[0], value_range[1], n_items)
        weights = self.rng.uniform(value_range[0], value_range[1], n_items)
        
        avg_weight_per_bin = weights.sum() / 2
        capacity = avg_weight_per_bin * capacity_factor
        capacity = max(capacity, weights.max() + 0.01)
        
        items = [
            Item(id=i, weight=float(w), value=float(v))
            for i, (w, v) in enumerate(zip(weights, values))
        ]
        
        instance_name = name or f"partition_n{n_items}"
        
        return Problem(
            items=items,
            num_bins=2,
            bin_capacity=float(capacity),
            name=instance_name
        )
    
    def generate_scaled(
        self,
        base_problem: Problem,
        scale_factor: float,
        name: Optional[str] = None
    ) -> Problem:
        """
        Scale an existing problem instance.
        
        Args:
            base_problem: Problem to scale
            scale_factor: Multiplier for items and bins
            name: New instance name
            
        Returns:
            Scaled problem instance
        """
        n_items = int(base_problem.n_items * scale_factor)
        num_bins = int(base_problem.num_bins * scale_factor)
        
        # Replicate items with slight variations
        items = []
        for i in range(n_items):
            base_item = base_problem.items[i % base_problem.n_items]
            noise = self.rng.uniform(0.95, 1.05)
            items.append(Item(
                id=i,
                weight=base_item.weight * noise,
                value=base_item.value * noise
            ))
        
        instance_name = name or f"{base_problem.name}_scaled_{scale_factor:.1f}x"
        
        return Problem(
            items=items,
            num_bins=max(1, num_bins),
            bin_capacity=base_problem.bin_capacity,
            name=instance_name
        )
    
    def generate_benchmark_suite(
        self,
        sizes: List[int] = [10, 20, 50, 100],
        num_bins_list: List[int] = [3, 5, 10],
        instance_types: List[str] = ["uniform", "normal", "correlated", "clustered", "bimodal"]
    ) -> List[Problem]:
        """
        Generate a comprehensive benchmark suite.
        
        Args:
            sizes: List of problem sizes (n_items)
            num_bins_list: List of bin counts
            instance_types: Types of instances to generate
            
        Returns:
            List of problem instances
        """
        problems = []
        
        for n in sizes:
            for k in num_bins_list:
                if k > n:  # Skip if more bins than items
                    continue
                    
                for inst_type in instance_types:
                    if inst_type == "uniform":
                        p = self.generate_uniform(n, k)
                    elif inst_type == "normal":
                        p = self.generate_normal(n, k)
                    elif inst_type == "correlated":
                        p = self.generate_correlated(n, k, correlation=0.7)
                    elif inst_type == "clustered":
                        p = self.generate_clustered(n, k, n_clusters=min(5, n//3))
                    elif inst_type == "bimodal":
                        p = self.generate_bimodal(n, k)
                    else:
                        continue
                    
                    problems.append(p)
        
        return problems
    
    def save_instance(self, problem: Problem, filepath: str):
        """Save a problem instance to JSON file."""
        problem.to_json(filepath)
    
    def load_instance(self, filepath: str) -> Problem:
        """Load a problem instance from JSON file."""
        return Problem.from_json(filepath)
    
    def save_benchmark_suite(self, problems: List[Problem], directory: str):
        """Save a benchmark suite to a directory."""
        import os
        os.makedirs(directory, exist_ok=True)
        
        manifest = []
        for i, p in enumerate(problems):
            filename = f"{p.name}.json"
            filepath = os.path.join(directory, filename)
            self.save_instance(p, filepath)
            manifest.append({
                "name": p.name,
                "filename": filename,
                "n_items": p.n_items,
                "num_bins": p.num_bins,
                "bin_capacity": p.bin_capacity
            })
        
        # Save manifest
        manifest_path = os.path.join(directory, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def load_benchmark_suite(self, directory: str) -> List[Problem]:
        """Load a benchmark suite from a directory."""
        import os
        
        manifest_path = os.path.join(directory, "manifest.json")
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        problems = []
        for entry in manifest:
            filepath = os.path.join(directory, entry["filename"])
            problems.append(self.load_instance(filepath))
        
        return problems
