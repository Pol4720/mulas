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

try:
    from .problem import Problem, Item
except ImportError:
    from discrete_logistics.core.problem import Problem, Item


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
        capacity_variation: float = 0.0,
        name: Optional[str] = None
    ) -> Problem:
        """
        Generate instance with uniformly distributed weights and values.
        
        Args:
            n_items: Number of items
            num_bins: Number of bins
            weight_range: (min, max) for uniform weight distribution
            value_range: (min, max) for uniform value distribution
            capacity_factor: Multiply average weight load by this to get base capacity
            capacity_variation: Variation factor for individual bin capacities (0-1)
                              0 = all bins same capacity, 1 = capacities vary up to 100%
            name: Instance name
            
        Returns:
            Problem instance
        """
        weights = self.rng.uniform(weight_range[0], weight_range[1], n_items)
        values = self.rng.uniform(value_range[0], value_range[1], n_items)
        
        # Calculate base capacity to ensure feasibility
        avg_weight_per_bin = weights.sum() / num_bins
        base_capacity = avg_weight_per_bin * capacity_factor
        base_capacity = max(base_capacity, weights.max() + 0.01)  # Must fit largest item
        
        # Generate individual bin capacities with variation
        if capacity_variation > 0:
            variations = 1 + self.rng.uniform(-capacity_variation, capacity_variation, num_bins)
            bin_capacities = [float(base_capacity * v) for v in variations]
            # Ensure at least one bin can fit the largest item
            max_cap_idx = np.argmax(bin_capacities)
            bin_capacities[max_cap_idx] = max(bin_capacities[max_cap_idx], weights.max() + 0.01)
        else:
            bin_capacities = [float(base_capacity)] * num_bins
        
        items = [
            Item(id=i, weight=float(w), value=float(v))
            for i, (w, v) in enumerate(zip(weights, values))
        ]
        
        instance_name = name or f"uniform_n{n_items}_k{num_bins}"
        
        return Problem(
            items=items,
            num_bins=num_bins,
            bin_capacities=bin_capacities,
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
        capacity_variation: float = 0.0,
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
            capacity_variation: Variation factor for individual bin capacities (0-1)
            name: Instance name
            
        Returns:
            Problem instance
        """
        weights = np.abs(self.rng.normal(weight_mean, weight_std, n_items))
        weights = np.clip(weights, 1, None)  # Ensure positive
        
        values = np.abs(self.rng.normal(value_mean, value_std, n_items))
        values = np.clip(values, 1, None)
        
        avg_weight_per_bin = weights.sum() / num_bins
        base_capacity = avg_weight_per_bin * capacity_factor
        base_capacity = max(base_capacity, weights.max() + 0.01)
        
        # Generate individual bin capacities with variation
        if capacity_variation > 0:
            variations = 1 + self.rng.uniform(-capacity_variation, capacity_variation, num_bins)
            bin_capacities = [float(base_capacity * v) for v in variations]
            max_cap_idx = np.argmax(bin_capacities)
            bin_capacities[max_cap_idx] = max(bin_capacities[max_cap_idx], weights.max() + 0.01)
        else:
            bin_capacities = [float(base_capacity)] * num_bins
        
        items = [
            Item(id=i, weight=float(w), value=float(v))
            for i, (w, v) in enumerate(zip(weights, values))
        ]
        
        instance_name = name or f"normal_n{n_items}_k{num_bins}"
        
        return Problem(
            items=items,
            num_bins=num_bins,
            bin_capacities=bin_capacities,
            name=instance_name
        )
    
    def generate_correlated(
        self,
        n_items: int,
        num_bins: int,
        correlation: float = 0.8,
        weight_range: Tuple[float, float] = (1, 100),
        capacity_factor: float = 1.5,
        capacity_variation: float = 0.0,
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
            capacity_variation: Variation factor for individual bin capacities (0-1)
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
        base_capacity = avg_weight_per_bin * capacity_factor
        base_capacity = max(base_capacity, weights.max() + 0.01)
        
        # Generate individual bin capacities with variation
        if capacity_variation > 0:
            variations = 1 + self.rng.uniform(-capacity_variation, capacity_variation, num_bins)
            bin_capacities = [float(base_capacity * v) for v in variations]
            max_cap_idx = np.argmax(bin_capacities)
            bin_capacities[max_cap_idx] = max(bin_capacities[max_cap_idx], weights.max() + 0.01)
        else:
            bin_capacities = [float(base_capacity)] * num_bins
        
        items = [
            Item(id=i, weight=float(w), value=float(v))
            for i, (w, v) in enumerate(zip(weights, values))
        ]
        
        instance_name = name or f"corr{correlation:.1f}_n{n_items}_k{num_bins}"
        
        return Problem(
            items=items,
            num_bins=num_bins,
            bin_capacities=bin_capacities,
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
        capacity_variation: float = 0.0,
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
            capacity_variation: Variation factor for individual bin capacities (0-1)
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
        base_capacity = avg_weight_per_bin * capacity_factor
        base_capacity = max(base_capacity, weights.max() + 0.01)
        
        # Generate individual bin capacities with variation
        if capacity_variation > 0:
            variations = 1 + self.rng.uniform(-capacity_variation, capacity_variation, num_bins)
            bin_capacities = [float(base_capacity * v) for v in variations]
            max_cap_idx = np.argmax(bin_capacities)
            bin_capacities[max_cap_idx] = max(bin_capacities[max_cap_idx], weights.max() + 0.01)
        else:
            bin_capacities = [float(base_capacity)] * num_bins
        
        items = [
            Item(id=i, weight=float(w), value=float(v))
            for i, (w, v) in enumerate(zip(weights, values))
        ]
        
        instance_name = name or f"clustered_n{n_items}_k{num_bins}_c{n_clusters}"
        
        return Problem(
            items=items,
            num_bins=num_bins,
            bin_capacities=bin_capacities,
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
        capacity_variation: float = 0.0,
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
            capacity_variation: Variation factor for individual bin capacities (0-1)
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
        base_capacity = avg_weight_per_bin * capacity_factor
        base_capacity = max(base_capacity, weights.max() + 0.01)
        
        # Generate individual bin capacities with variation
        if capacity_variation > 0:
            variations = 1 + self.rng.uniform(-capacity_variation, capacity_variation, num_bins)
            bin_capacities = [float(base_capacity * v) for v in variations]
            max_cap_idx = np.argmax(bin_capacities)
            bin_capacities[max_cap_idx] = max(bin_capacities[max_cap_idx], weights.max() + 0.01)
        else:
            bin_capacities = [float(base_capacity)] * num_bins
        
        items = [
            Item(id=i, weight=float(w), value=float(v))
            for i, (w, v) in enumerate(zip(weights, values))
        ]
        
        instance_name = name or f"bimodal_n{n_items}_k{num_bins}"
        
        return Problem(
            items=items,
            num_bins=num_bins,
            bin_capacities=bin_capacities,
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
        base_capacity = avg_weight_per_bin * capacity_factor
        base_capacity = max(base_capacity, weights.max() + 0.01)
        
        bin_capacities = [float(base_capacity)] * num_bins
        
        items = [
            Item(id=i, weight=float(w), value=float(v))
            for i, (w, v) in enumerate(zip(weights, values))
        ]
        
        instance_name = name or f"adversarial_{adversary_type}_n{n_items}_k{num_bins}"
        
        return Problem(
            items=items,
            num_bins=num_bins,
            bin_capacities=bin_capacities,
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
        base_capacity = avg_weight_per_bin * capacity_factor
        base_capacity = max(base_capacity, weights.max() + 0.01)
        
        bin_capacities = [float(base_capacity), float(base_capacity)]
        
        items = [
            Item(id=i, weight=float(w), value=float(v))
            for i, (w, v) in enumerate(zip(weights, values))
        ]
        
        instance_name = name or f"partition_n{n_items}"
        
        return Problem(
            items=items,
            num_bins=2,
            bin_capacities=bin_capacities,
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
            bin_capacities=base_problem.bin_capacities[:max(1, num_bins)] if num_bins <= len(base_problem.bin_capacities) 
                          else base_problem.bin_capacities + [base_problem.bin_capacities[-1]] * (num_bins - len(base_problem.bin_capacities)),
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
                "bin_capacities": p.bin_capacities
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
    
    # =========================================================================
    # Edge Cases and Pathological Instances
    # =========================================================================
    
    def generate_perfect_balance(
        self,
        num_bins: int,
        items_per_bin: int = 3,
        value_per_bin: float = 100.0,
        name: Optional[str] = None
    ) -> Problem:
        """
        Generate instance where perfect balance (diff=0) is achievable.
        
        Creates items such that they can be perfectly partitioned among bins
        with equal total value per bin. Useful for testing if algorithms
        can find the global optimum.
        
        Args:
            num_bins: Number of bins
            items_per_bin: Items designed for each bin
            value_per_bin: Target value sum per bin
            name: Instance name
            
        Returns:
            Problem instance with achievable perfect balance
        """
        items = []
        n_items = num_bins * items_per_bin
        
        for bin_idx in range(num_bins):
            # Create items for this bin that sum to value_per_bin
            remaining_value = value_per_bin
            for j in range(items_per_bin):
                if j < items_per_bin - 1:
                    # Random portion of remaining
                    value = self.rng.uniform(0.2, 0.5) * remaining_value
                else:
                    # Last item gets the rest
                    value = remaining_value
                
                remaining_value -= value
                weight = self.rng.uniform(5, 15)
                
                items.append(Item(
                    id=len(items),
                    weight=float(weight),
                    value=float(value)
                ))
        
        # Shuffle to avoid trivial ordering
        self.rng.shuffle(items)
        # Reassign IDs after shuffle
        for i, item in enumerate(items):
            item._id = i
        
        # Capacity to fit all items easily
        total_weight = sum(item.weight for item in items)
        capacity = total_weight / num_bins * 1.5
        
        instance_name = name or f"perfect_balance_k{num_bins}_n{n_items}"
        
        return Problem(
            items=items,
            num_bins=num_bins,
            bin_capacities=[float(capacity)] * num_bins,
            name=instance_name
        )
    
    def generate_impossible_balance(
        self,
        num_bins: int,
        n_items: int = 10,
        name: Optional[str] = None
    ) -> Problem:
        """
        Generate instance where perfect balance is impossible.
        
        Creates one item with value much larger than all others combined,
        guaranteeing a non-zero optimal difference.
        
        Args:
            num_bins: Number of bins
            n_items: Number of items
            name: Instance name
            
        Returns:
            Problem instance where diff > 0 is unavoidable
        """
        items = []
        
        # One dominant item
        items.append(Item(
            id=0,
            weight=20.0,
            value=1000.0  # Much larger than others
        ))
        
        # Rest are small
        for i in range(1, n_items):
            items.append(Item(
                id=i,
                weight=self.rng.uniform(5, 15),
                value=self.rng.uniform(10, 50)
            ))
        
        capacity = sum(item.weight for item in items) / num_bins * 2.0
        
        instance_name = name or f"impossible_balance_k{num_bins}_n{n_items}"
        
        return Problem(
            items=items,
            num_bins=num_bins,
            bin_capacities=[float(capacity)] * num_bins,
            name=instance_name
        )
    
    def generate_tight_capacity(
        self,
        n_items: int,
        num_bins: int,
        slack_factor: float = 0.05,
        name: Optional[str] = None
    ) -> Problem:
        """
        Generate instance with very tight capacity constraints.
        
        Items almost fill bins completely, making many assignments infeasible.
        Tests algorithm behavior under pressure.
        
        Args:
            n_items: Number of items
            num_bins: Number of bins
            slack_factor: How much extra capacity (0.05 = 5% slack)
            name: Instance name
            
        Returns:
            Problem instance with tight capacities
        """
        weights = self.rng.uniform(10, 50, n_items)
        values = self.rng.uniform(10, 50, n_items)
        
        total_weight = weights.sum()
        capacity_per_bin = total_weight / num_bins * (1 + slack_factor)
        
        # Ensure at least one item fits in each bin
        capacity_per_bin = max(capacity_per_bin, weights.max() + 1)
        
        items = [
            Item(id=i, weight=float(w), value=float(v))
            for i, (w, v) in enumerate(zip(weights, values))
        ]
        
        instance_name = name or f"tight_cap_n{n_items}_k{num_bins}_s{slack_factor:.2f}"
        
        return Problem(
            items=items,
            num_bins=num_bins,
            bin_capacities=[float(capacity_per_bin)] * num_bins,
            name=instance_name
        )
    
    def generate_single_large_item(
        self,
        n_items: int,
        num_bins: int,
        large_item_fraction: float = 0.4,
        name: Optional[str] = None
    ) -> Problem:
        """
        Generate instance with one item taking significant capacity.
        
        Tests handling of items that dominate bin capacity/value.
        
        Args:
            n_items: Number of items
            num_bins: Number of bins
            large_item_fraction: Fraction of total capacity for large item
            name: Instance name
            
        Returns:
            Problem instance with one dominant item
        """
        # Small items
        small_weights = self.rng.uniform(5, 15, n_items - 1)
        small_values = self.rng.uniform(10, 30, n_items - 1)
        
        # Large item
        total_small_weight = small_weights.sum()
        large_weight = total_small_weight * large_item_fraction / (1 - large_item_fraction)
        large_value = sum(small_values) * large_item_fraction / (1 - large_item_fraction)
        
        weights = np.append([large_weight], small_weights)
        values = np.append([large_value], small_values)
        
        total_weight = weights.sum()
        capacity = total_weight / num_bins * 1.3
        capacity = max(capacity, large_weight + 1)  # Must fit the large item
        
        items = [
            Item(id=i, weight=float(w), value=float(v))
            for i, (w, v) in enumerate(zip(weights, values))
        ]
        
        instance_name = name or f"single_large_n{n_items}_k{num_bins}"
        
        return Problem(
            items=items,
            num_bins=num_bins,
            bin_capacities=[float(capacity)] * num_bins,
            name=instance_name
        )
    
    def generate_3partition_instance(
        self,
        m: int,
        B: int = 100,
        name: Optional[str] = None
    ) -> Problem:
        """
        Generate 3-PARTITION style instance.
        
        Creates 3m items where perfect partition into m groups of 3 exists.
        Based on the NP-complete 3-PARTITION problem.
        
        Reference: Garey & Johnson (1979)
        
        Args:
            m: Number of bins (groups)
            B: Target sum per group
            name: Instance name
            
        Returns:
            Problem instance modeling 3-PARTITION
        """
        items = []
        
        for group in range(m):
            # Generate 3 items that sum to B
            # Using constraint: B/4 < a_i < B/2
            a1 = self.rng.uniform(B/4 + 1, B/2 - 1)
            a2 = self.rng.uniform(B/4 + 1, min(B/2 - 1, B - a1 - B/4 - 1))
            a3 = B - a1 - a2
            
            # Ensure constraint is satisfied
            if a3 <= B/4 or a3 >= B/2:
                # Retry with fixed values
                a1 = B/3 + self.rng.uniform(-B/20, B/20)
                a2 = B/3 + self.rng.uniform(-B/20, B/20)
                a3 = B - a1 - a2
            
            for val in [a1, a2, a3]:
                items.append(Item(
                    id=len(items),
                    weight=float(val),  # weight = value for 3-PARTITION
                    value=float(val)
                ))
        
        # Shuffle items
        self.rng.shuffle(items)
        for i, item in enumerate(items):
            item._id = i
        
        instance_name = name or f"3partition_m{m}_B{B}"
        
        return Problem(
            items=items,
            num_bins=m,
            bin_capacities=[float(B)] * m,  # Exact capacity = B
            name=instance_name
        )
    
    def generate_makespan_hard(
        self,
        n_items: int,
        num_bins: int,
        name: Optional[str] = None
    ) -> Problem:
        """
        Generate instance hard for makespan scheduling.
        
        Based on LPT worst-case constructions from Graham (1969).
        
        Args:
            n_items: Number of items
            num_bins: Number of bins
            name: Instance name
            
        Returns:
            Problem instance challenging for LPT
        """
        # LPT worst case: items of sizes 2k-1, 2k-1, 2k for k machines
        # Generalized for arbitrary n
        k = num_bins
        
        values = []
        for i in range(n_items):
            if i < 2 * k:
                values.append(2 * k - 1)
            else:
                values.append(2 * k)
        
        # Add some randomness
        values = np.array(values, dtype=float)
        values += self.rng.uniform(-0.5, 0.5, n_items)
        
        weights = self.rng.uniform(10, 30, n_items)
        total_weight = weights.sum()
        capacity = total_weight / num_bins * 1.5
        
        items = [
            Item(id=i, weight=float(w), value=float(v))
            for i, (w, v) in enumerate(zip(weights, values))
        ]
        
        instance_name = name or f"makespan_hard_n{n_items}_k{num_bins}"
        
        return Problem(
            items=items,
            num_bins=num_bins,
            bin_capacities=[float(capacity)] * num_bins,
            name=instance_name
        )
    
    def generate_test_suite_for_brute_force(
        self,
        max_items: int = 12,
        num_bins_options: List[int] = [2, 3, 4]
    ) -> List[Problem]:
        """
        Generate test suite suitable for brute force comparison.
        
        Creates small instances that brute force can solve optimally.
        
        Args:
            max_items: Maximum number of items (keep small for brute force)
            num_bins_options: List of bin counts to test
            
        Returns:
            List of small problem instances
        """
        problems = []
        
        for n in [6, 8, 10, max_items]:
            for k in num_bins_options:
                if k >= n:
                    continue
                
                # Various instance types
                problems.append(self.generate_uniform(n, k, name=f"bf_uniform_n{n}_k{k}"))
                problems.append(self.generate_perfect_balance(k, n // k, name=f"bf_perfect_n{n}_k{k}"))
                problems.append(self.generate_tight_capacity(n, k, slack_factor=0.1, name=f"bf_tight_n{n}_k{k}"))
                problems.append(self.generate_correlated(n, k, correlation=0.8, name=f"bf_corr_n{n}_k{k}"))
                
                if n >= 9 and k <= 3:
                    problems.append(self.generate_3partition_instance(k, name=f"bf_3part_n{3*k}_k{k}"))
        
        return problems
