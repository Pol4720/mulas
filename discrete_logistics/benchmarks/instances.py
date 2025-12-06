"""
Test Instance Sets Module
=========================

Provides standard test instances for benchmarking
bin packing algorithms.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

import sys
sys.path.insert(0, str(__file__).rsplit('benchmarks', 1)[0])

from core.problem import Problem, Item


@dataclass
class TestInstanceSet:
    """
    A collection of test instances with metadata.
    
    Attributes
    ----------
    name : str
        Name of the instance set
    description : str
        Description of the instance set
    instances : Dict[str, Problem]
        Dictionary of problem instances
    difficulty : str
        Difficulty level (easy, medium, hard)
    source : str
        Source of the instances
    """
    name: str
    description: str
    instances: Dict[str, Problem]
    difficulty: str = "medium"
    source: str = "generated"
    
    def __len__(self) -> int:
        return len(self.instances)
    
    def __iter__(self):
        return iter(self.instances.items())
    
    def get_instance(self, name: str) -> Optional[Problem]:
        """Get instance by name."""
        return self.instances.get(name)
    
    def filter_by_size(self, min_items: int = 0, max_items: int = float('inf')) -> 'TestInstanceSet':
        """Filter instances by size."""
        filtered = {
            name: prob for name, prob in self.instances.items()
            if min_items <= len(prob.items) <= max_items
        }
        return TestInstanceSet(
            name=f"{self.name}_filtered",
            description=f"Filtered: {min_items} <= n <= {max_items}",
            instances=filtered,
            difficulty=self.difficulty,
            source=self.source
        )


class StandardInstances:
    """
    Factory class for generating standard test instances.
    
    Provides various benchmark instance sets commonly used
    in bin packing research.
    """
    
    @staticmethod
    def small_uniform(seed: int = 42) -> TestInstanceSet:
        """
        Small instances with uniform distribution.
        
        Parameters
        ----------
        seed : int
            Random seed
            
        Returns
        -------
        TestInstanceSet
            Collection of small test instances
        """
        np.random.seed(seed)
        instances = {}
        
        configs = [
            (5, 2, 50),    # 5 items, 2 bins, capacity 50
            (8, 3, 50),
            (10, 3, 75),
            (12, 4, 60),
            (15, 4, 80),
        ]
        
        for n, k, c in configs:
            items = [
                Item(id=f"item_{i}", 
                     weight=np.random.uniform(5, 25),
                     value=np.random.uniform(10, 40))
                for i in range(n)
            ]
            
            instances[f"small_u_{n}_{k}"] = Problem(
                items=items,
                num_bins=k,
                bin_capacity=c,
                name=f"small_uniform_{n}_{k}_{c}"
            )
        
        return TestInstanceSet(
            name="small_uniform",
            description="Small instances (5-15 items) with uniform weight/value distribution",
            instances=instances,
            difficulty="easy",
            source="generated"
        )
    
    @staticmethod
    def medium_uniform(seed: int = 42) -> TestInstanceSet:
        """Medium instances with uniform distribution."""
        np.random.seed(seed)
        instances = {}
        
        configs = [
            (20, 4, 100),
            (25, 5, 100),
            (30, 5, 120),
            (35, 6, 120),
            (40, 6, 150),
            (50, 8, 150),
        ]
        
        for n, k, c in configs:
            items = [
                Item(id=f"item_{i}",
                     weight=np.random.uniform(5, 35),
                     value=np.random.uniform(10, 50))
                for i in range(n)
            ]
            
            instances[f"medium_u_{n}_{k}"] = Problem(
                items=items,
                num_bins=k,
                bin_capacity=c,
                name=f"medium_uniform_{n}_{k}_{c}"
            )
        
        return TestInstanceSet(
            name="medium_uniform",
            description="Medium instances (20-50 items) with uniform distribution",
            instances=instances,
            difficulty="medium",
            source="generated"
        )
    
    @staticmethod
    def large_uniform(seed: int = 42) -> TestInstanceSet:
        """Large instances with uniform distribution."""
        np.random.seed(seed)
        instances = {}
        
        configs = [
            (60, 8, 200),
            (75, 10, 200),
            (100, 10, 250),
            (120, 12, 250),
            (150, 15, 300),
        ]
        
        for n, k, c in configs:
            items = [
                Item(id=f"item_{i}",
                     weight=np.random.uniform(5, 40),
                     value=np.random.uniform(10, 60))
                for i in range(n)
            ]
            
            instances[f"large_u_{n}_{k}"] = Problem(
                items=items,
                num_bins=k,
                bin_capacity=c,
                name=f"large_uniform_{n}_{k}_{c}"
            )
        
        return TestInstanceSet(
            name="large_uniform",
            description="Large instances (60-150 items) with uniform distribution",
            instances=instances,
            difficulty="hard",
            source="generated"
        )
    
    @staticmethod
    def correlated_instances(seed: int = 42) -> TestInstanceSet:
        """Instances with correlated weights and values."""
        np.random.seed(seed)
        instances = {}
        
        correlations = [0.3, 0.5, 0.7, 0.9]
        
        for corr in correlations:
            n, k, c = 30, 5, 100
            
            weights = np.random.uniform(5, 30, n)
            noise = np.random.normal(0, 5, n)
            values = corr * weights + (1 - corr) * np.random.uniform(10, 40, n) + noise
            values = np.clip(values, 10, 50)
            
            items = [
                Item(id=f"item_{i}", weight=float(weights[i]), value=float(values[i]))
                for i in range(n)
            ]
            
            instances[f"corr_{int(corr*100)}"] = Problem(
                items=items,
                num_bins=k,
                bin_capacity=c,
                name=f"correlated_{int(corr*100)}"
            )
        
        return TestInstanceSet(
            name="correlated",
            description="Instances with varying weight-value correlation",
            instances=instances,
            difficulty="medium",
            source="generated"
        )
    
    @staticmethod
    def bimodal_instances(seed: int = 42) -> TestInstanceSet:
        """Instances with bimodal weight distribution."""
        np.random.seed(seed)
        instances = {}
        
        configs = [
            (20, 4, 80, "small"),
            (40, 6, 120, "medium"),
            (60, 8, 150, "large"),
        ]
        
        for n, k, c, size in configs:
            weights = []
            for _ in range(n):
                if np.random.random() < 0.5:
                    weights.append(np.random.normal(10, 2))
                else:
                    weights.append(np.random.normal(30, 3))
            
            weights = np.clip(weights, 5, 40)
            values = np.random.uniform(10, 50, n)
            
            items = [
                Item(id=f"item_{i}", weight=float(weights[i]), value=float(values[i]))
                for i in range(n)
            ]
            
            instances[f"bimodal_{size}"] = Problem(
                items=items,
                num_bins=k,
                bin_capacity=c,
                name=f"bimodal_{size}_{n}_{k}"
            )
        
        return TestInstanceSet(
            name="bimodal",
            description="Instances with bimodal weight distribution (light and heavy items)",
            instances=instances,
            difficulty="medium",
            source="generated"
        )
    
    @staticmethod
    def tight_capacity_instances(seed: int = 42) -> TestInstanceSet:
        """Instances where capacity constraints are tight."""
        np.random.seed(seed)
        instances = {}
        
        configs = [
            (15, 3, 80),   # Total weight ≈ 225, capacity = 240
            (25, 4, 100),  # Total weight ≈ 375, capacity = 400
            (35, 5, 120),  # Total weight ≈ 560, capacity = 600
        ]
        
        for n, k, c in configs:
            target_total = c * k * 0.95  # 95% utilization target
            avg_weight = target_total / n
            
            weights = np.random.uniform(avg_weight * 0.5, avg_weight * 1.5, n)
            # Scale to match target
            weights = weights * (target_total / weights.sum())
            
            values = np.random.uniform(10, 50, n)
            
            items = [
                Item(id=f"item_{i}", weight=float(weights[i]), value=float(values[i]))
                for i in range(n)
            ]
            
            instances[f"tight_{n}_{k}"] = Problem(
                items=items,
                num_bins=k,
                bin_capacity=c,
                name=f"tight_capacity_{n}_{k}_{c}"
            )
        
        return TestInstanceSet(
            name="tight_capacity",
            description="Instances with tight capacity constraints (~95% utilization)",
            instances=instances,
            difficulty="hard",
            source="generated"
        )
    
    @staticmethod
    def balanced_target_instances(seed: int = 42) -> TestInstanceSet:
        """Instances designed to have a known optimal balanced solution."""
        np.random.seed(seed)
        instances = {}
        
        # Create instances where perfect balance is achievable
        configs = [
            (12, 3, 100, 30),   # 12 items, 3 bins, each bin should have total value 30
            (20, 4, 100, 50),
            (30, 5, 120, 60),
        ]
        
        for n, k, c, target_value in configs:
            items_per_bin = n // k
            items = []
            
            for bin_idx in range(k):
                # Create items that sum to target_value
                bin_values = np.random.dirichlet(np.ones(items_per_bin)) * target_value
                bin_weights = np.random.uniform(5, 20, items_per_bin)
                
                for i in range(items_per_bin):
                    items.append(Item(
                        id=f"item_{bin_idx * items_per_bin + i}",
                        weight=float(bin_weights[i]),
                        value=float(bin_values[i])
                    ))
            
            # Shuffle items
            np.random.shuffle(items)
            # Re-assign IDs after shuffle
            for i, item in enumerate(items):
                items[i] = Item(id=f"item_{i}", weight=item.weight, value=item.value)
            
            instances[f"balanced_{n}_{k}"] = Problem(
                items=items,
                num_bins=k,
                bin_capacity=c,
                name=f"balanced_target_{n}_{k}"
            )
        
        return TestInstanceSet(
            name="balanced_target",
            description="Instances with known optimal balanced solution (objective = 0)",
            instances=instances,
            difficulty="medium",
            source="generated"
        )
    
    @staticmethod
    def scalability_suite(seed: int = 42) -> TestInstanceSet:
        """Suite for testing algorithm scalability."""
        np.random.seed(seed)
        instances = {}
        
        # Varying number of items
        for n in [10, 20, 30, 50, 75, 100, 150, 200]:
            k = max(2, n // 10)
            c = 100
            
            items = [
                Item(id=f"item_{i}",
                     weight=np.random.uniform(5, 25),
                     value=np.random.uniform(10, 40))
                for i in range(n)
            ]
            
            instances[f"scale_n{n}"] = Problem(
                items=items,
                num_bins=k,
                bin_capacity=c,
                name=f"scalability_n{n}_k{k}"
            )
        
        # Varying number of bins
        for k in [2, 3, 5, 8, 10, 15, 20]:
            n = 50
            c = 100
            
            items = [
                Item(id=f"item_{i}",
                     weight=np.random.uniform(5, 25),
                     value=np.random.uniform(10, 40))
                for i in range(n)
            ]
            
            instances[f"scale_k{k}"] = Problem(
                items=items,
                num_bins=k,
                bin_capacity=c,
                name=f"scalability_n{n}_k{k}"
            )
        
        return TestInstanceSet(
            name="scalability",
            description="Suite for testing algorithm scalability with varying n and k",
            instances=instances,
            difficulty="varied",
            source="generated"
        )
    
    @staticmethod
    def get_all_instances(seed: int = 42) -> Dict[str, TestInstanceSet]:
        """Get all standard instance sets."""
        return {
            'small_uniform': StandardInstances.small_uniform(seed),
            'medium_uniform': StandardInstances.medium_uniform(seed),
            'large_uniform': StandardInstances.large_uniform(seed),
            'correlated': StandardInstances.correlated_instances(seed),
            'bimodal': StandardInstances.bimodal_instances(seed),
            'tight_capacity': StandardInstances.tight_capacity_instances(seed),
            'balanced_target': StandardInstances.balanced_target_instances(seed),
            'scalability': StandardInstances.scalability_suite(seed),
        }
    
    @staticmethod
    def get_quick_test_set(seed: int = 42) -> Dict[str, Problem]:
        """Get a small set of instances for quick testing."""
        instances = {}
        small = StandardInstances.small_uniform(seed)
        
        for name, problem in list(small.instances.items())[:3]:
            instances[name] = problem
        
        return instances


class InstanceLoader:
    """
    Utility class for loading and saving problem instances.
    """
    
    @staticmethod
    def save_instance(problem: Problem, filepath: str):
        """Save a problem instance to JSON."""
        data = {
            'name': problem.name,
            'num_bins': problem.num_bins,
            'bin_capacity': problem.bin_capacity,
            'items': [
                {'id': item.id, 'weight': item.weight, 'value': item.value}
                for item in problem.items
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load_instance(filepath: str) -> Problem:
        """Load a problem instance from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        items = [
            Item(id=item['id'], weight=item['weight'], value=item['value'])
            for item in data['items']
        ]
        
        return Problem(
            items=items,
            num_bins=data['num_bins'],
            bin_capacity=data['bin_capacity'],
            name=data.get('name', Path(filepath).stem)
        )
    
    @staticmethod
    def save_instance_set(instance_set: TestInstanceSet, directory: str):
        """Save an entire instance set to a directory."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            'name': instance_set.name,
            'description': instance_set.description,
            'difficulty': instance_set.difficulty,
            'source': instance_set.source,
            'instances': list(instance_set.instances.keys())
        }
        
        with open(dir_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save each instance
        for name, problem in instance_set.instances.items():
            InstanceLoader.save_instance(problem, str(dir_path / f"{name}.json"))
    
    @staticmethod
    def load_instance_set(directory: str) -> TestInstanceSet:
        """Load an instance set from a directory."""
        dir_path = Path(directory)
        
        # Load metadata
        with open(dir_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Load instances
        instances = {}
        for name in metadata['instances']:
            filepath = dir_path / f"{name}.json"
            if filepath.exists():
                instances[name] = InstanceLoader.load_instance(str(filepath))
        
        return TestInstanceSet(
            name=metadata['name'],
            description=metadata['description'],
            instances=instances,
            difficulty=metadata.get('difficulty', 'medium'),
            source=metadata.get('source', 'loaded')
        )
