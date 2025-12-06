"""
Validators Module
=================

Input validation utilities for problem instances and solutions.
"""

from typing import List, Optional, Set
from dataclasses import dataclass

import sys
sys.path.insert(0, str(__file__).rsplit('utils', 1)[0])

from core.problem import Problem, Solution, Item, Bin


class ValidationError(Exception):
    """Exception raised for validation errors."""
    
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or [message]
    
    def __str__(self):
        if len(self.errors) == 1:
            return self.errors[0]
        return f"Multiple validation errors:\n" + "\n".join(f"  - {e}" for e in self.errors)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    
    def __bool__(self):
        return self.valid


def validate_item(item: Item) -> ValidationResult:
    """
    Validate a single item.
    
    Parameters
    ----------
    item : Item
        Item to validate
        
    Returns
    -------
    ValidationResult
        Validation result
    """
    errors = []
    warnings = []
    
    # Check ID
    if not item.id:
        errors.append("Item ID cannot be empty")
    
    # Check weight
    if item.weight <= 0:
        errors.append(f"Item {item.id}: Weight must be positive (got {item.weight})")
    
    # Check value
    if item.value < 0:
        errors.append(f"Item {item.id}: Value cannot be negative (got {item.value})")
    elif item.value == 0:
        warnings.append(f"Item {item.id}: Value is zero")
    
    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


def validate_problem(problem: Problem, raise_on_error: bool = True) -> ValidationResult:
    """
    Validate a problem instance.
    
    Checks:
    - All items have valid properties
    - Number of bins is positive
    - Bin capacity is positive
    - At least one item exists
    - Items can fit in bins (warning if total weight > total capacity)
    
    Parameters
    ----------
    problem : Problem
        Problem instance to validate
    raise_on_error : bool
        If True, raise ValidationError on invalid input
        
    Returns
    -------
    ValidationResult
        Validation result
        
    Raises
    ------
    ValidationError
        If validation fails and raise_on_error is True
    """
    errors = []
    warnings = []
    
    # Check basic parameters
    if problem.num_bins <= 0:
        errors.append(f"Number of bins must be positive (got {problem.num_bins})")
    
    if problem.bin_capacity <= 0:
        errors.append(f"Bin capacity must be positive (got {problem.bin_capacity})")
    
    if not problem.items:
        errors.append("Problem must have at least one item")
    
    # Validate each item
    item_ids: Set[str] = set()
    for item in problem.items:
        item_result = validate_item(item)
        errors.extend(item_result.errors)
        warnings.extend(item_result.warnings)
        
        # Check for duplicate IDs
        if item.id in item_ids:
            errors.append(f"Duplicate item ID: {item.id}")
        item_ids.add(item.id)
        
        # Check if item can fit in any bin
        if item.weight > problem.bin_capacity:
            errors.append(
                f"Item {item.id} weight ({item.weight}) exceeds bin capacity ({problem.bin_capacity})"
            )
    
    # Check total capacity
    if problem.items and problem.num_bins > 0:
        total_weight = sum(item.weight for item in problem.items)
        total_capacity = problem.bin_capacity * problem.num_bins
        
        if total_weight > total_capacity:
            warnings.append(
                f"Total item weight ({total_weight:.2f}) exceeds total capacity "
                f"({total_capacity:.2f}). Problem may be infeasible."
            )
        
        avg_weight_per_bin = total_weight / problem.num_bins
        if avg_weight_per_bin > problem.bin_capacity * 0.95:
            warnings.append(
                f"Average load per bin ({avg_weight_per_bin:.2f}) is very close to "
                f"capacity ({problem.bin_capacity}). Feasible solutions may be limited."
            )
    
    result = ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )
    
    if raise_on_error and not result.valid:
        raise ValidationError("Problem validation failed", errors)
    
    return result


def validate_solution(solution: Solution, problem: Problem, 
                     raise_on_error: bool = True) -> ValidationResult:
    """
    Validate a solution against its problem.
    
    Checks:
    - All items are assigned exactly once
    - Capacity constraints are satisfied
    - Solution uses correct number of bins
    
    Parameters
    ----------
    solution : Solution
        Solution to validate
    problem : Problem
        Original problem instance
    raise_on_error : bool
        If True, raise ValidationError on invalid solution
        
    Returns
    -------
    ValidationResult
        Validation result
        
    Raises
    ------
    ValidationError
        If validation fails and raise_on_error is True
    """
    errors = []
    warnings = []
    
    if not solution.bins:
        errors.append("Solution has no bins")
        result = ValidationResult(valid=False, errors=errors, warnings=warnings)
        if raise_on_error:
            raise ValidationError("Solution validation failed", errors)
        return result
    
    # Check number of bins
    if len(solution.bins) != problem.num_bins:
        warnings.append(
            f"Solution uses {len(solution.bins)} bins, expected {problem.num_bins}"
        )
    
    # Track assigned items
    assigned_items: Set[str] = set()
    problem_item_ids = {item.id for item in problem.items}
    
    # Validate each bin
    for bin_idx, bin_obj in enumerate(solution.bins):
        # Check capacity
        total_weight = sum(item.weight for item in bin_obj.items)
        
        if total_weight > problem.bin_capacity:
            errors.append(
                f"Bin {bin_idx + 1}: Weight ({total_weight:.2f}) exceeds "
                f"capacity ({problem.bin_capacity})"
            )
        
        # Check item assignments
        for item in bin_obj.items:
            if item.id in assigned_items:
                errors.append(f"Item {item.id} is assigned to multiple bins")
            assigned_items.add(item.id)
            
            if item.id not in problem_item_ids:
                errors.append(f"Item {item.id} not in original problem")
    
    # Check all items are assigned
    unassigned = problem_item_ids - assigned_items
    if unassigned:
        errors.append(f"Items not assigned: {', '.join(sorted(unassigned))}")
    
    extra = assigned_items - problem_item_ids
    if extra:
        errors.append(f"Unknown items in solution: {', '.join(sorted(extra))}")
    
    # Calculate solution quality warnings
    if not errors:
        bin_values = [sum(item.value for item in bin_obj.items) for bin_obj in solution.bins]
        value_range = max(bin_values) - min(bin_values)
        
        if value_range == 0:
            pass  # Perfect balance
        elif value_range < 0.01 * sum(bin_values):
            pass  # Nearly perfect
        else:
            avg_value = sum(bin_values) / len(bin_values)
            if max(bin_values) > 1.5 * avg_value:
                warnings.append("Solution has significant value imbalance")
    
    result = ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )
    
    if raise_on_error and not result.valid:
        raise ValidationError("Solution validation failed", errors)
    
    return result


def validate_algorithm_params(params: dict, param_spec: dict) -> ValidationResult:
    """
    Validate algorithm parameters against specification.
    
    Parameters
    ----------
    params : dict
        Parameters to validate
    param_spec : dict
        Parameter specification with types and ranges
        
    Returns
    -------
    ValidationResult
        Validation result
    """
    errors = []
    warnings = []
    
    for param_name, spec in param_spec.items():
        if param_name not in params:
            if spec.get('required', False):
                errors.append(f"Required parameter missing: {param_name}")
            continue
        
        value = params[param_name]
        
        # Type check
        expected_type = spec.get('type')
        if expected_type and not isinstance(value, expected_type):
            errors.append(
                f"Parameter {param_name}: expected {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
            continue
        
        # Range check
        min_val = spec.get('min')
        max_val = spec.get('max')
        
        if min_val is not None and value < min_val:
            errors.append(f"Parameter {param_name}: {value} < minimum {min_val}")
        
        if max_val is not None and value > max_val:
            errors.append(f"Parameter {param_name}: {value} > maximum {max_val}")
        
        # Allowed values
        allowed = spec.get('allowed')
        if allowed is not None and value not in allowed:
            errors.append(
                f"Parameter {param_name}: {value} not in allowed values {allowed}"
            )
    
    # Check for unknown parameters
    known_params = set(param_spec.keys())
    provided_params = set(params.keys())
    unknown = provided_params - known_params
    
    if unknown:
        warnings.append(f"Unknown parameters will be ignored: {', '.join(unknown)}")
    
    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )
