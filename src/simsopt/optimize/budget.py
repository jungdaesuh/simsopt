import time
import functools

class BudgetExceededError(Exception):
    """Raised when an optimization process exceeds its allocated budget."""
    pass

class BudgetConstraint:
    """
    Enforces matched-budget constraints for optimization runs.
    Supported modes: 'wall_clock' (seconds), 'op_count'.
    """
    def __init__(self, limit, mode='wall_clock'):
        self.limit = float(limit)
        self.mode = mode
        self.start_time = None
        self.op_count = 0
        self.active = False
        
    def start(self):
        """Starts tracking the budget."""
        self.start_time = time.perf_counter()
        self.op_count = 0
        self.active = True
        
    def check(self):
        """
        Checks if the budget has been exceeded.
        Raises BudgetExceededError if it has.
        """
        if not self.active:
            return
            
        if self.mode == 'wall_clock':
            elapsed = time.perf_counter() - self.start_time
            if elapsed > self.limit:
                raise BudgetExceededError(f"Wall-clock budget exceeded: {elapsed:.2f}s > {self.limit}s")
        elif self.mode == 'op_count':
            if self.op_count > self.limit:
                raise BudgetExceededError(f"Op-count budget exceeded: {self.op_count} > {self.limit}")

    def increment_ops(self, n=1):
        """Increments the tracked operation count."""
        self.op_count += n
        self.check()

def enforce_budget(budget_constraint):
    """Decorator to wrap optimization objective functions for budget enforcement."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            budget_constraint.check()
            result = func(*args, **kwargs)
            if budget_constraint.mode == 'op_count':
                budget_constraint.increment_ops()
            return result
        return wrapper
    return decorator
