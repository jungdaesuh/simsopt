import json
from pathlib import Path

class SealedOracle:
    """
    Sealed Oracle for blind evaluation of discovered optimizers against baselines.
    """
    def __init__(self, baselines_dir="benchmarks/gold_suite/baselines"):
        self.baselines_dir = Path(baselines_dir)
        self.desc_baseline = self._load_baseline("desc_alm.json")
        self.simsopt_baseline = self._load_baseline("simsopt_alm_2025.json")

    def _load_baseline(self, filename):
        path = self.baselines_dir / filename
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)["results"]
        return {}

    def evaluate(self, candidate_results):
        """
        Evaluates a candidate's results against the locked baselines.
        Returns a dict indicating 'Victory' or 'Defeat' for each category and budget.
        """
        evaluation = {}
        victory = True
        
        for category, budgets in candidate_results.items():
            evaluation[category] = {}
            for budget, candidate_metrics in budgets.items():
                desc_metrics = self.desc_baseline.get(category, {}).get(budget, {})
                simsopt_metrics = self.simsopt_baseline.get(category, {}).get(budget, {})
                
                # Assuming smaller is better for all metrics for simplicity in this oracle
                metric_name = list(candidate_metrics.keys())[0]
                cand_val = candidate_metrics[metric_name]
                desc_val = desc_metrics.get(metric_name, float('inf'))
                simsopt_val = simsopt_metrics.get(metric_name, float('inf'))
                
                # Victory requires beating both incumbents by at least 1%
                beat_desc = cand_val < desc_val * 0.99
                beat_simsopt = cand_val < simsopt_val * 0.99
                
                eval_str = "Victory" if beat_desc and beat_simsopt else "Defeat"
                evaluation[category][budget] = eval_str
                if eval_str == "Defeat":
                    victory = False
                    
        evaluation["overall"] = "Victory" if victory else "Defeat"
        return evaluation
