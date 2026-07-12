import json
from dataclasses import dataclass, asdict
from typing import List
import os

@dataclass
class DefenseSuccessRow:
    attack: str
    undefended: float
    adv_training: float
    input_smoothing: float
    chen_query_blinding: float
    midas_edge_naive: float
    midas_edge_adaptive: float

def load_defense_success_json(filepath: str) -> List[DefenseSuccessRow]:
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [DefenseSuccessRow(**row) for row in data]

def save_defense_success_json(rows: List[DefenseSuccessRow], filepath: str):
    data = [asdict(row) for row in rows]
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def generate_report(results: dict, template_filepath: str, output_filepath: str):
    """
    Given an existing template (e.g., from report.rs output), 
    update the `midas_edge_naive` and `midas_edge_adaptive` columns 
    with the computed PyTorch evaluation results, and write them back.
    `results` is a dict mapping attack_name -> {"naive": float, "adaptive": float}.
    """
    rows = load_defense_success_json(template_filepath)
    if not rows:
        raise RuntimeError(f"Template JSON {template_filepath} not found or empty.")
    
    for row in rows:
        if row.attack in results:
            row.midas_edge_naive = results[row.attack]["naive"]
            row.midas_edge_adaptive = results[row.attack]["adaptive"]
            
    save_defense_success_json(rows, output_filepath)
