"""Assert every headline number in the .tex matches the result artifacts.
Guards against transcription drift between results/*.json and the manuscript.

Key-path notes vs. the original brief (all paths verified against real JSON):
  - eff["comparison"]["param_reduction_x"]  : confirmed present, value 15.6
  - tr["zero_shot"]["distilled_mean"]        : confirmed present, value 0.7656
                                               (round(*100,1)==76.6)
  - rob["student"]["mCA_overall"]            : confirmed present, value 0.7574
                                               (round(*100,1)==75.7)
  - rob["mallanet"]["mCA_overall"]           : confirmed present, value 0.3866
                                               (round(*100,1)==38.7)
  - tr["zero_shot"]["supervised_mean"]       : confirmed present, value 0.6268
                                               (round(*100,1)==62.7)
  - tr["adaptation_distilled_seed0"]["full_finetune"]["acc"] : value 0.978
  - es["floor_all_models"]                   : confirmed int 11
  - "distilled mean 36.6" in required dict   : refers to error count (36.6±2.1
    errors) in the ablation section, not digit_transfer; verified in .tex only.
"""
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
tex = "\n".join(p.read_text() for p in (repo_root / "paper" / "sections").glob("*.tex"))

eff = json.load(open(repo_root / "results" / "efficiency_metrics.json"))
rob = json.load(open(repo_root / "results" / "robustness.json"))
tr  = json.load(open(repo_root / "results" / "digit_transfer.json"))
es  = json.load(open(repo_root / "results" / "error_structure.json"))

# --- artifact cross-checks (fail loud if an artifact value changed) ---
assert eff["comparison"]["param_reduction_x"] == 15.6, \
    f"param_reduction_x changed: {eff['comparison']['param_reduction_x']}"
assert eff["comparison"]["mac_reduction_x"] == 14.8, \
    f"mac_reduction_x changed: {eff['comparison']['mac_reduction_x']}"
assert eff["comparison"]["speedup_b1_x"] == 9.5, \
    f"speedup_b1_x changed: {eff['comparison']['speedup_b1_x']}"
assert eff["models"][0]["cpu_latency_ms_b1_mean"] == 8.28, \
    f"cpu_latency changed: {eff['models'][0]['cpu_latency_ms_b1_mean']}"

assert round(tr["zero_shot"]["distilled_mean"] * 100, 1) == 76.6, \
    f"distilled_mean*100 = {round(tr['zero_shot']['distilled_mean']*100,1)} != 76.6"
assert round(tr["zero_shot"]["supervised_mean"] * 100, 1) == 62.7, \
    f"supervised_mean*100 = {round(tr['zero_shot']['supervised_mean']*100,1)} != 62.7"
assert round(tr["adaptation_distilled_seed0"]["full_finetune"]["acc"] * 100, 1) == 97.8, \
    f"full_finetune acc*100 = {round(tr['adaptation_distilled_seed0']['full_finetune']['acc']*100,1)} != 97.8"

assert round(rob["student"]["mCA_overall"] * 100, 1) == 75.7, \
    f"student mCA_overall*100 = {round(rob['student']['mCA_overall']*100,1)} != 75.7"
assert round(rob["mallanet"]["mCA_overall"] * 100, 1) == 38.7, \
    f"mallanet mCA_overall*100 = {round(rob['mallanet']['mCA_overall']*100,1)} != 38.7"

assert es["floor_all_models"] == 11, \
    f"floor_all_models changed: {es['floor_all_models']}"

# --- .tex presence checks ---
required = {
    "param ratio 15.6x":       "15.6" in tex,
    "mac ratio 14.8x":         "14.8" in tex,
    "speedup 9.5x":            "9.5" in tex,
    "cpu latency 8.28":        "8.28" in tex,
    "parity p=0.345":          "0.345" in tex,
    "ablation distilled mean 36.6 errors":  "36.6" in tex,
    "error floor 11":          str(es["floor_all_models"]) in tex,
    "zero-shot 76.6%":         "76.6" in tex,
    "supervised 62.7%":        "62.7" in tex,
    "fine-tune 97.8%":         "97.8" in tex,
    "mCA student 75.7%":       "75.7" in tex,
    "mCA mallanet 38.7%":      "38.7" in tex,
}

fails = [k for k, ok in required.items() if not ok]
if fails:
    print("AUDIT FAILED -- missing/incorrect in .tex:", fails)
    sys.exit(1)

print("AUDIT PASSED -- all headline numbers present and consistent")
