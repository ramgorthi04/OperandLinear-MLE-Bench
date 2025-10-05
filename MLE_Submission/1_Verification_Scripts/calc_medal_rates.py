#!/usr/bin/env python3
"""
Compute medal rates for MLE-Benchmark problems and summarize results.

Definition (medalled seed): a seed is counted as medalled if either
any_medal == True or medal_achieved == True in its competition_results.json.

Evaluation convention (required): every problem is evaluated as if it has
exactly 3 seeds. If fewer than 3 seed result files are present, the missing
seeds are assumed to be non-medal. If more than 3 seed result files are found,
only the first 3 in lexicographic order are counted.

Outputs:
- Per-problem totals: assumed total seeds (always 3), medalled seeds, medal rate.
- Summary mean medal rate across problems ± one standard error of the mean (SEM).
- Optional CSV/JSON exports.

By default, the script scans the parent directory of this script (expected to be
/Users/ram/Documents/MLE_Submission) and reports Overall + LITE + MEDIUM + HARD
subsets. You can filter to specific subsets or explicit problems via CLI flags.

Usage examples:
  # Run from anywhere; defaults to scanning the parent of this script
  python3 calc_medal_rates.py

  # Limit to specific subsets (comma-separated)
  python3 calc_medal_rates.py --subsets LITE,MEDIUM

  # Limit to specific problems (folder names, comma-separated)
  python3 calc_medal_rates.py --problems tweet-sentiment-extraction,leaf-classification

  # Export detailed rows to CSV and JSON
  python3 calc_medal_rates.py --csv out.csv --json out.json

Notes:
- Problems with zero detected seeds are included as 0/3 (medals/assumed seeds).
- SEM is undefined for n <= 1; displayed as n/a.
- You can switch medal flag logic via ---flag-mode: union (default), any_medal, or medal_achieved.
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Iterable, Set

# Constants
SKIP_DIRS = {"0_Environment_Setup", "1_Verification_Scripts"}
DEFAULT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "medal_rates.csv")

# Subset definitions: exact folder slugs
SUBSETS: Dict[str, Set[str]] = {
    # --- LITE SUBSET ---
    "LITE": {
        "aerial-cactus-identification",
        "aptos2019-blindness-detection",
        "denoising-dirty-documents",
        "detecting-insults-in-social-commentary",
        "dog-breed-identification",
        "dogs-vs-cats-redux-kernels-edition",
        "histopathologic-cancer-detection",
        "jigsaw-toxic-comment-classification-challenge",
        "leaf-classification",
        "mlsp-2013-birds",
        "new-york-city-taxi-fare-prediction",
        "nomad2018-predict-transparent-conductors",
        "plant-pathology-2020-fgvc7",
        "random-acts-of-pizza",
        "ranzcr-clip-catheter-line-classification",
        "siim-isic-melanoma-classification",
        "spooky-author-identification",
        "tabular-playground-series-dec-2021",
        "tabular-playground-series-may-2022",
        "text-normalization-challenge-english-language",
        "text-normalization-challenge-russian-language",
        "the-icml-2013-whale-challenge-right-whale-redux",
    },
    # --- MEDIUM SUBSET ---
    "MEDIUM": {
        "AI4Code",
        "alaska2-image-steganalysis",
        "billion-word-imputation",
        "cassava-leaf-disease-classification",
        "cdiscount-image-classification-challenge",
        "chaii-hindi-and-tamil-question-answering",
        "champs-scalar-coupling",
        "facebook-recruiting-iii-keyword-extraction",
        "freesound-audio-tagging-2019",
        "google-quest-challenge",
        "h-and-m-personalized-fashion-recommendations",
        "herbarium-2020-fgvc7",
        "herbarium-2021-fgvc8",
        "herbarium-2022-fgvc9",
        "hotel-id-2021-fgvc8",
        "hubmap-kidney-segmentation",
        "icecube-neutrinos-in-deep-ice",
        "imet-2020-fgvc7",
        "inaturalist-2019-fgvc6",
        "iwildcam-2020-fgvc7",
        "jigsaw-unintended-bias-in-toxicity-classification",
        "kuzushiji-recognition",
        "learning-agency-lab-automated-essay-scoring-2",
        "lmsys-chatbot-arena",
        "multi-modal-gesture-recognition",
        "osic-pulmonary-fibrosis-progression",
        "petfinder-pawpularity-score",
        "plant-pathology-2021-fgvc8",
        "seti-breakthrough-listen",
        "statoil-iceberg-classifier-challenge",
        "tensorflow-speech-recognition-challenge",
        "tensorflow2-question-answering",
        "tgs-salt-identification-challenge",
        "tweet-sentiment-extraction",
        "us-patent-phrase-to-phrase-matching",
        "uw-madison-gi-tract-image-segmentation",
        "ventilator-pressure-prediction",
        "whale-categorization-playground",
    },
    # --- HARD SUBSET ---
    "HARD": {
        "3d-object-detection-for-autonomous-vehicles",
        "bms-molecular-translation",
        "google-research-identify-contrails-reduce-global-warming",
        "hms-harmful-brain-activity-classification",
        "iwildcam-2019-fgvc6",
        "nfl-player-contact-detection",
        "predict-volcanic-eruptions-ingv-oe",
        "rsna-2022-cervical-spine-fracture-detection",
        "rsna-breast-cancer-detection",
        "rsna-miccai-brain-tumor-radiogenomic-classification",
        "siim-covid19-detection",
        "smartphone-decimeter-2022",
        "stanford-covid-vaccine",
        "vesuvius-challenge-ink-detection",
        "vinbigdata-chest-xray-abnormalities-detection",
    },
}

@dataclass
class ProblemStats:
    problem: str
    total_seeds: int
    medalled_seeds: int

    @property
    def medal_rate(self) -> float:
        if self.total_seeds <= 0:
            return float("nan")
        return self.medalled_seeds / self.total_seeds


def discover_problems(root: str) -> List[str]:
    problems: List[str] = []
    try:
        for entry in sorted(os.listdir(root)):
            if entry in SKIP_DIRS:
                continue
            p = os.path.join(root, entry)
            if os.path.isdir(p):
                problems.append(entry)
    except FileNotFoundError:
        pass
    return problems


def iter_results_jsons(problem_dir: str) -> Iterable[str]:
    for dirpath, dirnames, filenames in os.walk(problem_dir):
        # no special pruning needed here except skip env/scripts if nested
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn == "competition_results.json":
                yield os.path.join(dirpath, fn)


def seed_medalled(data: dict, flag_mode: str = "union") -> bool:
    any_medal = bool(data.get("any_medal") is True)
    medal_achieved = bool(data.get("medal_achieved") is True)
    if flag_mode == "any_medal":
        return any_medal
    if flag_mode == "medal_achieved":
        return medal_achieved
    # union
    return any_medal or medal_achieved


def compute_problem_stats(root: str, problems: Iterable[str], flag_mode: str = "union") -> Dict[str, ProblemStats]:
    out: Dict[str, ProblemStats] = {}
    ASSUMED_SEEDS = 3
    for prob in problems:
        prob_dir = os.path.join(root, prob)
        found_paths = []
        if os.path.isdir(prob_dir):
            found_paths = sorted(list(iter_results_jsons(prob_dir)))
        # Deterministically select at most 3 seeds
        selected = found_paths[:ASSUMED_SEEDS]
        medalled = 0
        for fpath in selected:
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                continue
            if seed_medalled(data, flag_mode=flag_mode):
                medalled += 1
        # Missing seeds are assumed non-medal; total is always ASSUMED_SEEDS
        out[prob] = ProblemStats(problem=prob, total_seeds=ASSUMED_SEEDS, medalled_seeds=medalled)
    return out


def mean_sem(values: List[float]) -> Tuple[float, Optional[float]]:
    vals = [v for v in values if not (math.isnan(v) or math.isinf(v))]
    n = len(vals)
    if n == 0:
        return float("nan"), None
    mean = sum(vals) / n
    if n <= 1:
        return mean, None
    # sample standard deviation
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    std = math.sqrt(var)
    sem = std / math.sqrt(n)
    return mean, sem


def format_mean_sem(mean: float, sem: Optional[float], decimals: int = 4) -> str:
    if math.isnan(mean):
        return "n/a"
    if sem is None:
        return f"{mean:.{decimals}f} ± n/a"
    return f"{mean:.{decimals}f} ± {sem:.{decimals}f}"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute medal rates (per problem and subset) for MLE-Benchmark results.")
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT, help="Root directory to scan (default: parent of this script)")
    parser.add_argument("--subsets", type=str, default=None, help="Comma-separated subset names to include (LITE,MEDIUM,HARD). Default: all.")
    parser.add_argument("--problems", type=str, default=None, help="Comma-separated explicit problem folder names to include (overrides subsets if provided).")
    parser.add_argument("--flag-mode", type=str, choices=["union", "any_medal", "medal_achieved"], default="union", help="Which flag(s) define a medalled seed (default: union)")
    parser.add_argument("--csv", type=str, default=None, help="Optional path to write detailed CSV output.")
    parser.add_argument("--json", type=str, default=None, help="Optional path to write detailed JSON output.")
    parser.add_argument("--quiet", action="store_true", help="Reduce per-problem output; only print summaries.")
    return parser.parse_args(argv)


def resolve_problem_selection(root: str, args: argparse.Namespace) -> Tuple[List[str], Dict[str, Set[str]]]:
    available = set(discover_problems(root))
    # Remove any skip dirs if discovered
    available -= SKIP_DIRS

    # Start with explicit problems if provided
    if args.problems:
        wanted = set([p.strip() for p in args.problems.split(",") if p.strip()])
        # Do not intersect with "available"; include even if folder is missing
        problems = sorted(list(wanted))
        subset_map: Dict[str, Set[str]] = {"CUSTOM": set(problems)}
        return problems, subset_map

    # Otherwise use subsets
    chosen_subsets = list(SUBSETS.keys())
    if args.subsets:
        chosen_subsets = [s.strip().upper() for s in args.subsets.split(",") if s.strip()]
        for s in chosen_subsets:
            if s not in SUBSETS:
                print(f"Warning: unknown subset '{s}'. Known: {', '.join(SUBSETS.keys())}", file=sys.stderr)
    # Include the full declared subset membership even if folders are missing on disk
    subset_map = {name: set(SUBSETS.get(name, set())) for name in chosen_subsets}

    # Overall is union of chosen subsets; if none chosen, it's union of all declared problems
    union = set().union(*subset_map.values()) if subset_map else set()
    if not union:
        # Fall back to all discovered or declared problems (prefer discovered if no subsets)
        union = available
    problems = sorted(list(union))
    return problems, subset_map


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    import csv
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            fh.write("")
        return
    # Use the union of all keys to accommodate both detail and summary rows
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    root = os.path.abspath(args.root)

    problems, subset_map = resolve_problem_selection(root, args)
    if not problems:
        print("No problems found to analyze.")
        return 0

    stats_by_problem = compute_problem_stats(root, problems, flag_mode=args.flag_mode)

    # Build detailed rows
    detailed_rows: List[Dict[str, object]] = []
    for prob in sorted(stats_by_problem.keys()):
        ps = stats_by_problem[prob]
        detailed_rows.append({
            "problem": ps.problem,
            "total_seeds": ps.total_seeds,
            "medalled_seeds": ps.medalled_seeds,
            "medal_rate": round(ps.medal_rate, 6) if not math.isnan(ps.medal_rate) else None,
        })

    # Print per-problem unless quiet
    if not args.quiet:
        print("Per-problem medal rates (assumed 3 seeds per problem):")
        for row in detailed_rows:
            rate_str = f"{row['medal_rate']:.4f}" if row["medal_rate"] is not None else "n/a"
            print(f"  - {row['problem']}: {row['medalled_seeds']}/3 (rate={rate_str})")
        print()

    # Compute and print summaries
    def summarize(label: str, probs: Iterable[str]) -> Dict[str, object]:
        chosen = [stats_by_problem[p] for p in sorted(probs) if p in stats_by_problem]
        rates = [ps.medal_rate for ps in chosen]
        mean, sem = mean_sem(rates)
        formatted = format_mean_sem(mean, sem)
        print(f"{label} summary: {formatted} (n={len(chosen)} problems)")
        return {
            "label": label,
            "n_problems": len(chosen),
            "mean_medal_rate": None if math.isnan(mean) else round(mean, 6),
            "sem": None if sem is None else round(sem, 6),
        }

    print("Summaries (mean medal rate ± SEM):")
    summary_rows: List[Dict[str, object]] = []

    # Overall (selected problems)
    summary_rows.append(summarize("OVERALL", stats_by_problem.keys()))

    # Each subset
    for subset_name, subset_problems in subset_map.items():
        summary_rows.append(summarize(subset_name, subset_problems))

    # Build CSV rows (details + summaries) and always write default CSV in script dir
    rows = []
    for row in detailed_rows:
        r = dict(row)
        r["label"] = "DETAIL"
        rows.append(r)
    for sr in summary_rows:
        rows.append({
            "label": sr["label"],
            "problem": None,
            "total_seeds": None,
            "medalled_seeds": None,
            "medal_rate": sr["mean_medal_rate"],
            "sem": sr["sem"],
            "n_problems": sr["n_problems"],
        })

    # Always write default CSV alongside this script
    write_csv(DEFAULT_CSV_PATH, rows)
    print(f"\nWrote CSV: {DEFAULT_CSV_PATH}")

    # Optionally also write user-specified CSV path
    if args.csv:
        write_csv(args.csv, rows)
        print(f"Wrote CSV: {args.csv}")

    if args.json:
        out = {
            "root": root,
            "flag_mode": args.flag_mode,
            "problems": detailed_rows,
            "summaries": summary_rows,
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"Wrote JSON: {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
