"""Create a synthetic cohort (same schema as the SQL output) for testing the pipeline without MIMIC access."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmxai.data.synthetic import make_synthetic_cohort  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--patients", type=int, default=1200)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    df = make_synthetic_cohort(a.out, a.patients, seed=a.seed)
    print(f"{len(df)} synthetic stays, {df['subject_id'].nunique()} patients, prevalence {df['label'].mean():.1%}")
