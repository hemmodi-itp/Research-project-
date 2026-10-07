"""Extract the leakage-aware cohort from local PhysioNet CSV releases (requires credentialed access).

python scripts/build_cohort.py --hosp .../mimiciv/3.1/hosp --ed .../mimic-iv-ed/2.2/ed \
    --cxr .../mimic-cxr-jpg/2.1.0 --out data/cohort.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmxai.data.cohort import build_cohort  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hosp", required=True, help="MIMIC-IV hosp module directory")
    p.add_argument("--ed", required=True, help="MIMIC-IV-ED ed directory")
    p.add_argument("--cxr", required=True, help="MIMIC-CXR-JPG root (with mimic-cxr-2.0.0-metadata.csv.gz)")
    p.add_argument("--sql", default=str(Path(__file__).resolve().parents[1] / "sql" / "cohort.sql"))
    p.add_argument("--out", default="data/cohort.csv")
    a = p.parse_args()
    df = build_cohort(a.sql, a.hosp, a.ed, a.cxr)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} ED stays, {df['subject_id'].nunique()} patients, "
          f"pneumonia prevalence {df['label'].mean():.1%} -> {a.out}")
