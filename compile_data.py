import pandas as pd, numpy as np, re
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from scipy.interpolate import interp1d

def load_flopp(directory: str, source_label: str) -> list[dict]:
    records = []
    for f in sorted(p for p in Path(directory).iterdir() if p.suffix.lower() == ".csv"):
        # Parse polymer from the first "word" of the filename.
        # Examples: "ABS_001.csv" → "ABS",  "Nylon-007.csv" → "Nylon"
        polymer_raw = re.split(r"[_\-\s\ \.]", f.stem)[0]

        # FLOPP CSVs are 2 columns, no header, scientific notation
        df = pd.read_csv(f, header=None, names=["wn", "absorbance"])

        records.append({
            "spectrum_id":       f"{source_label}_{f.stem}",
            "source":            source_label,
            "sample_id":         f.stem,
            "polymer_class_raw": polymer_raw,
            "wn":                df["wn"].to_numpy(dtype=np.float32),
            "intensity":         df["absorbance"].to_numpy(dtype=np.float32),
            "intensity_type":    "absorbance",
            "resolution_cm":     4.0,
        })
    return records

flopp   = load_flopp("data/raw/FLOPP/",   "FLOPP")
flopp_e = load_flopp("data/raw/FLOPP-e/", "FLOPP-e")
print(f"FLOPP loaded: {len(flopp)} files")
print(f"FLOPP-e loaded: {len(flopp_e)} files")

def load_villegas_c4(c4_root: str) -> list[dict]:
    records = []
    for polymer_dir in sorted(Path(c4_root).iterdir()):
        if not polymer_dir.is_dir():
            continue
        polymer_raw = polymer_dir.name          # e.g. "LDPE"
        for f in sorted(polymer_dir.glob("*.csv")):
            df = pd.read_csv(f, header=None, names=["wn", "T_pct"])

            # CRITICAL: convert %T → absorbance
            T = df["T_pct"].to_numpy(dtype=np.float32) / 100.0
            T = np.clip(T, 1e-4, 1.0)          # avoid log(0) for saturated dips
            absorbance = -np.log10(T)

            records.append({
                "spectrum_id":       f"VC_c4_{f.stem}",
                "source":            "Villegas-c4",
                "sample_id":         f.stem,    # e.g. "LDPE001"
                "polymer_class_raw": polymer_raw,
                "wn":                df["wn"].to_numpy(dtype=np.float32),
                "intensity":         absorbance,
                "intensity_type":    "absorbance",
                "resolution_cm":     4.0,
            })
    return records

vc_c4 = load_villegas_c4("data/raw/Villegas-FTIR-Plastics/")
print(f"VC-4 loaded: {len(vc_c4)} files")

CANONICAL_LO, CANONICAL_HI, CANONICAL_N = 700.0, 3996.0, 882
canonical_wn = np.linspace(CANONICAL_LO, CANONICAL_HI, CANONICAL_N).astype(np.float32)

def resample(wn_native, y_native):
    # sort ascending and dedup
    order = np.argsort(wn_native)
    wn, y = wn_native[order], y_native[order]
    wn, idx = np.unique(wn, return_index=True); y = y[idx]
    if wn.min() > CANONICAL_LO + 100 or wn.max() < CANONICAL_HI - 100:
        return None  # too little coverage — reject
    f = interp1d(wn, y, kind="linear", bounds_error=False,
                 fill_value=(float(np.median(y[:5])), float(np.median(y[-5:]))))
    return f(canonical_wn).astype(np.float32)

all_records = flopp + flopp_e + vc_c4 #+ records  # OpenSpecy records last

rows = []
for r in all_records:
    y_canon = resample(r["wn"], r["intensity"])
    if y_canon is None:
        continue
    rows.append({
        "spectrum_id":       r["spectrum_id"],
        "source":            r["source"],
        "sample_id":         r["sample_id"],
        "polymer_class_raw": r["polymer_class_raw"],
        "intensity_type":    r["intensity_type"],
        "intensity":         y_canon.tolist(),   # length 882
    })

df = pd.DataFrame(rows)
df.to_parquet("data/processed/all_spectra.parquet",
              engine="pyarrow", compression="zstd", index=False)
print(f"Wrote {len(df)} spectra to all_spectra.parquet")
print(df.groupby(["source", "polymer_class_raw"]).size().unstack(fill_value=0))
