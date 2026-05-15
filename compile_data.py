import pandas as pd, numpy as np, re
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from scipy.interpolate import interp1d


def pct_T_to_absorbance(y_pct: np.ndarray) -> np.ndarray:
    """Convert %-transmittance to absorbance.

    Values above 100 (baseline noise) are kept (yield small negative A);
    only the lower bound is clipped to avoid log10(0).
    """
    T = y_pct.astype(np.float32) / 100.0
    T = np.clip(T, 1e-4, None)
    return (-np.log10(T)).astype(np.float32)


def load_flopp(directory: str, source_label: str) -> list[dict]:
    """FLOPP / FLOPP-e: two-column CSVs (wavenumber, %T), no header."""
    records = []
    for f in sorted(p for p in Path(directory).iterdir() if p.suffix.lower() == ".csv"):
        # Parse polymer from the first "word" of the filename.
        # Examples: "ABS 10. Brown LEGO Fragment.CSV" -> "ABS", "Nylon-007.csv" -> "Nylon"
        polymer_raw = re.split(r"[_\-\s.]", f.stem)[0]

        df = pd.read_csv(f, header=None, names=["wn", "y"])
        # FLOPP / FLOPP-e are stored as %T (range ~0-100), not absorbance.
        absorbance = pct_T_to_absorbance(df["y"].to_numpy(dtype=np.float32))

        records.append({
            "spectrum_id":       f"{source_label}_{f.stem}",
            "source":            source_label,
            "sample_id":         f.stem,
            "polymer_class_raw": polymer_raw,
            "wn":                df["wn"].to_numpy(dtype=np.float32),
            "intensity":         absorbance,
            "intensity_type":    "absorbance",
            "resolution_cm":     4.0,
            "instrument_mode":   "ATR",
        })
    return records


def load_villegas_c4(c4_root: str) -> list[dict]:
    """Villegas FTIR-PLASTIC-c4: per-polymer folders of single-sample CSVs.

    Each CSV has a ~12-line metadata header (TITLE SAMPLE NAME, NPOINTS, ...)
    followed by two columns: wavenumber (cm-1), %T.
    """
    records = []
    for polymer_dir in sorted(Path(c4_root).iterdir()):
        if not polymer_dir.is_dir():
            continue
        # Directory names look like "HDPE_c4", "LDPE_c4", ... strip the cN suffix.
        polymer_raw = re.sub(r"_c\d+$", "", polymer_dir.name)

        for f in sorted(polymer_dir.glob("*.csv")):
            df = _read_villegas_csv(f)
            if df is None:
                continue
            absorbance = pct_T_to_absorbance(df["y"].to_numpy(dtype=np.float32))

            records.append({
                "spectrum_id":       f"VC_c4_{f.stem}",
                "source":            "Villegas-c4",
                "sample_id":         f.stem,
                "polymer_class_raw": polymer_raw,
                "wn":                df["wn"].to_numpy(dtype=np.float32),
                "intensity":         absorbance,
                "intensity_type":    "absorbance",
                "resolution_cm":     4.0,
                "instrument_mode":   "ATR",
            })
    return records


_NUMERIC_LINE = re.compile(r"^\s*[-+]?\d")


def _read_villegas_csv(path: Path) -> pd.DataFrame | None:
    """Skip the Villegas metadata header and parse the (wn, %T) data block.

    The header length is nominally 12 lines but we detect the first numeric
    line to stay robust against minor format variations.
    """
    with path.open() as fh:
        skip = 0
        for line in fh:
            if _NUMERIC_LINE.match(line):
                break
            skip += 1
        else:
            return None
    return pd.read_csv(path, header=None, names=["wn", "y"], skiprows=skip)


def load_openspecy(directory: str) -> list[dict]:
    """OpenSpecy FTIR library.

    Two CSVs are shipped together:
      - OpenSpecy_FTIR_library.csv: long-format (Wavelength, Intensity,
        SampleName, group); intensities are already min-max normalized to [0,1].
      - OpenSpecy_FTIR_library_metadata.csv: one row per SampleName with
        SpectrumIdentity (polymer label), InstrumentMode (ATR/transmission/
        DRIFTS/reflection - heterogeneous), SpectrumType, etc.
    """
    root = Path(directory)
    long_df = pd.read_csv(root / "OpenSpecy_FTIR_library.csv")
    meta = pd.read_csv(root / "OpenSpecy_FTIR_library_metadata.csv")

    # All entries here are SpectrumType == "FTIR" by construction of the file,
    # but assert it so we notice if the upstream layout ever changes.
    if "SpectrumType" in meta.columns:
        meta = meta[meta["SpectrumType"].astype(str).str.upper() == "FTIR"]

    meta["SampleName"] = meta["SampleName"].astype(int)
    meta_by_sample = meta.set_index("SampleName")

    records = []
    for sample_name, grp in long_df.groupby("SampleName"):
        sid = int(sample_name)
        if sid not in meta_by_sample.index:
            continue
        m = meta_by_sample.loc[sid]
        if isinstance(m, pd.DataFrame):  # duplicate metadata rows -> take first
            m = m.iloc[0]

        grp = grp.sort_values("Wavelength")
        polymer_raw = str(m.get("SpectrumIdentity", "UNKNOWN")).strip() or "UNKNOWN"
        mode = m.get("InstrumentMode", None)
        mode = None if (pd.isna(mode) or str(mode).strip() == "") else str(mode).strip()

        # SpectralResolution is a free-text field (e.g. "4/cm", "8 cm-1"); parse
        # the first number if possible, else leave None.
        res = m.get("SpectralResolution", None)
        res_val = None
        if isinstance(res, str):
            mnum = re.search(r"(\d+(?:\.\d+)?)", res)
            if mnum:
                res_val = float(mnum.group(1))

        records.append({
            "spectrum_id":       f"OS_{sid}",
            "source":            "OpenSpecy",
            "sample_id":         str(sid),
            "polymer_class_raw": polymer_raw,
            "wn":                grp["Wavelength"].to_numpy(dtype=np.float32),
            "intensity":         grp["Intensity"].to_numpy(dtype=np.float32),
            # Already min-max normalized to [0,1] per spectrum in the upstream file.
            "intensity_type":    "normalized",
            "resolution_cm":     res_val,
            "instrument_mode":   mode,
        })
    return records


flopp   = load_flopp("data/raw/FLOPP/",   "FLOPP")
flopp_e = load_flopp("data/raw/FLOPP-e/", "FLOPP-e")
vc_c4   = load_villegas_c4("data/raw/Villegas-FTIR-Plastics/")
os_recs = load_openspecy("data/raw/OpenSpecy/")
print(f"FLOPP    loaded: {len(flopp)} files")
print(f"FLOPP-e  loaded: {len(flopp_e)} files")
print(f"VC-c4    loaded: {len(vc_c4)} files")
print(f"OpenSpecy loaded: {len(os_recs)} spectra")


CANONICAL_LO, CANONICAL_HI, CANONICAL_N = 700.0, 3996.0, 882
canonical_wn = np.linspace(CANONICAL_LO, CANONICAL_HI, CANONICAL_N).astype(np.float32)

# Big-6 whitelist for the POC. Extend later to {"PA","ABS",...} as the
# scope grows. The generic FLOPP/FLOPP-e label "PE" (no HDPE/LDPE
# distinction) is intentionally excluded — revisit if PE recall is weak.
TARGET_CLASSES = {"HDPE", "LDPE", "PET", "PP", "PS", "PVC"}


def resample(wn_native, y_native):
    order = np.argsort(wn_native)
    wn, y = wn_native[order], y_native[order]
    wn, idx = np.unique(wn, return_index=True); y = y[idx]
    if wn.min() > CANONICAL_LO + 100 or wn.max() < CANONICAL_HI - 100:
        return None
    f = interp1d(wn, y, kind="linear", bounds_error=False,
                 fill_value=(float(np.median(y[:5])), float(np.median(y[-5:]))))
    return f(canonical_wn).astype(np.float32)


all_records = flopp + flopp_e + vc_c4# + os_recs

rows = []
for r in all_records:
    if r["polymer_class_raw"] not in TARGET_CLASSES:
        continue
    y_canon = resample(r["wn"], r["intensity"])
    if y_canon is None:
        continue
    rows.append({
        "spectrum_id":       r["spectrum_id"],
        "source":            r["source"],
        "sample_id":         r["sample_id"],
        "polymer_class_raw": r["polymer_class_raw"],
        "intensity_type":    r["intensity_type"],
        "instrument_mode":   r.get("instrument_mode"),
        "resolution_cm":     r.get("resolution_cm"),
        "intensity":         y_canon.tolist(),  # length 882
    })

df = pd.DataFrame(rows)
df.to_parquet("data/processed/all_spectra.parquet",
              engine="pyarrow", compression="zstd", index=False)
print(f"Wrote {len(df)} spectra to all_spectra.parquet")
print(df.groupby(["source", "polymer_class_raw"]).size().unstack(fill_value=0))