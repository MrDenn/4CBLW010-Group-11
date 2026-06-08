"""Harmonize every raw spectral dataset into one canonical parquet.

Output: data/processed/all_spectra.parquet, one row per spectrum on the
shared 882-channel wavenumber grid. Each row carries provenance (source,
instrument_mode, resolution_cm, intensity_type) AND a membership tag so
downstream code can mine the corpus two ways:

  - by data source / instrument, for hardware-agnostic train/eval splits
    (see SOURCE_* config in src/config.py and the split modes in src/data.py);
  - by label_role: "known" (one of the six target polymers, used for the
    closed-set classifier) vs "open" (everything else, kept for the
    later open-set / out-of-distribution evaluation).

Backward compatibility: src/data.py:load_parquet filters to the six target
classes via `polymer_class_raw`, so the open rows added here are dropped
automatically from current training/eval. Nothing downstream needs to
change until the open-set work explicitly opts in via label_role == "open".

See research-notes/datasets_and_integration.md for the per-dataset format,
label location, and the normalization/drop decisions behind this script.
"""
import pandas as pd, numpy as np, re
from pathlib import Path
from collections import Counter
from scipy.interpolate import interp1d


RAW = Path("data/raw")


def pct_T_to_absorbance(y_pct: np.ndarray) -> np.ndarray:
    """Convert %-transmittance to absorbance.

    Values above 100 (baseline noise) are kept (yield small negative A);
    only the lower bound is clipped to avoid log10(0).
    """
    T = y_pct.astype(np.float32) / 100.0
    T = np.clip(T, 1e-4, None)
    return (-np.log10(T)).astype(np.float32)


# ---------------------------------------------------------------------------
# Label normalization: raw label -> (canonical_label, role, category)
# ---------------------------------------------------------------------------
# `polymer_class_raw` in the parquet keeps the canonical_label. For knowns
# that is the big-6 short form (so src/data.py's existing class filter keeps
# working); for opens it is a cleaned material token. `label_role` and
# `category` are the new tags that make open-set mining explicit.

TARGET_CLASSES = ("HDPE", "LDPE", "PET", "PP", "PS", "PVC")

# Exact (lowercased, whitespace-collapsed) raw label -> big-6 short form.
# Generic PE with no HDPE/LDPE distinction is deliberately NOT here (it is
# dropped, see _DROP) because we cannot assign it to a known sub-class and it
# is not a legitimate "unknown" either.
_BIG6_ALIASES = {
    "hdpe": "HDPE",
    "ldpe": "LDPE",
    "pet": "PET", "pete": "PET",
    "polyethylene terephthalate": "PET",
    "poly(ethylene terephthalate)": "PET",
    "polytehylene terephthalate": "PET",   # OpenSpecy typo
    "polyethylene terephtalate": "PET",    # OpenSpecy typo
    "pp": "PP",
    "polypropylene": "PP", "poly(propylene)": "PP",
    "polypropylene isotactic": "PP", "fibre polypropylene": "PP",
    "ps": "PS",
    "polystyrene": "PS", "poly(styrene)": "PS", "polystyrene expanded": "PS",
    "pvc": "PVC",
    "polyvinylchloride": "PVC", "polyvinyl chloride": "PVC",
    "poly(vinyl chloride)": "PVC", "poly(vinylchloride)": "PVC",
}

# Raw labels that are dropped entirely (neither a clean known nor a useful
# open-set negative):
#   - generic / ambiguous PE: a known plastic family we cannot resolve to
#     HDPE vs LDPE, so it would corrupt both the known classes and the open
#     set. (Chemically MODIFIED PE — wax, chlorinated, oxidized, foamed — is
#     kept as an open "other_plastic" because its spectrum is genuinely
#     distinct.) PE-family variants too close to a known sub-class (LLDPE,
#     MDPE, ULDPE, TPCLDPE) are dropped for the same reason.
#   - tentative ("... like") or biofouled ("... + fouling") marine IDs whose
#     label quality is too low to trust on either side.
_DROP = {
    "pe", "polyethylene", "poly(ethylene)",
    "lldpe", "mdpe", "uldpe", "tpcldpe",
    "poly(ethylene) like", "poly(propylene) like", "poly(styrene) like",
    "poly(ethylene) + fouling",
}


def _clean(raw: str) -> str:
    """Collapse whitespace and strip; preserve the original wording for opens."""
    return re.sub(r"\s+", " ", str(raw).strip())


# Keyword -> coarse category. First matching keyword wins, scanned top to
# bottom, so put the more specific buckets first (e.g. animal-fibre keywords
# before the generic "fibre" textile fallback). This keeps the long OpenSpecy
# tail (~200 mostly-singleton labels) maintainable without a giant dict.
_CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("mineral",          ("quartz", "feldspar", "biotite", "muscovite", "mica", "calcite")),
    ("rubber",           ("rubber", "epdm", "butadiene", "isoprene", "neoprene",
                          "polychloroprene", "nitrile", "elastomer", "silicone",
                          "windscreen wiper", "sealing ring", "silicone seal")),
    ("textile_natural",  ("wool", "fur ", "silk", "cashmere", "mohair", "angora",
                          "cotton", "linen", "flax", "hemp", "jute", "kapok",
                          "cocoanut", "camel hair", "animal hair", "animal fibre",
                          "alpaca", "merino", "yak", "down", "hair")),
    ("organic",          ("algae", "alginic", "chitin", "broodcomb", "honeycomb",
                          "nectar", "amber", "wood", "zein", "coal", "beech",
                          "pine", "mahagoni", "grass", "turf", "cellulose",
                          "cigarette filter", "poplar")),
    ("textile_synthetic", ("fibre", "fiber", "viscose", "aramid", "polyester")),
]


# Exact (lowercased) raw label -> (canonical open label, category). These are
# abbreviations / variants the keyword scan below would otherwise miscategorize.
_OPEN_ALIASES = {
    "pes":   ("Polyester", "textile_synthetic"),
    "r":     ("Rubber", "rubber"),
    "sr":    ("Silicone rubber", "rubber"),
    "ca":    ("Cellulose acetate", "organic"),
    "pa":    ("Polyamide", "other_plastic"),
    "pa6":   ("Polyamide", "other_plastic"),
    "pa66":  ("Polyamide", "other_plastic"),
    "pa11":  ("Polyamide", "other_plastic"),
    "pa12":  ("Polyamide", "other_plastic"),
    "pa69":  ("Polyamide", "other_plastic"),
    "pa612": ("Polyamide", "other_plastic"),
    "pea":   ("Poly(ester amide)", "other_plastic"),
    "eaa":   ("Ethylene acrylic acid", "other_plastic"),
    "pet g": ("PETG", "other_plastic"),
    "petg":  ("PETG", "other_plastic"),
}

# FLOPP-e fibre codes "C1".."C12" are unidentified fibres; collapse the lot to
# one bucket rather than keeping a dozen singleton labels.
_C_FIBRE = re.compile(r"^c\d+$")

_UNKNOWN_LABELS = {"unknown", "other plastic", "morphotype", "morphotype 1",
                   "morphotype 2", "fiber (unassigned)"}


def _categorize_open(label: str) -> str:
    """Best-effort coarse family for an open-set label (for stratified
    open-set analysis). Anything unrecognized falls back to "other_plastic"
    if it looks polymeric, else "unknown"."""
    low = label.lower()
    if low in _UNKNOWN_LABELS:
        return "unknown"
    for category, keywords in _CATEGORY_KEYWORDS:
        if any(k in low for k in keywords):
            return category
    # Looks like a synthetic polymer name -> other_plastic; else unknown.
    if re.search(r"poly|nylon|\bpa\b|\bpa\d|acryl|styrene|vinyl|amide|ester|"
                 r"urethan|carbonate|pmma|abs|ptfe|pvdf|pom|pbt|san|sbc|eva|"
                 r"peva|pla|\bpc\b|\bpu\b|copolymer|resin",
                 low):
        return "other_plastic"
    return "unknown"


def normalize_label(raw: str, source: str) -> tuple[str, str, str] | None:
    """Map a raw dataset label to (canonical_label, role, category).

    Returns None for labels in the drop set. `role` is "known" iff the label
    resolves to one of the six target polymers.
    """
    label = _clean(raw)
    low = label.lower()
    if not low or low in ("nan", "none"):
        return None
    if low in _DROP:
        return None
    if low in _BIG6_ALIASES:
        return _BIG6_ALIASES[low], "known", "big6_plastic"
    if low in _OPEN_ALIASES:
        canon, category = _OPEN_ALIASES[low]
        return canon, "open", category
    if _C_FIBRE.match(low):
        return "Fiber (unassigned)", "open", "unknown"
    return label, "open", _categorize_open(label)


# ---------------------------------------------------------------------------
# Loaders. Each returns a list of dicts with the NATIVE (pre-resample)
# spectrum plus provenance; `polymer_class_raw` holds the raw label, which
# normalize_label() resolves at the assembly stage.
# ---------------------------------------------------------------------------


def load_flopp(directory: Path, source_label: str) -> list[dict]:
    """FLOPP / FLOPP-e: two-column CSVs (wavenumber, %T), no header.

    Read from the original 'FLOPP and FLOPP-e' archive, which contains the
    full polymer range (big-6 + textiles, rubber, other plastics) rather than
    the pre-filtered big-6 subset of the older FLOPP/ and FLOPP-e/ dirs.
    """
    records = []
    for f in sorted(p for p in directory.iterdir() if p.suffix.lower() == ".csv"):
        # Polymer label = first "word" of the filename.
        # "ABS 10. Brown LEGO Fragment.CSV" -> "ABS"; "Nylon-007.csv" -> "Nylon".
        polymer_raw = re.split(r"[_\-\s.]", f.stem)[0]
        df = pd.read_csv(f, header=None, names=["wn", "y"])
        records.append({
            "spectrum_id":       f"{source_label}_{f.stem}",
            "source":            source_label,
            "sample_id":         f.stem,
            "polymer_class_raw": polymer_raw,
            "wn":                df["wn"].to_numpy(dtype=np.float32),
            "intensity":         pct_T_to_absorbance(df["y"].to_numpy(dtype=np.float32)),
            "intensity_type":    "absorbance",
            "resolution_cm":     4.0,
            "instrument_mode":   "ATR",
        })
    return records


def load_villegas_c4(c4_root: Path) -> list[dict]:
    """Villegas FTIR-PLASTIC-c4: per-polymer folders of single-sample CSVs.

    Each CSV has a ~12-line metadata header (TITLE SAMPLE NAME, NPOINTS, ...)
    followed by two columns: wavenumber (cm-1), %T.
    """
    records = []
    for polymer_dir in sorted(Path(c4_root).iterdir()):
        if not polymer_dir.is_dir():
            continue
        # Directory names look like "HDPE_c4", "LDPE_c4", ...; strip the cN suffix.
        polymer_raw = re.sub(r"_c\d+$", "", polymer_dir.name)
        for f in sorted(polymer_dir.glob("*.csv")):
            df = _read_villegas_csv(f)
            if df is None:
                continue
            records.append({
                "spectrum_id":       f"VC_c4_{f.stem}",
                "source":            "Villegas-c4",
                "sample_id":         f.stem,
                "polymer_class_raw": polymer_raw,
                "wn":                df["wn"].to_numpy(dtype=np.float32),
                "intensity":         pct_T_to_absorbance(df["y"].to_numpy(dtype=np.float32)),
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


def _read_two_col_csv(path: Path) -> pd.DataFrame | None:
    """Skip a text header to the first numeric line and parse (wn, %T).

    Unlike `_read_villegas_csv`, this reads only the first two columns and
    coerces them to numeric, so it tolerates the malformed multi-column
    exports in the Baskaran set (two files carry a stray shifted duplicate
    column: `wn, %T, , , <dup>`) without misaligning into all-NaN.
    """
    with path.open() as fh:
        skip = 0
        for line in fh:
            if _NUMERIC_LINE.match(line):
                break
            skip += 1
        else:
            return None
    df = pd.read_csv(path, header=None, usecols=[0, 1], names=["wn", "y"], skiprows=skip)
    df["wn"] = pd.to_numeric(df["wn"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["wn", "y"])
    return df if len(df) >= 50 else None


def load_openspecy(directory: Path) -> list[dict]:
    """OpenSpecy FTIR library, ATR-mode only (all polymers, knowns + opens).

    Two CSVs are shipped together:
      - OpenSpecy_FTIR_library.csv: long-format (Wavelength, Intensity,
        SampleName, group); intensities are already min-max normalized to [0,1].
      - OpenSpecy_FTIR_library_metadata.csv: one row per SampleName with
        SpectrumIdentity (polymer label), InstrumentMode, SpectrumType,
        SpectralResolution, ...

    Only the InstrumentMode == ATR* + SpectrumType == FTIR filter is applied
    here so the OpenSpecy contribution is comparable to the other ATR sources;
    the big-6-vs-open decision is deferred to normalize_label() so the open
    rows are retained for open-set work.
    """
    long_df = pd.read_csv(directory / "OpenSpecy_FTIR_library.csv")
    meta = pd.read_csv(directory / "OpenSpecy_FTIR_library_metadata.csv")

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
        if isinstance(m, pd.DataFrame):
            m = m.iloc[0]

        mode = m.get("InstrumentMode", None)
        mode_str = "" if (pd.isna(mode) or str(mode).strip() == "") else str(mode).strip()
        if not mode_str.upper().startswith("ATR"):
            continue

        grp = grp.sort_values("Wavelength")
        # SpectralResolution is free text (e.g. "4/cm", "8 cm-1"); parse the
        # first number if present, else leave None.
        res_val = None
        res = m.get("SpectralResolution", None)
        if isinstance(res, str):
            mnum = re.search(r"(\d+(?:\.\d+)?)", res)
            if mnum:
                res_val = float(mnum.group(1))

        records.append({
            "spectrum_id":       f"OS_{sid}",
            "source":            "OpenSpecy",
            "sample_id":         str(sid),
            "polymer_class_raw": str(m.get("SpectrumIdentity", "")),
            "wn":                grp["Wavelength"].to_numpy(dtype=np.float32),
            "intensity":         grp["Intensity"].to_numpy(dtype=np.float32),
            "intensity_type":    "normalized",
            "resolution_cm":     res_val,
            "instrument_mode":   mode_str,
        })
    return records


def load_poseidon(d4_path: Path) -> list[dict]:
    """Poseidon (Kedzierski) marine-plastic ATR library.

    The labelled resource is the single wide CSV D4_4_publication.csv:
      columns = [Nom, Interpretation, <wn_1>, <wn_2>, ...]
    one row per spectrum, absorbance values, wavenumber grid given by the
    column headers (descending). The Interpretation column is the material
    label. (The raw IR_Spectra/*.txt files are the unlabelled per-shot
    measurements; the labels live here, keyed by Nom.)
    """
    df = pd.read_csv(d4_path)
    df.columns = [c.strip() for c in df.columns]
    wn_cols = [c for c in df.columns if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", c)]
    wn = np.array([float(c) for c in wn_cols], dtype=np.float32)

    records = []
    for _, row in df.iterrows():
        nom = _clean(row.get("Nom", ""))
        label = row.get("Interpretation", "")
        y = row[wn_cols].to_numpy(dtype=np.float32)
        mask = ~np.isnan(y)
        if mask.sum() < 50:
            continue
        records.append({
            "spectrum_id":        f"Poseidon_{nom}",
            "source":             "Poseidon",
            "sample_id":          nom,
            "polymer_class_raw":  label,
            "physical_sample_id": f"Poseidon::{nom}",
            "wn":                 wn[mask],
            "intensity":          y[mask],
            "intensity_type":     "absorbance",
            "resolution_cm":      None,
            "instrument_mode":    "ATR",
        })
    return records


def load_cowger_atr(atr_root: Path, meta_csv: Path) -> list[dict]:
    """High-throughput (Cowger) ATR-FTIR.

    Each plate folder ATR/<plate>/ holds an export.txt: a text wide-matrix
    where row 0 is the wavenumber header (descending) and each subsequent row
    is [<well>#<rep>, <well>.<rep>, absorbance...]. Material labels come from
    joined_cell_metadata.csv keyed by (Plate, Cell=well). The .0/.1/.2 OPUS
    binary files are the same spectra and are NOT parsed (export.txt suffices).

    Plate 6 ships no export.txt (OPUS-binary only) and is skipped. The three
    shots per well share one physical_sample_id so replicates never straddle a
    split boundary.
    """
    meta = pd.read_csv(meta_csv, encoding="latin-1", on_bad_lines="skip")
    meta["__key"] = (meta["Plate"].astype(str).str.strip()
                     + "_" + meta["Cell"].astype(str).str.strip())
    material_by_key = dict(zip(meta["__key"], meta["Material"]))

    records = []
    for export in sorted(atr_root.glob("*/export.txt")):
        plate = export.parent.name
        df = pd.read_csv(export)
        # Columns: first two are id-ish (well#rep, well.rep), rest are wn.
        wn_cols = [c for c in df.columns if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", str(c))]
        wn = np.array([float(c) for c in wn_cols], dtype=np.float32)
        for _, row in df.iterrows():
            tag = str(row.iloc[0])                       # e.g. "A12#1"
            mwell = re.match(r"([A-H]\d+)", tag)
            if mwell is None:
                continue
            well = mwell.group(1)
            material = material_by_key.get(f"{plate}_{well}")
            if material is None or (isinstance(material, float) and np.isnan(material)):
                continue
            y = row[wn_cols].to_numpy(dtype=np.float32)
            mask = ~np.isnan(y)
            if mask.sum() < 50:
                continue
            records.append({
                "spectrum_id":        f"Cowger_p{plate}_{tag.replace('#', 'r')}",
                "source":             "Cowger",
                "sample_id":          f"{plate}_{tag}",
                "polymer_class_raw":  str(material),
                "physical_sample_id": f"Cowger::{plate}_{well}",
                "wn":                 wn[mask],
                "intensity":          y[mask],
                "intensity_type":     "absorbance",
                "resolution_cm":      None,
                "instrument_mode":    "ATR",
            })
    return records


def load_inhouse(directory: Path, source_label: str = "In-house") -> list[dict]:
    """In-house lab ATR-FTIR: JCAMP-style .txt (## header, then wn  %T).

    Physical-sample identity comes from the ``##TITLE=`` header: spectra whose
    TITLE matches exactly are replicate measurements of the same physical
    sample. physical_sample_id is keyed on (class, title) so an observed
    cross-class TITLE collision cannot merge two different materials.
    """
    records = []
    for f in sorted(Path(directory).glob("*.txt")):
        polymer_raw = re.split(r"[ _\-.]", f.stem)[0].upper()
        title = None
        y_units = "UNKNOWN"
        wn, y = [], []
        for line in f.read_text(errors="ignore").splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("##"):
                up = s.upper()
                if up.startswith("##TITLE="):
                    title = s.split("=", 1)[1].strip()
                elif up.startswith("##YUNITS="):
                    y_units = s.split("=", 1)[1].strip().upper()
                continue
            parts = s.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    wn.append(float(parts[0])); y.append(float(parts[1]))
                except ValueError:
                    continue
        if len(wn) < 50:
            continue
        wn = np.asarray(wn, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        intensity = pct_T_to_absorbance(y) if "T" in y_units else y
        title = title or f.stem
        records.append({
            "spectrum_id":        f"{source_label}_{f.stem}",
            "source":             source_label,
            "sample_id":          f.stem,
            "polymer_class_raw":  polymer_raw,
            "physical_sample_id": f"{source_label}::{polymer_raw}::{title}",
            "wn":                 wn,
            "intensity":          intensity,
            "intensity_type":     "absorbance",
            "resolution_cm":      4.0,
            "instrument_mode":    "ATR",
        })
    return records


def load_food_packaging(directory: Path, labels_csv: Path) -> list[dict]:
    """Food packaging (Baskaran & Sathiavelu 2020) ATR-FTIR of multilayer films.

    96 two-column CSVs (wavenumber, %T) with a short text header
    (XLabel/YLabel/FileType/DisplayDirection/PeakDirection), one per separated
    film layer. The files carry no material labels; labels.csv (shipped beside
    the data) maps each filename -> polymer using Table 1 of the paper:
      PP / PET / LDPE (PE family collapsed; m- metalized resins kept) as big-6
      knowns, Polyurethane as an open-set foreign. Rows whose `material` is
      blank (the SBR and EVA singletons, which resemble big-6 PS/PP and PE) are
      skipped entirely. Small n -> intended as eval / open-set material, not
      training. See research-notes/datasets_and_integration.md.
    """
    labels = pd.read_csv(labels_csv)
    material_by_file = {str(r["filename"]): str(r["material"]).strip()
                        for _, r in labels.iterrows()}
    records = []
    for f in sorted(directory.glob("*.csv")):
        if f.name == labels_csv.name:
            continue
        material = material_by_file.get(f.name, "")
        if not material or material.lower() == "nan":
            continue  # unlabelled or deliberately excluded (SBR / EVA)
        df = _read_two_col_csv(f)  # skip header, take first 2 cols (wn, %T)
        if df is None:
            continue
        records.append({
            "spectrum_id":        f"Baskaran_{f.stem}",
            "source":             "Baskaran",
            "sample_id":          f.stem,
            "polymer_class_raw":  material,
            "physical_sample_id": f"Baskaran::{f.stem}",
            "wn":                 df["wn"].to_numpy(dtype=np.float32),
            "intensity":          pct_T_to_absorbance(df["y"].to_numpy(dtype=np.float32)),
            "intensity_type":     "absorbance",
            "resolution_cm":      4.0,
            "instrument_mode":    "ATR",
        })
    return records


# ---------------------------------------------------------------------------
# Load every source
# ---------------------------------------------------------------------------

flopp    = load_flopp(RAW / "FLOPP and FLOPP-e" / "FLOPP .csv",   "FLOPP")
flopp_e  = load_flopp(RAW / "FLOPP and FLOPP-e" / "FLOPP-e .csv", "FLOPP-e")
vc_c4    = load_villegas_c4(RAW / "Villegas-FTIR-Plastics")
os_recs  = load_openspecy(RAW / "OpenSpecy")
poseidon = load_poseidon(RAW / "Poseidon_files_(Kedzierski)" /
                         "Poseidon_files_V0.1.1" / "Data" / "IR_References" /
                         "D4_4_publication.csv")
cowger   = load_cowger_atr(RAW / "High throughput (Cowger)" / "ATR",
                           RAW / "High throughput (Cowger)" / "joined_cell_metadata.csv")
inhouse  = load_inhouse(RAW / "Lab Combined")
food     = load_food_packaging(RAW / "Food packaging (Baskaran)",
                               RAW / "Food packaging (Baskaran)" / "labels.csv")

print(f"FLOPP     loaded: {len(flopp)} files")
print(f"FLOPP-e   loaded: {len(flopp_e)} files")
print(f"VC-c4     loaded: {len(vc_c4)} files")
print(f"OpenSpecy loaded: {len(os_recs)} spectra")
print(f"Poseidon  loaded: {len(poseidon)} spectra")
print(f"Cowger    loaded: {len(cowger)} spectra")
print(f"In-house  loaded: {len(inhouse)} spectra")
print(f"Baskaran  loaded: {len(food)} spectra")


# ---------------------------------------------------------------------------
# Resample onto the canonical grid
# ---------------------------------------------------------------------------

CANONICAL_LO, CANONICAL_HI, CANONICAL_N = 700.0, 3996.0, 882
canonical_wn = np.linspace(CANONICAL_LO, CANONICAL_HI, CANONICAL_N).astype(np.float32)


def resample(wn_native, y_native):
    order = np.argsort(wn_native)
    wn, y = wn_native[order], y_native[order]
    wn, idx = np.unique(wn, return_index=True); y = y[idx]
    if wn.min() > CANONICAL_LO + 100 or wn.max() < CANONICAL_HI - 100:
        return None
    f = interp1d(wn, y, kind="linear", bounds_error=False,
                 fill_value=(float(np.median(y[:5])), float(np.median(y[-5:]))))
    return f(canonical_wn).astype(np.float32)


# ---------------------------------------------------------------------------
# Assemble: normalize labels, resample, tag, write
# ---------------------------------------------------------------------------

all_records = flopp + flopp_e + vc_c4 + os_recs + poseidon + cowger + inhouse + food

rows = []
dropped = Counter()
unknown_labels = Counter()
for r in all_records:
    norm = normalize_label(r["polymer_class_raw"], r["source"])
    if norm is None:
        dropped[_clean(r["polymer_class_raw"]).lower()] += 1
        continue
    label, role, category = norm
    y_canon = resample(r["wn"], r["intensity"])
    if y_canon is None:
        continue
    if category == "unknown" and role == "open":
        unknown_labels[label] += 1
    rows.append({
        "spectrum_id":        r["spectrum_id"],
        "source":             r["source"],
        "sample_id":          r["sample_id"],
        "polymer_class_raw":  label,          # canonical: big-6 short form or cleaned material
        "label_role":         role,           # "known" | "open"
        "category":           category,       # coarse family for open-set analysis
        # Per-particle identity for leak-free, group-aware splitting. Loaders
        # that track replicates set this; others default to one sample/spectrum.
        "physical_sample_id": r.get("physical_sample_id") or f"{r['source']}::{r['sample_id']}",
        "intensity_type":     r["intensity_type"],
        "instrument_mode":    r.get("instrument_mode"),
        "resolution_cm":      r.get("resolution_cm"),
        "intensity":          y_canon.tolist(),  # length 882
    })

df = pd.DataFrame(rows)
df.to_parquet("data/processed/all_spectra.parquet",
              engine="pyarrow", compression="zstd", index=False)

print(f"\nWrote {len(df)} spectra to all_spectra.parquet "
      f"({(df['label_role'] == 'known').sum()} known, "
      f"{(df['label_role'] == 'open').sum()} open)")

print("\n--- known classes (source x class) ---")
known = df[df["label_role"] == "known"]
print(known.groupby(["source", "polymer_class_raw"]).size().unstack(fill_value=0))

print("\n--- open-set spectra (source x category) ---")
openset = df[df["label_role"] == "open"]
if not openset.empty:
    print(openset.groupby(["source", "category"]).size().unstack(fill_value=0))

if unknown_labels:
    print(f"\n--- labels routed to category 'unknown' ({sum(unknown_labels.values())} spectra) ---")
    for lbl, n in unknown_labels.most_common(20):
        print(f"  {n:4d}  {lbl}")

print(f"\n--- dropped raw labels ({sum(dropped.values())} spectra) ---")
for lbl, n in dropped.most_common(15):
    print(f"  {n:4d}  {lbl}")
