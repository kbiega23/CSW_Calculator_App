# engine/engine.py
# CSW Savings Calculator (no Excel dependency)
# - Reads data/savings_lookup.csv once (cached)
# - Robust column normalization (fixes the 'LookKey' KeyError)
# - Exact Excel-style LookKey building + Office hours interpolation
# - Cooling_adjust & MF infiltration factors handled as specified
# - Returns a rich debug payload for the UI panel

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
import re
import difflib

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LOOKUP_CSV = DATA_DIR / "savings_lookup.csv"

HOURS_BUCKETS = [2080, 2912, 8760]       # Office interpolation buckets
HOTEL_OCCUPANCY_BUCKET = 100             # Fixed per spec
PTHP_HDD_SPLIT = 7999                    # PTHP band split (<=7999=Low, >=8000=High)


# ---------------------------
# Dataclasses (inputs / out)
# ---------------------------

@dataclass
class Inputs:
    # Step 1
    state: str
    city: str
    hdd: float
    cdd: float

    # Step 2
    building_type: str                 # Office | School | Hotel | Hospital | Multi-family
    school_subtype: Optional[str] = None  # Primary | Secondary (if School)

    # Step 3 (varies by type)
    area_sf: float = 0.0
    floors: Optional[int] = None
    annual_hours: Optional[float] = None          # Office only
    occupancy_rate_pct: Optional[float] = None    # Hotel only (fixed bucket=100 per spec)
    existing_window: str = "Single pane"          # Single pane | Double pane
    hvac_label: str = "Other"
    heating_fuel_label: str = "Natural Gas"       # Natural Gas | Electric | None
    cooling_installed: bool = True
    mf_infiltration_include: Optional[bool] = True  # MF only

    # Step 4
    elec_rate_per_kwh: float = 0.12
    gas_rate_per_therm: float = 1.00
    csw_installed_area_sf: Optional[float] = None
    csw_panes: str = "Double"  # Single | Double


@dataclass
class EngineResult:
    per_sf: Dict[str, float]
    totals: Dict[str, float]
    eui: Dict[str, Optional[float]]
    debug: Dict[str, Any]


# ---------------------------
# CSV loading / normalization
# ---------------------------

# Required (core) and optional columns
CANONICAL_COLUMNS = {
    # canonical_name: set of normalized aliases that should map to it
    "LookKey": {"lookkey", "look_key", "key", "lookupkey"},
    "ElecHeat_kWh_SF": {"elecheat_kwh_sf", "elecheatkwhsf", "elecheat", "elecheatkwhrsfs", "elecheatkwh_ft2"},
    "Cool_kWh_SF": {"cool_kwh_sf", "coolkwhsf", "cool"},
    "GasHeat_therm_SF": {"gasheat_therm_sf", "gasheatthermsf", "gasheat", "gasheattherm_ft2", "gasheat_therm_ft2"},
    "Cool_adjust": {"cool_adjust", "cooladjust", "cooling_adjust", "cooladj"},
    "Infil_Heat_Factor": {"infil_heat_factor", "infilheatfactor", "infil_heat", "infil_heat_adj"},
    "Infil_Cool_Factor": {"infil_cool_factor", "infilcoolfactor", "infil_cool", "infil_cool_adj"},
    # optional:
    "Base_EUI": {"base_eui", "euibase", "baseeui"},
    "CSW_EUI": {"csw_eui", "euicsw", "csweui"},
}

NUMERIC_COLS = [
    "ElecHeat_kWh_SF",
    "Cool_kWh_SF",
    "GasHeat_therm_SF",
    "Cool_adjust",
    "Infil_Heat_Factor",
    "Infil_Cool_Factor",
    "Base_EUI",
    "CSW_EUI",
]

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.strip().lower())

def _apply_column_normalization(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Renames columns to canonical names using case/space-insensitive matching.
    Returns: (df_renamed, mapping_original_to_canonical, missing_canonicals)
    """
    original_cols = list(df.columns)
    norm_to_originals: Dict[str, List[str]] = {}
    for c in original_cols:
        norm_to_originals.setdefault(_norm(c), []).append(c)

    rename_map: Dict[str, str] = {}
    present_canonicals: set = set()

    # 1) If exact canonical exists (case/space-insensitive), prefer that
    for canonical, aliases in CANONICAL_COLUMNS.items():
        all_aliases = set(aliases) | {_norm(canonical)}
        chosen: Optional[str] = None
        for alias in all_aliases:
            if alias in norm_to_originals:
                # Pick the first original col name for this alias
                chosen = norm_to_originals[alias][0]
                break
        if chosen:
            rename_map[chosen] = canonical
            present_canonicals.add(canonical)

    # 2) Apply rename
    df = df.rename(columns=rename_map)

    # 3) Report missing canonicals
    missing = [c for c in CANONICAL_COLUMNS.keys() if c not in df.columns]

    return df, rename_map, missing

@lru_cache(maxsize=1)
def load_lookup() -> Dict[str, Any]:
    if not LOOKUP_CSV.exists():
        raise FileNotFoundError(
            f"Missing {LOOKUP_CSV}. Place savings_lookup.csv into the repo's /data/ folder."
        )
    # Read as strings to avoid dtype surprises; we'll coerce numerics later
    df = pd.read_csv(LOOKUP_CSV, dtype=str, encoding="utf-8-sig").fillna("")

    df, rename_map, missing_canonicals = _apply_column_normalization(df)

    # Strip whitespace around LookKey to avoid invisible mismatches
    if "LookKey" in df.columns:
        df["LookKey"] = df["LookKey"].astype(str).str.strip()

    # Create numeric views with graceful coercion and sensible defaults
    for col in NUMERIC_COLS:
        if col not in df.columns:
            df[col] = ""  # keep string col for auditing
        nums = pd.to_numeric(df[col].replace("", pd.NA), errors="coerce")
        if col in {"Cool_adjust", "Infil_Heat_Factor", "Infil_Cool_Factor"}:
            nums = nums.fillna(1.0)
        df[col + "__num"] = nums

    audit = {
        "path": str(LOOKUP_CSV),
        "rows": int(df.shape[0]),
        "columns_present": list(df.columns),
        "normalized_from": rename_map,          # original -> canonical (for headers we fixed)
        "missing_canonical_columns": missing_canonicals,
        "sample_keys": df["LookKey"].head(20).tolist() if "LookKey" in df.columns else [],
    }
    return {"df": df, "audit": audit}


# ---------------------------
# Key building helpers
# ---------------------------

def _base_token(existing_window: str) -> str:
    return "Single" if "single" in existing_window.lower() else "Double"

def _csw_token(csw_panes: str) -> str:
    return "Single" if csw_panes.strip().lower().startswith("single") else "Double"

def _fuel_code(label: str) -> str:
    s = label.strip().lower()
    if "gas" in s:
        return "Gas"
    if "elec" in s:
        return "Elec"
    return "None"

def _hvac_code(building_type: str, hvac_label: str, derived_size: Optional[str] = None) -> str:
    s = hvac_label.strip().lower()
    bt = building_type.strip().lower()

    if bt == "office":
        if "built-up vav" in s:
            return "VAV"
        if "electric reheat" in s:
            return "PVAV_Elec"
        if "hydronic reheat" in s:
            return "PVAV_Gas"
        return "Other"

    if bt == "school":
        if "vav" in s:
            return "VAV"
        if "fan coil" in s:
            return "FCU"
        return "Other"

    if bt == "hotel":
        if "pthp" in s:
            return "PTHP"
        if "ptac" in s:
            return "PTAC"
        if "fan coil" in s:
            return "FCU"
        return "Other"

    if bt == "hospital":
        if "vav" in s:
            return "VAV"
        return "Other"

    if bt == "multi-family":
        # MF size forces HVAC (per spec)
        if derived_size == "Low":
            return "PTAC"
        return "FCU"

    return "Other"

def _derive_office_size(area_sf: float, hvac_code: str) -> str:
    # Large if area > 30k and HVAC is Built-up VAV (code='VAV'); else Mid
    return "Large" if (area_sf > 30000 and hvac_code == "VAV") else "Mid"

def _derive_hotel_size(hvac_code: str) -> str:
    # Small if PTAC/PTHP, else Large
    return "Small" if hvac_code in {"PTAC", "PTHP"} else "Large"

def _derive_mf_size(floors: Optional[int]) -> str:
    if floors is None:
        return "Low"  # default conservative
    return "Low" if floors < 4 else "Mid"

def _closest_buckets(hours: float) -> Tuple[int, int, float]:
    """
    Returns (lo_bucket, hi_bucket, t) for linear interpolation where
    value = (1-t)*lo + t*hi
    """
    # Clamp if out of range
    if hours <= HOURS_BUCKETS[0]:
        return HOURS_BUCKETS[0], HOURS_BUCKETS[0], 0.0
    if hours >= HOURS_BUCKETS[-1]:
        return HOURS_BUCKETS[-1], HOURS_BUCKETS[-1], 0.0

    # Find neighbors
    for i in range(len(HOURS_BUCKETS) - 1):
        lo = HOURS_BUCKETS[i]
        hi = HOURS_BUCKETS[i + 1]
        if lo <= hours <= hi:
            span = hi - lo
            t = 0.0 if span == 0 else (hours - lo) / span
            return lo, hi, t
    # Fallback, shouldn't happen
    return HOURS_BUCKETS[1], HOURS_BUCKETS[1], 0.0


# ---------------------------
# Lookup & compute
# ---------------------------

def _select_row(df: pd.DataFrame, key: str) -> Optional[pd.Series]:
    if "LookKey" not in df.columns:
        return None
    hit = df.loc[df["LookKey"] == key]
    if hit.empty:
        return None
    return hit.iloc[0]

def _similar_keys(df: pd.DataFrame, target: str, limit: int = 10) -> List[str]:
    if "LookKey" not in df.columns:
        return []
    keys = df["LookKey"].astype(str).tolist()
    return difflib.get_close_matches(target, keys, n=limit, cutoff=0.5)

def _prefix_candidates(df: pd.DataFrame, prefix: str, limit: int = 30) -> List[str]:
    if "LookKey" not in df.columns:
        return []
    pref = prefix.strip()
    matches = df[df["LookKey"].str.startswith(pref, na=False)]["LookKey"].head(limit)
    return matches.tolist()

def _building_tag_for_prefix(bt: str, school_sub: Optional[str], mf_size: Optional[str]) -> str:
    bt_l = bt.lower()
    if bt_l == "office":
        return "Office"
    if bt_l == "school":
        return "PS" if (school_sub or "").lower().startswith("pri") else "SS"
    if bt_l == "hotel":
        return "Hotel"
    if bt_l == "hospital":
        return "Hosp"
    if bt_l == "multi-family":
        return f"{'Low' if (mf_size or 'Low')=='Low' else 'Mid'}MF"
    return ""

def _safe_num(s: Any, default: float = 0.0) -> float:
    try:
        return float(s)
    except Exception:
        return default

def _collect_prefix_inventory(df: pd.DataFrame, base: str, csw: str, tag: str) -> List[str]:
    pref = f"{base}{csw}{tag}"
    return _prefix_candidates(df, pref, limit=100)

def compute_savings(inp: Inputs) -> EngineResult:
    loaded = load_lookup()
    df: pd.DataFrame = loaded["df"]
    audit: Dict[str, Any] = loaded["audit"]

    # Build tokens
    base = _base_token(inp.existing_window)
    csw = _csw_token(inp.csw_panes)
    fuel = _fuel_code(inp.heating_fuel_label)

    # Derive size/HVAC codes
    if inp.building_type == "Multi-family":
        mf_size = _derive_mf_size(inp.floors)
        hvac = _hvac_code("Multi-family", inp.hvac_label, mf_size)
        # Force MF HVAC by size anyway
        hvac = "PTAC" if mf_size == "Low" else "FCU"
        # Force fuel to Elec for PTAC or when user picked None (per spec)
        if hvac == "PTAC" or fuel == "None":
            fuel = "Elec"
    else:
        mf_size = None
        hvac = _hvac_code(inp.building_type, inp.hvac_label)

    # Build LookKey(s)
    attempted_keys: List[str] = []
    matched_rows: Dict[str, Dict[str, Any]] = {}

    building = inp.building_type

    if building == "Office":
        size = _derive_office_size(inp.area_sf or 0, hvac)
        hours = inp.annual_hours or HOURS_BUCKETS[1]
        lo_b, hi_b, t = _closest_buckets(hours)

        def build_key(bucket: int) -> str:
            return f"{base}{csw}{size}Office{hvac}{fuel}{bucket}"

        k_lo = build_key(lo_b)
        k_hi = build_key(hi_b)
        attempted_keys.extend([k_lo, k_hi])

        row_lo = _select_row(df, k_lo)
        row_hi = _select_row(df, k_hi)

        # Fallback: if a bucket is missing, re-use whichever exists
        if row_lo is None and row_hi is not None:
            row_lo, lo_b = row_hi, hi_b
            t = 0.0
        if row_hi is None and row_lo is not None:
            row_hi, hi_b = row_lo, lo_b
            t = 0.0

        if row_lo is None or row_hi is None:
            # No match at all -> build debug and raise
            tag = _building_tag_for_prefix(building, None, None)
            prefix = f"{base}{csw}{size}Office{hvac}{fuel}"
            debug_payload = _build_debug(df, audit, attempted_keys, building, inp.school_subtype, mf_size,
                                         base, csw, tag, prefix)
            raise ValueError(f"Could not compute savings for the selected combination: no LookKey match.",
                             debug_payload)

        matched_rows[k_lo] = row_to_debug(row_lo)
        matched_rows[k_hi] = row_to_debug(row_hi)

        # Interpolate per-SF values
        e_kwh_sf = (1 - t) * _safe_num(row_lo["ElecHeat_kWh_SF__num"]) + t * _safe_num(row_hi["ElecHeat_kWh_SF__num"])
        c_kwh_sf = (1 - t) * _safe_num(row_lo["Cool_kWh_SF__num"]) + t * _safe_num(row_hi["Cool_kWh_SF__num"])
        g_thm_sf = (1 - t) * _safe_num(row_lo["GasHeat_therm_SF__num"]) + t * _safe_num(row_hi["GasHeat_therm_SF__num"])

        # Cooling installed? If NO, multiply by Cool_adjust
        if not inp.cooling_installed:
            # Use average Cool_adjust across the two rows (or 1.0 if missing)
            adj = ( _safe_num(row_lo["Cool_adjust__num"], 1.0) + _safe_num(row_hi["Cool_adjust__num"], 1.0) ) / 2.0
            c_kwh_sf *= adj

        per_sf = {
            "ElecHeat_kWh_SF": round(e_kwh_sf, 6),
            "Cool_kWh_SF": round(c_kwh_sf, 6),
            "GasHeat_therm_SF": round(g_thm_sf, 6),
        }

        # EUI values: average the two if present
        base_eui = _avg_ignore_nan(row_lo["Base_EUI__num"], row_hi["Base_EUI__num"])
        csw_eui  = _avg_ignore_nan(row_lo["CSW_EUI__num"], row_hi["CSW_EUI__num"])

    elif building == "School":
        sub = (inp.school_subtype or "Primary").strip().lower()
        subcode = "PS" if sub.startswith("pri") else "SS"
        key = f"{base}{csw}{subcode}{hvac}{fuel}"
        attempted_keys.append(key)
        row = _select_row(df, key)
        if row is None:
            tag = _building_tag_for_prefix(building, inp.school_subtype, None)
            prefix = f"{base}{csw}{tag}{hvac}{fuel}" if tag else f"{base}{csw}"
            debug_payload = _build_debug(df, audit, attempted_keys, building, inp.school_subtype, None,
                                         base, csw, tag, prefix)
            raise ValueError(f"Could not compute savings for the selected combination: no LookKey match.",
                             debug_payload)
        matched_rows[key] = row_to_debug(row)

        e_kwh_sf = _safe_num(row["ElecHeat_kWh_SF__num"])
        c_kwh_sf = _safe_num(row["Cool_kWh_SF__num"])
        g_thm_sf = _safe_num(row["GasHeat_therm_SF__num"])
        if not inp.cooling_installed:
            c_kwh_sf *= _safe_num(row["Cool_adjust__num"], 1.0)
        per_sf = {
            "ElecHeat_kWh_SF": round(e_kwh_sf, 6),
            "Cool_kWh_SF": round(c_kwh_sf, 6),
            "GasHeat_therm_SF": round(g_thm_sf, 6),
        }
        base_eui = _safe_num(row["Base_EUI__num"], math.nan)
        csw_eui  = _safe_num(row["CSW_EUI__num"], math.nan)

    elif building == "Hotel":
        size = _derive_hotel_size(hvac)
        pthp_band = ""
        if hvac == "PTHP":
            pthp_band = "High" if (inp.hdd or 0) > PTHP_HDD_SPLIT else "Low"
        # Occupancy bucket fixed at 100 per spec
        key = f"{base}{csw}{size}Hotel{hvac}{fuel}{pthp_band}{HOTEL_OCCUPANCY_BUCKET}"
        attempted_keys.append(key)
        row = _select_row(df, key)
        if row is None:
            tag = _building_tag_for_prefix(building, None, None)
            prefix = f"{base}{csw}{size}Hotel{hvac}{fuel}"
            debug_payload = _build_debug(df, audit, attempted_keys, building, None, None,
                                         base, csw, tag, prefix)
            raise ValueError(f"Could not compute savings for the selected combination: no LookKey match.",
                             debug_payload)
        matched_rows[key] = row_to_debug(row)

        e_kwh_sf = _safe_num(row["ElecHeat_kWh_SF__num"])
        c_kwh_sf = _safe_num(row["Cool_kWh_SF__num"])
        g_thm_sf = _safe_num(row["GasHeat_therm_SF__num"])
        if not inp.cooling_installed:
            c_kwh_sf *= _safe_num(row["Cool_adjust__num"], 1.0)
        per_sf = {
            "ElecHeat_kWh_SF": round(e_kwh_sf, 6),
            "Cool_kWh_SF": round(c_kwh_sf, 6),
            "GasHeat_therm_SF": round(g_thm_sf, 6),
        }
        base_eui = _safe_num(row["Base_EUI__num"], math.nan)
        csw_eui  = _safe_num(row["CSW_EUI__num"], math.nan)

    elif building == "Hospital":
        key = f"{base}{csw}Hosp{hvac}{fuel}"
        attempted_keys.append(key)
        row = _select_row(df, key)
        if row is None:
            tag = _building_tag_for_prefix(building, None, None)
            prefix = f"{base}{csw}{tag}{hvac}{fuel}"
            debug_payload = _build_debug(df, audit, attempted_keys, building, None, None,
                                         base, csw, tag, prefix)
            raise ValueError(f"Could not compute savings for the selected combination: no LookKey match.",
                             debug_payload)
        matched_rows[key] = row_to_debug(row)

        e_kwh_sf = _safe_num(row["ElecHeat_kWh_SF__num"])
        c_kwh_sf = _safe_num(row["Cool_kWh_SF__num"])
        g_thm_sf = _safe_num(row["GasHeat_therm_SF__num"])
        if not inp.cooling_installed:
            c_kwh_sf *= _safe_num(row["Cool_adjust__num"], 1.0)
        per_sf = {
            "ElecHeat_kWh_SF": round(e_kwh_sf, 6),
            "Cool_kWh_SF": round(c_kwh_sf, 6),
            "GasHeat_therm_SF": round(g_thm_sf, 6),
        }
        base_eui = _safe_num(row["Base_EUI__num"], math.nan)
        csw_eui  = _safe_num(row["CSW_EUI__num"], math.nan)

    elif building == "Multi-family":
        size = _derive_mf_size(inp.floors)
        hvac = "PTAC" if size == "Low" else "FCU"  # enforce again
        if hvac == "PTAC" or fuel == "None":
            fuel = "Elec"
        key = f"{base}{csw}{size}MF{hvac}{fuel}"
        attempted_keys.append(key)
        row = _select_row(df, key)
        if row is None:
            tag = _building_tag_for_prefix(building, None, size)
            prefix = f"{base}{csw}{tag}{hvac}{fuel}" if tag else f"{base}{csw}"
            debug_payload = _build_debug(df, audit, attempted_keys, building, None, size,
                                         base, csw, tag, prefix)
            raise ValueError(f"Could not compute savings for the selected combination: no LookKey match.",
                             debug_payload)
        matched_rows[key] = row_to_debug(row)

        e_kwh_sf = _safe_num(row["ElecHeat_kWh_SF__num"])
        c_kwh_sf = _safe_num(row["Cool_kWh_SF__num"])
        g_thm_sf = _safe_num(row["GasHeat_therm_SF__num"])

        # Apply infiltration factors ONLY when user EXCLUDES infiltration (per spec)
        if inp.mf_infiltration_include is False:
            e_kwh_sf *= _safe_num(row["Infil_Heat_Factor__num"], 1.0)
            c_kwh_sf *= _safe_num(row["Infil_Cool_Factor__num"], 1.0)

        if not inp.cooling_installed:
            c_kwh_sf *= _safe_num(row["Cool_adjust__num"], 1.0)

        per_sf = {
            "ElecHeat_kWh_SF": round(e_kwh_sf, 6),
            "Cool_kWh_SF": round(c_kwh_sf, 6),
            "GasHeat_therm_SF": round(g_thm_sf, 6),
        }
        base_eui = _safe_num(row["Base_EUI__num"], math.nan)
        csw_eui  = _safe_num(row["CSW_EUI__num"], math.nan)

    else:
        raise ValueError(f"Unsupported building type: {building}")

    # Area to use for totals
    area = inp.csw_installed_area_sf if (inp.csw_installed_area_sf and inp.csw_installed_area_sf > 0) else inp.area_sf
    area = max(area or 0.0, 0.0)

    elec_heat_kwh = per_sf["ElecHeat_kWh_SF"] * area
    cool_kwh      = per_sf["Cool_kWh_SF"] * area
    gas_therms    = per_sf["GasHeat_therm_SF"] * area

    total_kwh   = elec_heat_kwh + cool_kwh
    total_therm = gas_therms

    cost_savings = total_kwh * float(inp.elec_rate_per_kwh) + total_therm * float(inp.gas_rate_per_therm)

    # EUI savings if both present
    eui_savings = None
    if not math.isnan(base_eui) and not math.isnan(csw_eui):
        eui_savings = base_eui - csw_eui

    # Debug payload
    btag = _building_tag_for_prefix(building, inp.school_subtype, mf_size)
    # "Expected pattern" inventory for this building family
    inventory = _collect_prefix_inventory(df, base, csw, btag) if btag else []
    # If a key missed, provide prefix & similarity suggestions for the last attempted key
    last_key = attempted_keys[-1] if attempted_keys else ""
    prefix_for_suggest = re.sub(r"\d+$", "", last_key)  # drop trailing bucket numbers etc.
    prefix_suggestions = _prefix_candidates(df, prefix_for_suggest, limit=50)
    similar = _similar_keys(df, last_key, limit=20)

    debug = {
        "attempted_keys": attempted_keys,
        "matched_keys": matched_rows,
        "expected_building_prefix": f"{base}{csw}{btag}",
        "inventory_keys_for_building_prefix": inventory[:50],
        "prefix_used_for_suggestions": prefix_for_suggest,
        "prefix_candidates": prefix_suggestions[:50],
        "similar_keys": similar,
        "csv_audit": audit,
    }

    return EngineResult(
        per_sf=per_sf,
        totals={
            "elec_kwh": round(elec_heat_kwh, 3),
            "cool_kwh": round(cool_kwh, 3),
            "gas_therms": round(gas_therms, 3),
            "total_kwh": round(total_kwh, 3),
            "total_therms": round(total_therm, 3),
            "cost_savings": round(cost_savings, 2),
            "area_used_sf": area,
        },
        eui={
            "eui_base_kbtusf": None if math.isnan(base_eui) else round(base_eui, 3),
            "eui_csw_kbtusf": None if math.isnan(csw_eui) else round(csw_eui, 3),
            "eui_savings_kbtusf": None if eui_savings is None else round(eui_savings, 3),
        },
        debug=debug,
    )


# ---------------------------
# Debug helpers
# ---------------------------

def row_to_debug(row: pd.Series) -> Dict[str, Any]:
    return {
        "LookKey": row.get("LookKey", ""),
        "ElecHeat_kWh_SF": _safe_num(row.get("ElecHeat_kWh_SF__num", 0)),
        "Cool_kWh_SF": _safe_num(row.get("Cool_kWh_SF__num", 0)),
        "GasHeat_therm_SF": _safe_num(row.get("GasHeat_therm_SF__num", 0)),
        "Cool_adjust": _safe_num(row.get("Cool_adjust__num", 1.0)),
        "Infil_Heat_Factor": _safe_num(row.get("Infil_Heat_Factor__num", 1.0)),
        "Infil_Cool_Factor": _safe_num(row.get("Infil_Cool_Factor__num", 1.0)),
        "Base_EUI": _safe_num(row.get("Base_EUI__num", math.nan), math.nan),
        "CSW_EUI": _safe_num(row.get("CSW_EUI__num", math.nan), math.nan),
    }

def _avg_ignore_nan(a: Any, b: Any) -> float:
    a = _safe_num(a, math.nan)
    b = _safe_num(b, math.nan)
    if math.isnan(a) and math.isnan(b):
        return math.nan
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    return 0.5 * (a + b)
