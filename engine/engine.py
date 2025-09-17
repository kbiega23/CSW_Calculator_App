# engine/engine.py
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math, re, difflib
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LOOKUP_CSV = DATA_DIR / "savings_lookup.csv"

HOURS_BUCKETS = [2080, 2912, 8760]
HOTEL_OCCUPANCY_BUCKET = 100
PTHP_HDD_SPLIT = 7999

# ---------- Input / Output dataclasses ----------
@dataclass
class Inputs:
    state: str
    city: str
    hdd: float
    cdd: float
    building_type: str                     # Office | School | Hotel | Hospital | Multi-family
    school_subtype: Optional[str] = None   # Primary | Secondary (School only)
    area_sf: float = 0.0
    floors: Optional[int] = None
    annual_hours: Optional[float] = None   # Office only
    occupancy_rate_pct: Optional[float] = None  # Hotel only (not used; bucket fixed at 100)
    existing_window: str = "Single pane"   # Single pane | Double pane
    hvac_label: str = "Other"
    heating_fuel_label: str = "Natural Gas"  # Natural Gas | Electric | None
    cooling_installed: bool = True
    mf_infiltration_include: Optional[bool] = True
    elec_rate_per_kwh: float = 0.12
    gas_rate_per_therm: float = 1.00
    csw_installed_area_sf: Optional[float] = None
    csw_panes: str = "Double"              # Single | Double

@dataclass
class EngineResult:
    per_sf: Dict[str, float]
    totals: Dict[str, float]
    eui: Dict[str, Optional[float]]
    debug: Dict[str, Any]

# ---------- CSV loading / normalization ----------
CANONICAL_COLUMNS = {
    "LookKey": {"lookkey", "look_key", "key", "lookupkey"},
    "ElecHeat_kWh_SF": {"elecheat_kwh_sf", "elecheatkwhsf", "elecheat"},
    "Cool_kWh_SF": {"cool_kwh_sf", "coolkwhsf", "cool"},
    "GasHeat_therm_SF": {"gasheat_therm_sf", "gasheatthermsf", "gasheat"},
    "Cool_adjust": {"cool_adjust", "cooladjust", "cooling_adjust", "cooladj"},
    "Infil_Heat_Factor": {"infil_heat_factor", "infilheatfactor"},
    "Infil_Cool_Factor": {"infil_cool_factor", "infilcoolfactor"},
    "Base_EUI": {"base_eui", "baseeui"},
    "CSW_EUI": {"csw_eui", "csweui"},
}
NUMERIC_COLS = [
    "ElecHeat_kWh_SF","Cool_kWh_SF","GasHeat_therm_SF",
    "Cool_adjust","Infil_Heat_Factor","Infil_Cool_Factor",
    "Base_EUI","CSW_EUI",
]

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.strip().lower())

def _normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str,str], List[str]]:
    orig = list(df.columns)
    norm_map: Dict[str, List[str]] = {}
    for c in orig:
        norm_map.setdefault(_norm(c), []).append(c)
    rename: Dict[str,str] = {}
    present = set()
    for canonical, aliases in CANONICAL_COLUMNS.items():
        for alias in (set(aliases) | {_norm(canonical)}):
            if alias in norm_map:
                rename[norm_map[alias][0]] = canonical
                present.add(canonical)
                break
    df = df.rename(columns=rename)
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    return df, rename, missing

@lru_cache(maxsize=1)
def load_lookup() -> Dict[str, Any]:
    if not LOOKUP_CSV.exists():
        raise FileNotFoundError(f"Missing {LOOKUP_CSV}.")
    df = pd.read_csv(LOOKUP_CSV, dtype=str, encoding="utf-8-sig").fillna("")
    df, renamed, missing = _normalize_columns(df)
    if "LookKey" in df.columns:
        df["LookKey"] = df["LookKey"].astype(str).str.strip()

    for col in NUMERIC_COLS:
        if col not in df.columns:
            df[col] = ""
        nums = pd.to_numeric(df[col].replace("", pd.NA), errors="coerce")
        if col in {"Cool_adjust","Infil_Heat_Factor","Infil_Cool_Factor"}:
            nums = nums.fillna(1.0)
        df[col+"__num"] = nums

    audit = {
        "path": str(LOOKUP_CSV),
        "rows": int(df.shape[0]),
        "columns_present": list(df.columns),
        "normalized_from": renamed,
        "missing_canonical_columns": missing,
        "sample_keys": df["LookKey"].head(20).tolist() if "LookKey" in df.columns else [],
    }
    return {"df": df, "audit": audit}

# ---------- helpers for key building ----------
def _base(existing_window: str) -> str:
    return "Single" if "single" in existing_window.lower() else "Double"

def _csw(panes: str) -> str:
    return "Single" if panes.strip().lower().startswith("single") else "Double"

def _fuel(label: str) -> str:
    s = label.strip().lower()
    if "gas" in s: return "Gas"
    if "elec" in s: return "Elec"
    return "None"

def _hvac_code(bt: str, hvac_label: str, mf_size: Optional[str]=None) -> str:
    s = hvac_label.strip().lower(); bt = bt.strip().lower()
    if bt == "office":
        if "built-up vav" in s: return "VAV"
        if "electric reheat" in s: return "PVAV_Elec"
        if "hydronic reheat" in s: return "PVAV_Gas"
        return "Other"
    if bt == "school":
        if "vav" in s: return "VAV"
        if "fan coil" in s: return "FCU"
        return "Other"
    if bt == "hotel":
        if "pthp" in s: return "PTHP"
        if "ptac" in s: return "PTAC"
        if "fan coil" in s: return "FCU"
        return "Other"
    if bt == "hospital":
        if "vav" in s: return "VAV"
        return "Other"
    if bt == "multi-family":
        return "PTAC" if (mf_size == "Low") else "FCU"
    return "Other"

def _derive_office_size(area_sf: float, hvac_code: str) -> str:
    return "Large" if (area_sf > 30000 and hvac_code == "VAV") else "Mid"

def _derive_hotel_size(hvac_code: str) -> str:
    return "Small" if hvac_code in {"PTAC","PTHP"} else "Large"

def _derive_mf_size(floors: Optional[int]) -> str:
    if floors is None: return "Low"
    return "Low" if floors < 4 else "Mid"

def _closest_buckets(hours: float) -> Tuple[int,int,float]:
    if hours <= HOURS_BUCKETS[0]: return HOURS_BUCKETS[0], HOURS_BUCKETS[0], 0.0
    if hours >= HOURS_BUCKETS[-1]: return HOURS_BUCKETS[-1], HOURS_BUCKETS[-1], 0.0
    for i in range(len(HOURS_BUCKETS)-1):
        lo, hi = HOURS_BUCKETS[i], HOURS_BUCKETS[i+1]
        if lo <= hours <= hi:
            t = (hours - lo) / (hi - lo) if hi>lo else 0.0
            return lo, hi, t
    return HOURS_BUCKETS[1], HOURS_BUCKETS[1], 0.0

# ---------- lookup helpers ----------
def _row(df: pd.DataFrame, key: str) -> Optional[pd.Series]:
    if "LookKey" not in df.columns: return None
    hit = df.loc[df["LookKey"] == key]
    return None if hit.empty else hit.iloc[0]

def _prefix(df: pd.DataFrame, p: str, n:int=30) -> List[str]:
    if "LookKey" not in df.columns: return []
    return df[df["LookKey"].str.startswith(p, na=False)]["LookKey"].head(n).tolist()

def _similar(df: pd.DataFrame, target: str, n:int=10) -> List[str]:
    if "LookKey" not in df.columns: return []
    keys = df["LookKey"].astype(str).tolist()
    return difflib.get_close_matches(target, keys, n=n, cutoff=0.5)

def _safe_num(x: Any, default: float=0.0) -> float:
    try: return float(x)
    except Exception: return default

def _avg(a: Any, b: Any) -> float:
    A, B = _safe_num(a, math.nan), _safe_num(b, math.nan)
    if math.isnan(A) and math.isnan(B): return math.nan
    if math.isnan(A): return B
    if math.isnan(B): return A
    return 0.5*(A+B)

def _btag(bt: str, school_sub: Optional[str], mf_size: Optional[str]) -> str:
    x = bt.lower()
    if x=="office": return "Office"
    if x=="school": return "PS" if (school_sub or "").lower().startswith("pri") else "SS"
    if x=="hotel":  return "Hotel"
    if x=="hospital": return "Hosp"
    if x=="multi-family": return f"{'Low' if (mf_size or 'Low')=='Low' else 'Mid'}MF"
    return ""

def _inventory(df: pd.DataFrame, base: str, csw: str, tag: str) -> List[str]:
    return _prefix(df, f"{base}{csw}{tag}", 100)

def _rowdbg(r: pd.Series) -> Dict[str, Any]:
    return {
        "LookKey": r.get("LookKey",""),
        "ElecHeat_kWh_SF": _safe_num(r.get("ElecHeat_kWh_SF__num",0)),
        "Cool_kWh_SF": _safe_num(r.get("Cool_kWh_SF__num",0)),
        "GasHeat_therm_SF": _safe_num(r.get("GasHeat_therm_SF__num",0)),
        "Cool_adjust": _safe_num(r.get("Cool_adjust__num",1.0)),
        "Infil_Heat_Factor": _safe_num(r.get("Infil_Heat_Factor__num",1.0)),
        "Infil_Cool_Factor": _safe_num(r.get("Infil_Cool_Factor__num",1.0)),
        "Base_EUI": _safe_num(r.get("Base_EUI__num", math.nan), math.nan),
        "CSW_EUI": _safe_num(r.get("CSW_EUI__num", math.nan), math.nan),
    }

def _debug_payload(
    df: pd.DataFrame,
    audit: Dict[str, Any],
    attempted: List[str],
    bt: str,
    school_sub: Optional[str],
    mf_size: Optional[str],
    base: str,
    csw: str,
    prefix_hint: str,
) -> Dict[str, Any]:
    """Builds the debug dict when a LookKey miss occurs (and for UI panel)."""
    btag = _btag(bt, school_sub, mf_size)
    last_key = attempted[-1] if attempted else ""
    # For suggestions, drop trailing digits (e.g., Office hour buckets)
    prefix_for_suggest = re.sub(r"\d+$", "", prefix_hint)
    return {
        "attempted_keys": attempted,
        "matched_keys": {},  # none; this is a miss path
        "expected_building_prefix": f"{base}{csw}{btag}",
        "inventory_keys_for_building_prefix": _inventory(df, base, csw, btag)[:50],
        "prefix_used_for_suggestions": prefix_for_suggest,
        "prefix_candidates": _prefix(df, prefix_for_suggest, 50),
        "similar_keys": _similar(df, last_key, 20),
        "csv_audit": audit,
    }

# ---------- main compute ----------
def compute_savings(inp: Inputs) -> EngineResult:
    loaded = load_lookup()
    df, audit = loaded["df"], loaded["audit"]

    base = _base(inp.existing_window)
    csw  = _csw(inp.csw_panes)
    fuel = _fuel(inp.heating_fuel_label)

    if inp.building_type == "Multi-family":
        mf_size = _derive_mf_size(inp.floors)
        hvac = _hvac_code("Multi-family", inp.hvac_label, mf_size)
        hvac = "PTAC" if mf_size == "Low" else "FCU"  # enforce
        if hvac == "PTAC" or fuel == "None":
            fuel = "Elec"
    else:
        mf_size = None
        hvac = _hvac_code(inp.building_type, inp.hvac_label)

    attempted: List[str] = []
    matched: Dict[str, Dict[str, Any]] = {}

    bt = inp.building_type
    if bt == "Office":
        size = _derive_office_size(inp.area_sf or 0, hvac)
        hours = inp.annual_hours or HOURS_BUCKETS[1]
        lo_b, hi_b, t = _closest_buckets(hours)
        def key(b:int) -> str: return f"{base}{csw}{size}Office{hvac}{fuel}{b}"
        k_lo, k_hi = key(lo_b), key(hi_b)
        attempted += [k_lo, k_hi]
        r_lo, r_hi = _row(df, k_lo), _row(df, k_hi)
        if r_lo is None and r_hi is not None: r_lo, lo_b, t = r_hi, hi_b, 0.0
        if r_hi is None and r_lo is not None: r_hi, hi_b, t = r_lo, lo_b, 0.0
        if r_lo is None or r_hi is None:
            raise ValueError(
                "no LookKey match.",
                _debug_payload(df, audit, attempted, bt, inp.school_subtype, mf_size, base, csw, f"{base}{csw}{size}Office{hvac}{fuel}")
            )
        matched[k_lo] = _rowdbg(r_lo); matched[k_hi] = _rowdbg(r_hi)
        e = (1-t)*_safe_num(r_lo["ElecHeat_kWh_SF__num"]) + t*_safe_num(r_hi["ElecHeat_kWh_SF__num"])
        c = (1-t)*_safe_num(r_lo["Cool_kWh_SF__num"])     + t*_safe_num(r_hi["Cool_kWh_SF__num"])
        g = (1-t)*_safe_num(r_lo["GasHeat_therm_SF__num"])+ t*_safe_num(r_hi["GasHeat_therm_SF__num"])
        if not inp.cooling_installed:
            adj = (_safe_num(r_lo["Cool_adjust__num"],1.0)+_safe_num(r_hi["Cool_adjust__num"],1.0))/2.0
            c *= adj
        per_sf = {"ElecHeat_kWh_SF": round(e,6), "Cool_kWh_SF": round(c,6), "GasHeat_therm_SF": round(g,6)}
        base_eui = _avg(r_lo["Base_EUI__num"], r_hi["Base_EUI__num"])
        csw_eui  = _avg(r_lo["CSW_EUI__num"],  r_hi["CSW_EUI__num"])

    elif bt == "School":
        sub = (inp.school_subtype or "Primary").lower()
        subcode = "PS" if sub.startswith("pri") else "SS"
        key = f"{base}{csw}{subcode}{hvac}{fuel}"
        attempted.append(key)
        r = _row(df, key)
        if r is None:
            raise ValueError(
                "no LookKey match.",
                _debug_payload(df, audit, attempted, bt, inp.school_subtype, None, base, csw, f"{base}{csw}{subcode}{hvac}{fuel}")
            )
        matched[key] = _rowdbg(r)
        e = _safe_num(r["ElecHeat_kWh_SF__num"]); c = _safe_num(r["Cool_kWh_SF__num"]); g = _safe_num(r["GasHeat_therm_SF__num"])
        if not inp.cooling_installed: c *= _safe_num(r["Cool_adjust__num"],1.0)
        per_sf = {"ElecHeat_kWh_SF": round(e,6), "Cool_kWh_SF": round(c,6), "GasHeat_therm_SF": round(g,6)}
        base_eui = _safe_num(r["Base_EUI__num"], math.nan); csw_eui = _safe_num(r["CSW_EUI__num"], math.nan)

    elif bt == "Hotel":
        size = _derive_hotel_size(hvac)
        band = "High" if (hvac == "PTHP" and (inp.hdd or 0) > PTHP_HDD_SPLIT) else ("Low" if hvac=="PTHP" else "")
        key = f"{base}{csw}{size}Hotel{hvac}{fuel}{band}{HOTEL_OCCUPANCY_BUCKET}"
        attempted.append(key)
        r = _row(df, key)
        if r is None:
            raise ValueError(
                "no LookKey match.",
                _debug_payload(df, audit, attempted, bt, None, None, base, csw, f"{base}{csw}{size}Hotel{hvac}{fuel}")
            )
        matched[key] = _rowdbg(r)
        e = _safe_num(r["ElecHeat_kWh_SF__num"]); c = _safe_num(r["Cool_kWh_SF__num"]); g = _safe_num(r["GasHeat_therm_SF__num"])
        if not inp.cooling_installed: c *= _safe_num(r["Cool_adjust__num"],1.0)
        per_sf = {"ElecHeat_kWh_SF": round(e,6), "Cool_kWh_SF": round(c,6), "GasHeat_therm_SF": round(g,6)}
        base_eui = _safe_num(r["Base_EUI__num"], math.nan); csw_eui = _safe_num(r["CSW_EUI__num"], math.nan)

    elif bt == "Hospital":
        key = f"{base}{csw}Hosp{hvac}{fuel}"
        attempted.append(key)
        r = _row(df, key)
        if r is None:
            raise ValueError(
                "no LookKey match.",
                _debug_payload(df, audit, attempted, bt, None, None, base, csw, f"{base}{csw}Hosp{hvac}{fuel}")
            )
        matched[key] = _rowdbg(r)
        e = _safe_num(r["ElecHeat_kWh_SF__num"]); c = _safe_num(r["Cool_kWh_SF__num"]); g = _safe_num(r["GasHeat_therm_SF__num"])
        if not inp.cooling_installed: c *= _safe_num(r["Cool_adjust__num"],1.0)
        per_sf = {"ElecHeat_kWh_SF": round(e,6), "Cool_kWh_SF": round(c,6), "GasHeat_therm_SF": round(g,6)}
        base_eui = _safe_num(r["Base_EUI__num"], math.nan); csw_eui = _safe_num(r["CSW_EUI__num"], math.nan)

    elif bt == "Multi-family":
        size = _derive_mf_size(inp.floors)
        hvac = "PTAC" if size == "Low" else "FCU"
        if hvac == "PTAC" or fuel == "None": fuel = "Elec"
        key = f"{base}{csw}{size}MF{hvac}{fuel}"
        attempted.append(key)
        r = _row(df, key)
        if r is None:
            raise ValueError(
                "no LookKey match.",
                _debug_payload(df, audit, attempted, bt, None, size, base, csw, f"{base}{csw}{size}MF{hvac}{fuel}")
            )
        matched[key] = _rowdbg(r)
        e = _safe_num(r["ElecHeat_kWh_SF__num"]); c = _safe_num(r["Cool_kWh_SF__num"]); g = _safe_num(r["GasHeat_therm_SF__num"])
        if inp.mf_infiltration_include is False:
            e *= _safe_num(r["Infil_Heat_Factor__num"],1.0)
            c *= _safe_num(r["Infil_Cool_Factor__num"],1.0)
        if not inp.cooling_installed: c *= _safe_num(r["Cool_adjust__num"],1.0)
        per_sf = {"ElecHeat_kWh_SF": round(e,6), "Cool_kWh_SF": round(c,6), "GasHeat_therm_SF": round(g,6)}
        base_eui = _safe_num(r["Base_EUI__num"], math.nan); csw_eui = _safe_num(r["CSW_EUI__num"], math.nan)

    else:
        raise ValueError(f"Unsupported building type: {bt}")

    area = inp.csw_installed_area_sf if (inp.csw_installed_area_sf and inp.csw_installed_area_sf>0) else inp.area_sf
    area = max(area or 0.0, 0.0)

    elec_heat_kwh = per_sf["ElecHeat_kWh_SF"] * area
    cool_kwh      = per_sf["Cool_kWh_SF"] * area
    gas_therms    = per_sf["GasHeat_therm_SF"] * area
    total_kwh     = elec_heat_kwh + cool_kwh
    cost_savings  = total_kwh * float(inp.elec_rate_per_kwh) + gas_therms * float(inp.gas_rate_per_therm)

    eui_savings = None
    if not math.isnan(base_eui) and not math.isnan(csw_eui):
        eui_savings = base_eui - csw_eui

    # Success-path debug (also useful)
    btag = _btag(bt, inp.school_subtype, mf_size)
    last_key = attempted[-1] if attempted else ""
    prefix_for_suggest = re.sub(r"\d+$","", last_key)
    debug = {
        "attempted_keys": attempted,
        "matched_keys": matched,
        "expected_building_prefix": f"{base}{csw}{btag}",
        "inventory_keys_for_building_prefix": _inventory(df, base, csw, btag)[:50],
        "prefix_used_for_suggestions": prefix_for_suggest,
        "prefix_candidates": _prefix(df, prefix_for_suggest, 50),
        "similar_keys": _similar(df, last_key, 20),
        "csv_audit": audit,
    }

    return EngineResult(
        per_sf=per_sf,
        totals={
            "elec_kwh": round(elec_heat_kwh,3),
            "cool_kwh": round(cool_kwh,3),
            "gas_therms": round(gas_therms,3),
            "total_kwh": round(total_kwh,3),
            "total_therms": round(gas_therms,3),
            "cost_savings": round(cost_savings,2),
            "area_used_sf": area,
        },
        eui={
            "eui_base_kbtusf": None if math.isnan(base_eui) else round(base_eui,3),
            "eui_csw_kbtusf": None if math.isnan(csw_eui) else round(csw_eui,3),
            "eui_savings_kbtusf": None if eui_savings is None else round(eui_savings,3),
        },
        debug=debug,
    )
