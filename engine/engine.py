# engine/engine.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------- Loaders ----------

def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    return df

def load_weather(data_dir: Optional[Path] = None) -> pd.DataFrame:
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    df = _read_csv(data_dir / "weather_information.csv")
    # normalize headers
    colmap = {}
    for c in df.columns:
        lc = c.lower()
        if "state" in lc and "hdd" not in lc and "cdd" not in lc: colmap[c] = "State"
        elif "city" in lc: colmap[c] = "Cities"
        elif "heating degree" in lc or "(hdd" in lc or lc.startswith("hdd"): colmap[c] = "Heating Degree Days (HDD)"
        elif "cooling degree" in lc or "(cdd" in lc or lc.startswith("cdd"): colmap[c] = "Cooling Degree Days (CDD)"
    df = df.rename(columns=colmap)
    return df[["State","Cities","Heating Degree Days (HDD)","Cooling Degree Days (CDD)"]]

def load_lists(data_dir: Optional[Path] = None) -> pd.DataFrame:
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    return _read_csv(data_dir / "lists.csv")

def load_savings_lookup(data_dir: Optional[Path] = None) -> pd.DataFrame:
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    df = _read_csv(data_dir / "savings_lookup.csv")
    # normalize key columns to the Excel headers
    rename = {
        "Look-up_Conc":"LookKey","LookKey":"LookKey",
        "Base":"Base","CSW":"CSW","Size":"Size","Building_Type":"Building_Type",
        "HVAC_Type":"HVAC_Type","Fuel":"Fuel","PTHP":"PTHP","Hours":"Hours",
        "Electric_savings_Heat_kWhperSF":"ElecHeat_kWh_SF",
        "electric_savings_Cooling_and_Aux_kWhperSF":"Cool_kWh_SF",
        "Gas_savings_Heat_thermsperSF":"GasHeat_therm_SF",
        "Base_EUI_kBtuperSFperyr":"Base_EUI","CSW_EUI_kBtuperSFperyr":"CSW_EUI",
        "Calculated_Savings_EUI":"Savings_EUI_frac",
        "Cool_adjust":"Cool_adjust",
        "infiltration_savings_reduction_factors_Heat":"Infil_Heat_Factor",
        "infiltration_reduction_factors_Cool":"Infil_Cool_Factor",
    }
    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
    for c in ["Hours","ElecHeat_kWh_SF","Cool_kWh_SF","GasHeat_therm_SF","Base_EUI","CSW_EUI",
              "Savings_EUI_frac","Cool_adjust","Infil_Heat_Factor","Infil_Cool_Factor"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ---------- Helpers (mirror Excel’s VLOOKUP keys) ----------

def hvac_label_to_code(lists_df: pd.DataFrame, building_type: str, hvac_label: str) -> str:
    bt = building_type.strip().lower()
    hvac_col = None
    for c in lists_df.columns:
        if c.strip().lower().startswith("hvac type") and bt in c.strip().lower():
            hvac_col = c; break
    if hvac_col is None:
        for c in lists_df.columns:
            if c.strip().lower().startswith("hvac type"): hvac_col = c; break
    if hvac_col is None: return hvac_label

    cols = lists_df.columns.tolist()
    idx = cols.index(hvac_col)
    code_col = None
    for j in range(idx+1, min(idx+4, len(cols))):
        name = cols[j].strip().lower()
        if name in ("building type","type","hvac code") or "code" in name:
            code_col = cols[j]; break
    if code_col is None and idx+1 < len(cols): code_col = cols[idx+1]

    mapping = dict(zip(lists_df[hvac_col].astype(str).str.strip(),
                       lists_df[code_col].astype(str).str.strip()))
    return mapping.get(hvac_label, mapping.get(hvac_label.strip(), hvac_label))

def _office_keys(base_glz:str, csw:str, size:str, hvac_code:str, fuel:str, hours:float) -> Tuple[str,str,float]:
    high = 8760 if hours > 2912 else 2912
    low  = 2912 if high == 8760 else 2080
    alpha = 0.0 if high==low else max(0.0, min(1.0, (hours-low)/(high-low)))
    mk = lambda h: f"{base_glz}{csw}{size}Office{hvac_code}{fuel}{int(h)}"
    return mk(low), mk(high), alpha

def _school_key(base_glz:str, csw:str, subcode:str, hvac_code:str, fuel:str) -> str:
    return f"{base_glz}{csw}{subcode}{hvac_code}{fuel}"

def _hospital_key(base_glz:str, csw:str, hvac_code:str, fuel:str) -> str:
    return f"{base_glz}{csw}Hosp{hvac_code}{fuel}"

def _hotel_key(base_glz:str, csw:str, size:str, hvac_code:str, fuel:str, pthp_band:str, occ_bucket:int) -> str:
    return f"{base_glz}{csw}{size}Hotel{hvac_code}{fuel}{pthp_band}{occ_bucket}"

def _mf_key(base_glz:str, csw:str, size_code:str, hvac_code:str, fuel:str) -> str:
    return f"{base_glz}{csw}{size_code}MF{hvac_code}{fuel}"

# ---------- API ----------

@dataclass
class ComputeInputs:
    building_type: str
    sub_building_type: Optional[str]
    area_total_sf: float
    floors: int
    annual_hours: float
    existing_window: str           # "Single pane" or "Double pane"
    csw_type: str                  # "Single" / "Double"
    hvac_label: str                # user-facing label
    heating_fuel: str              # "Natural Gas","Electric","None"
    cooling_installed: str         # "Yes"/"No"
    csw_installed_sf: float
    electric_rate: float
    gas_rate: float
    infiltration_included: Optional[str] = None   # MF only: "Included"/"Excluded"
    school_level: Optional[str] = None            # "Primary School"/"Secondary School"
    hotel_occupancy: Optional[float] = None       # not used by Excel (bucket fixed at 100)
    hdd: Optional[float] = None
    cdd: Optional[float] = None

@dataclass
class ComputeResult:
    elec_kwh_yr: float
    gas_therm_yr: float
    cost_savings_usd: float
    eui_base: Optional[float]
    eui_csw: Optional[float]
    eui_savings_kbtusf: Optional[float]
    details: Dict[str, float]

def compute_savings_exact(inp: ComputeInputs, savings_lookup: pd.DataFrame, lists_df: pd.DataFrame) -> ComputeResult:
    bt = inp.building_type.strip()
    base_glz = "Single" if "single" in inp.existing_window.lower() else "Double"
    csw = inp.csw_type.strip().title() if inp.csw_type else "Single"

    hvac_code = hvac_label_to_code(lists_df, bt, inp.hvac_label)

    fuel = inp.heating_fuel.title()
    if fuel.lower() == "none":
        fuel = "Electric"

    cool_adjust = 1.0

    # OFFICE
    if bt == "Office":
        is_large = (inp.area_total_sf > 30000) and (inp.hvac_label.strip().lower().startswith("built-up vav"))
        size = "Large" if is_large else "Mid"
        low_key, high_key, alpha = _office_keys(base_glz, csw, size, hvac_code, fuel, inp.annual_hours)
        lo = savings_lookup[savings_lookup["LookKey"] == low_key].iloc[:1]
        hi = savings_lookup[savings_lookup["LookKey"] == high_key].iloc[:1]
        if lo.empty or hi.empty:
            raise KeyError(f"Office lookup failed for keys {low_key} or {high_key}")
        s_lo, t_lo, u_lo = lo["ElecHeat_kWh_SF"].item(), lo["Cool_kWh_SF"].item(), lo["GasHeat_therm_SF"].item()
        s_hi, t_hi, u_hi = hi["ElecHeat_kWh_SF"].item(), hi["Cool_kWh_SF"].item(), hi["GasHeat_therm_SF"].item()
        if str(inp.cooling_installed).strip().lower() == "no" and "Cool_adjust" in hi:
            ca = hi["Cool_adjust"].item()
            cool_adjust = float(ca) if pd.notna(ca) else 1.0
        elec_heat_sf = s_lo + (s_hi - s_lo) * alpha
        cool_sf      = (t_lo + (t_hi - t_lo) * alpha) * cool_adjust
        gas_heat_sf  = u_lo + (u_hi - u_lo) * alpha
        eui_base = (lo["Base_EUI"].item() + (hi["Base_EUI"].item() - lo["Base_EUI"].item()) * alpha) if "Base_EUI" in lo and "Base_EUI" in hi else None
        eui_csw  = (lo["CSW_EUI"].item()  + (hi["CSW_EUI"].item()  - lo["CSW_EUI"].item())  * alpha) if "CSW_EUI" in lo and "CSW_EUI" in hi else None

    # SCHOOL
    elif bt == "School":
        subcode = "SS" if (inp.school_level and inp.school_level.lower().startswith("secondary")) else "PS"
        key = _school_key(base_glz, csw, subcode, hvac_code, fuel)
        row = savings_lookup[savings_lookup["LookKey"] == key].iloc[:1]
        if row.empty:
            raise KeyError(f"School lookup failed for key {key}")
        row = row.iloc[0]
        cool_adjust = float(row.get("Cool_adjust", 1.0)) if str(inp.cooling_installed).strip().lower() == "no" else 1.0
        elec_heat_sf = float(row["ElecHeat_kWh_SF"] or 0.0)
        cool_sf      = float(row["Cool_kWh_SF"] or 0.0) * cool_adjust
        gas_heat_sf  = float(row["GasHeat_therm_SF"] or 0.0)
        eui_base     = float(row.get("Base_EUI")) if "Base_EUI" in row else None
        eui_csw      = float(row.get("CSW_EUI")) if "CSW_EUI" in row else None

    # HOTEL
    elif bt == "Hotel":
        size = "Small" if hvac_code in ("PTAC","PTHP") else "Large"
        if hvac_code in ("PTAC","PTHP"): fuel = "Electric"
        pthp_band = "High" if (hvac_code == "PTHP" and (inp.hdd or 0) > 7999) else ("Low" if hvac_code=="PTHP" else "")
        occ_bucket = 100  # Excel workbook uses 100
        key = _hotel_key(base_glz, csw, size, hvac_code, fuel, pthp_band, occ_bucket)
        row = savings_lookup[savings_lookup["LookKey"] == key].iloc[:1]
        if row.empty:
            raise KeyError(f"Hotel lookup failed for key {key}")
        row = row.iloc[0]
        cool_adjust = float(row.get("Cool_adjust", 1.0)) if str(inp.cooling_installed).strip().lower() == "no" else 1.0
        elec_heat_sf = float(row["ElecHeat_kWh_SF"] or 0.0)
        cool_sf      = float(row["Cool_kWh_SF"] or 0.0) * cool_adjust
        gas_heat_sf  = float(row["GasHeat_therm_SF"] or 0.0)
        eui_base     = float(row.get("Base_EUI")) if "Base_EUI" in row else None
        eui_csw      = float(row.get("CSW_EUI")) if "CSW_EUI" in row else None

    # HOSPITAL
    elif bt == "Hospital":
        key = _hospital_key(base_glz, csw, hvac_code, fuel)
        row = savings_lookup[savings_lookup["LookKey"] == key].iloc[:1]
        if row.empty:
            raise KeyError(f"Hospital lookup failed for key {key}")
        row = row.iloc[0]
        cool_adjust = float(row.get("Cool_adjust", 1.0)) if str(inp.cooling_installed).strip().lower() == "no" else 1.0
        elec_heat_sf = float(row["ElecHeat_kWh_SF"] or 0.0)
        cool_sf      = float(row["Cool_kWh_SF"] or 0.0) * cool_adjust
        gas_heat_sf  = float(row["GasHeat_therm_SF"] or 0.0)
        eui_base     = float(row.get("Base_EUI")) if "Base_EUI" in row else None
        eui_csw      = float(row.get("CSW_EUI")) if "CSW_EUI" in row else None

    # MULTI-FAMILY
    elif bt in ("Multi-family","Multifamily","Multi family"):
        size_code = "Low" if (inp.floors or 0) < 4 else "Mid"
        hvac_code = "PTAC" if size_code == "Low" else "FCU"
        fuel = "Electric" if (inp.heating_fuel.lower()=="none" or hvac_code=="PTAC") else inp.heating_fuel.title()
        key = _mf_key(base_glz, csw, size_code, hvac_code, fuel)
        row = savings_lookup[savings_lookup["LookKey"] == key].iloc[:1]
        if row.empty:
            raise KeyError(f"Multi-family lookup failed for key {key}")
        row = row.iloc[0]
        infil_heat = 1.0; infil_cool = 1.0
        if (inp.infiltration_included and inp.infiltration_included.strip().lower() == "excluded"):
            infil_heat = float(row.get("Infil_Heat_Factor", 1.0) or 1.0)
            infil_cool = float(row.get("Infil_Cool_Factor", 1.0) or 1.0)
        cool_adjust = float(row.get("Cool_adjust", 1.0)) if str(inp.cooling_installed).strip().lower() == "no" else 1.0
        elec_heat_sf = float(row["ElecHeat_kWh_SF"] or 0.0) * infil_heat
        cool_sf      = float(row["Cool_kWh_SF"] or 0.0) * infil_cool * cool_adjust
        gas_heat_sf  = float(row["GasHeat_therm_SF"] or 0.0) * infil_heat
        eui_base     = float(row.get("Base_EUI")) if "Base_EUI" in row else None
        eui_csw      = float(row.get("CSW_EUI")) if "CSW_EUI" in row else None

    else:
        raise ValueError(f"Unsupported building type: {bt}")

    kwh_total  = (elec_heat_sf + cool_sf) * (inp.csw_installed_sf or 0)
    therm_total = (gas_heat_sf) * (inp.csw_installed_sf or 0)

    eui_savings = ((kwh_total * 3.413 + therm_total * 100.0) / inp.area_total_sf) if (inp.area_total_sf or 0) > 0 else None
    cost = (kwh_total * (inp.electric_rate or 0.0)) + (therm_total * (inp.gas_rate or 0.0))

    details = {
        "elec_heat_kWh_perSF": float(elec_heat_sf or 0.0),
        "cool_kWh_perSF": float(cool_sf or 0.0),
        "gas_heat_therm_perSF": float(gas_heat_sf or 0.0),
        "cool_adjust_applied": float(cool_adjust or 1.0),
    }
    return ComputeResult(
        elec_kwh_yr=float(kwh_total or 0.0),
        gas_therm_yr=float(therm_total or 0.0),
        cost_savings_usd=float(cost or 0.0),
        eui_base=float(eui_base) if eui_base is not None else None,
        eui_csw=float(eui_csw) if eui_csw is not None else None,
        eui_savings_kbtusf=float(eui_savings) if eui_savings is not None else None,
        details=details
    )
