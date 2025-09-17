# engine/engine.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

# ---------------- Paths ----------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# ---------------- CSV helpers ----------------
def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    return df

# ---------------- Public loaders used by the app ----------------
def load_weather() -> pd.DataFrame:
    """
    Load weather_information.csv and normalize common headers:
      - State
      - Cities
      - Heating Degree Days (HDD)
      - Cooling Degree Days (CDD)
    Also drops any stray 'None' columns created by Excel exports.
    """
    p = DATA_DIR / "weather_information.csv"
    df = _read_csv(p)

    # Rename flexible headers to canonical
    ren = {}
    for c in df.columns:
        lc = c.lower()
        if "state" in lc and "hdd" not in lc and "cdd" not in lc:
            ren[c] = "State"
        elif "cities" in lc or lc == "city":
            ren[c] = "Cities"
        elif "heating degree" in lc or "(hdd" in lc or lc == "hdd":
            ren[c] = "Heating Degree Days (HDD)"
        elif "cooling degree" in lc or "(cdd" in lc or lc == "cdd":
            ren[c] = "Cooling Degree Days (CDD)"
        elif lc.startswith("none"):
            # we'll drop these below
            pass

    df = df.rename(columns=ren)
    # Drop stray None* columns if present
    keep = {"State", "Cities", "Heating Degree Days (HDD)", "Cooling Degree Days (CDD)"}
    df = df[[c for c in df.columns if c in keep]].copy()

    # Coerce HDD/CDD to numeric
    for c in ["Heating Degree Days (HDD)", "Cooling Degree Days (CDD)"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def load_lists() -> pd.DataFrame:
    p = DATA_DIR / "lists.csv"
    return _read_csv(p) if p.exists() else pd.DataFrame()

# ---------------- Simple internal savings model (placeholder) ----------------
def _estimate_wwr(building_type: str, floors: int) -> float:
    bt = (building_type or "").lower()
    f = max(1, int(floors or 1))
    base = 0.25
    if bt == "office":
        base = 0.30
    elif bt == "school":
        base = 0.20
    elif bt == "hotel":
        base = 0.28
    elif bt == "hospital":
        base = 0.18
    elif "multi" in bt:
        base = 0.22
    return min(0.60, round(base + 0.01 * max(0, f - 1), 3))

def _building_factors(bt: str) -> tuple[float, float]:
    """(heat_factor, cool_factor) relative to 'office'=1.0."""
    b = (bt or "").lower()
    if b == "school":
        return 0.85, 0.85
    if b == "hotel":
        return 0.95, 1.15
    if b == "hospital":
        return 1.20, 1.25
    if "multi" in b:
        return 1.00, 0.75
    return 1.00, 1.00  # office

def _window_factor(existing_window_type_ui: str) -> float:
    # Savings potential higher for single-pane baseline
    return 1.00 if "single" in (existing_window_type_ui or "").lower() else 0.60

def compute_savings(
    *,
    weather_hdd: float,
    weather_cdd: float,
    building_type: str,
    sub_building_type: str,           # unused in the simple model but kept for API stability
    hvac_ui: str,                      # unused in the simple model (placeholder)
    heating_fuel_ui: str,             # "Natural Gas" | "Electric" | "None"
    cooling_installed: bool,
    existing_window_type_ui: str,     # "Single pane" | "Double pane"
    csw_glazing_ui: str,              # currently unused; future hook
    building_area_sf: float,
    annual_operating_hours: float | None = None,  # only meaningful for office
    hotel_occupancy_pct: float | None = None,     # not used here
    include_infiltration: bool | None = None,     # only MF; simple model ignores
    floors: int | None = None,
    csw_installed_sf: float | None = None,
    electric_rate: float | None = None,
    gas_rate: float | None = None,
) -> dict:
    """
    Lightweight, deterministic placeholder used in the earlier working build.
    Produces reasonable, nonzero estimates without any Excel or lookup CSVs.
    """

    # Inputs
    bt = building_type or "Office"
    floors = int(floors or 1)
    area_total = float(building_area_sf or 0.0)
    csw_area = float(csw_installed_sf or 0.0) or area_total
    hdd = float(weather_hdd or 0.0)
    cdd = float(weather_cdd or 0.0)
    elec_rate = float(electric_rate or 0.12)
    gas_rate = float(gas_rate or 1.00)
    fuel = (heating_fuel_ui or "Natural Gas").strip().title()

    # Core drivers
    wwr = _estimate_wwr(bt, floors)
    wf = _window_factor(existing_window_type_ui)
    heat_f, cool_f = _building_factors(bt)

    # Normalize HDD/CDD relative to typical continental values
    hdd_norm = (hdd / 6000.0) if hdd > 0 else 0.0
    cdd_norm = (cdd / 1200.0) if cdd > 0 else 0.0

    # Base envelopes (very conservative placeholders):
    # - Start from a small “savings potential intensity” and scale by WWR, HDD/CDD, window factor, building factor.
    # - Convert heating from kBtu to kWh or therms depending on fuel.
    #   (1 therm = 100,000 Btu; 1 kWh = 3.413 kBtu)
    heat_kbtu_sf = 6.0 * wwr * hdd_norm * wf * heat_f          # kBtu/sf/yr saved by reducing envelope heat loss
    cool_kwh_sf  = 0.9 * wwr * cdd_norm * wf * cool_f          # kWh/sf/yr saved for cooling & aux

    # Allocate heating savings to electric or gas
    if fuel == "Electric":
        elec_heat_kwh_sf = heat_kbtu_sf / 3.413
        gas_heat_therm_sf = 0.0
    elif fuel == "None":
        # Treat as electric space heat for the purpose of “savings capture”
        elec_heat_kwh_sf = heat_kbtu_sf / 3.413
        gas_heat_therm_sf = 0.0
    else:
        # Natural Gas
        elec_heat_kwh_sf = 0.0
        gas_heat_therm_sf = heat_kbtu_sf / 100.0  # 100 kBtu per therm

    # Cooling installed? If not, keep a small residual (fans/aux), else 100%
    if not cooling_installed:
        cool_kwh_sf *= 0.10

    # Totals over CSW installed area
    total_kwh = (elec_heat_kwh_sf + cool_kwh_sf) * csw_area
    total_therms = (gas_heat_therm_sf) * csw_area

    # Costs
    cost_savings = total_kwh * elec_rate + total_therms * gas_rate

    # EUI placeholders (coarse): show relative improvement signal
    base_eui = 55.0 * heat_f + 18.0 * cool_f   # kBtu/sf/yr
    csw_eui  = base_eui - (heat_kbtu_sf + cool_kwh_sf * 3.413)
    eui_sav  = base_eui - csw_eui

    return {
        "elec_heat_kwh_per_sf": float(elec_heat_kwh_sf),
        "cool_kwh_per_sf": float(cool_kwh_sf),
        "gas_heat_therm_per_sf": float(gas_heat_therm_sf),
        "total_kwh": float(total_kwh),
        "total_therms": float(total_therms),
        "cost_savings_usd": float(cost_savings),
        "eui_base": float(base_eui),
        "eui_csw": float(csw_eui),
        "eui_savings_kbtusf": float(eui_sav),
    }
