
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def load_weather():
    # Load the CSV exactly as committed in the repo
    df = pd.read_csv(DATA_DIR / "weather_information.csv")
    return df

def load_lists():
    # The "Lists" sheet is a wide sheet; you'll likely reshape into named lists.
    # For now, return raw and let the UI filter columns to build dropdowns.
    df = pd.read_csv(DATA_DIR / "lists.csv")
    return df

def load_savings_lookup():
    df = pd.read_csv(DATA_DIR / "savings_lookup.csv")
    df.columns = [str(c).strip() for c in df.columns]
    return df

def load_regression(building_key: str):
    """
    building_key: one of 'office','school','hotel','ps','ss','hosp','mf'
    Maps to the corresponding 'Regresson List_*' CSV.
    """
    key_map = {
        "office": "regresson_list_office.csv",
        "school": "regresson_list_sh.csv",     # adjust if 'SH' means School; confirm mapping
        "lh": "regresson_list_lh.csv",
        "ps": "regresson_list_ps.csv",
        "ss": "regresson_list_ss.csv",
        "hosp": "regresson_list_hosp.csv",
        "mf": "regresson_list_mf.csv",
    }
    fname = key_map.get(building_key.lower())
    if not fname:
        raise ValueError(f"Unknown building_key: {building_key}")
    df = pd.read_csv(DATA_DIR / fname)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def compute_hdd_cdd(weather_df, state: str, city: str):
    # Try matching state & city by normalized columns
    cols = [c.lower() for c in weather_df.columns]
    # Heuristic: find likely columns for state/city/hdd/cdd
    def find_col(substrs):
        for s in substrs:
            for c in weather_df.columns:
                if s in c.lower():
                    return c
        return None
    state_col = find_col(["state"])
    city_col  = find_col(["city"])
    hdd_col   = find_col(["hdd"])
    cdd_col   = find_col(["cdd"])
    row = weather_df[
        (weather_df[state_col].astype(str).str.strip().str.lower() == state.strip().lower()) &
        (weather_df[city_col].astype(str).str.strip().str.lower() == city.strip().lower())
    ]
    if row.empty:
        return None, None
    return float(row.iloc[0][hdd_col]), float(row.iloc[0][cdd_col])

def regression_value(reg_df, base: str, csw_type: str, hvac: str, fuel: str, size: str, which: str = "heating"):
    """
    Select coefficients a,b,c from regression table by keys.
    which: 'heating' or 'cooling'
    Returns dict(a=..., b=..., c=...)
    """
    # Normalize columns likely named like: 'Base','CSW Type','HVAC Type','Fuel'
    # and two sub-tables for Heating Coefficients and Cooling Coefficients labeled 'a','b','c' under section headers.
    # Because sheet is denormalized, a one-time schema clean may be needed.
    # This function is a placeholder to be filled once schema is confirmed.
    raise NotImplementedError("Wire specific regression key selections here based on your sheet schema.")

def savings_from_regression(a: float, b: float, c: float, x: float) -> float:
    # Apply a + b*x + c*x^2
    return a + b * x + c * (x ** 2)
