"""
Generates two test datasets that should trigger a HOLD recommendation.

Each dataset is two CSVs:
    test_hold_N_safex_prices.csv   — Date, Close
    test_hold_N_sa_factors.csv     — Date, ZAR_USD, Rainfall_mm, Fuel_Price

Usage:
    python generate_test_data.py
"""
import numpy as np
import pandas as pd


def summarise(name, prices):
    print(f"\n{name}")
    print(f"  Last price         : R {prices[-1]:,.2f}")
    print(f"  Last 5-day change  : {(prices[-1] / prices[-6] - 1) * 100:+.2f}%")
    print(f"  Last 10-day change : {(prices[-1] / prices[-11] - 1) * 100:+.2f}%")
    print(f"  Last 30-day change : {(prices[-1] / prices[-31] - 1) * 100:+.2f}%")


# ============================================================== DATASET 1 ====
# Story:  Severe drought across the Free State maize belt.
#         Rainfall collapses, the rand weakens, fuel creeps up.
#         The market prices in a much smaller crop and rallies hard
#         over the last 60 business days.
# Expected outcome: HOLD

rng = np.random.default_rng(101)
dates = pd.bdate_range("2022-06-01", periods=440)
n = len(dates)

# Quietly rising baseline for the first 380 days
base = np.linspace(3200, 3650, n)

# The drought rally — accelerating over the last 60 days
rally = np.zeros(n)
t = np.linspace(0, 1, 60)
rally[-60:] = (t ** 1.5) * 1500          # +1500 R/ton, weighted to the end

# A mild correction mid-series so the pre-rally period isn't boring
correction = np.zeros(n)
correction[180:220] = -np.sin(np.linspace(0, np.pi, 40)) * 90

prices_1 = np.round(base + rally + correction + rng.normal(0, 15, n), 2)

safex_1 = pd.DataFrame({"Date": dates, "Close": prices_1})
factors_1 = pd.DataFrame({
    "Date":        dates,
    "ZAR_USD":     np.round(14.5 + np.linspace(0, 2.5, n) + rng.normal(0, 0.02, n), 4),
    "Rainfall_mm": np.round(np.clip(rng.gamma(2, 5, n) * (1 - np.arange(n) / n * 0.95), 0.1, None), 2),
    "Fuel_Price":  np.round(20.0 + np.linspace(0, 3.5, n) + rng.normal(0, 0.02, n), 2),
})

safex_1.to_csv("test_hold_1_safex_prices.csv", index=False)
factors_1.to_csv("test_hold_1_sa_factors.csv", index=False)
summarise("Dataset 1 — Drought rally", prices_1)


# ============================================================== DATASET 2 ====
# Story:  Global grain shortage. Export demand surges, the rand slides,
#         fuel spikes. Prices rally for 90 days, then accelerate
#         sharply in the last 3 weeks as the shortage bites.
# Expected outcome: HOLD

rng = np.random.default_rng(202)
dates = pd.bdate_range("2022-06-01", periods=440)
n = len(dates)

# Slowly rising baseline
base = np.linspace(3100, 3500, n)

# A short correction around month 9
correction = np.zeros(n)
correction[180:220] = -np.sin(np.linspace(0, np.pi, 40)) * 110

# The long export rally: last 90 days
rally = np.zeros(n)
rally[-90:] = np.linspace(0, 1200, 90)   # +1200 over 90 days

# The final acceleration: last 20 days
accel = np.zeros(n)
accel[-20:] = np.linspace(0, 500, 20)    # extra +500 over 20 days

prices_2 = np.round(base + rally + accel + correction + rng.normal(0, 14, n), 2)

safex_2 = pd.DataFrame({"Date": dates, "Close": prices_2})
factors_2 = pd.DataFrame({
    "Date":        dates,
    "ZAR_USD":     np.round(14.2 + np.linspace(0, 2.8, n) + rng.normal(0, 0.02, n), 4),
    "Rainfall_mm": np.round(np.clip(rng.gamma(2, 6, n) - np.arange(n) / n * 4, 0.1, None), 2),
    "Fuel_Price":  np.round(19.5 + np.linspace(0, 4.0, n) + rng.normal(0, 0.02, n), 2),
})

safex_2.to_csv("test_hold_2_safex_prices.csv", index=False)
factors_2.to_csv("test_hold_2_sa_factors.csv", index=False)
summarise("Dataset 2 — Export surge", prices_2)

print("\nDone. Four CSVs written to the current folder.")