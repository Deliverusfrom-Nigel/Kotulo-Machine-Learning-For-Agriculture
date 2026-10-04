"""
Generates test datasets that trigger HOLD or SELL recommendations.

Each dataset is two CSVs:
    test_hold_N_safex_prices.csv   — Date, Close
    test_hold_N_sa_factors.csv     — Date, ZAR_USD, Rainfall_mm, Fuel_Price
    test_sell_N_safex_prices.csv   — Date, Close
    test_sell_N_sa_factors.csv     — Date, ZAR_USD, Rainfall_mm, Fuel_Price

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

base = np.linspace(3200, 3650, n)

rally = np.zeros(n)
t = np.linspace(0, 1, 60)
rally[-60:] = (t ** 1.5) * 1500

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
summarise("Dataset 1 — Drought rally (HOLD)", prices_1)


# ============================================================== DATASET 2 ====
# Story:  Global grain shortage. Export demand surges, the rand slides,
#         fuel spikes. Prices rally for 90 days, then accelerate
#         sharply in the last 3 weeks as the shortage bites.
# Expected outcome: HOLD

rng = np.random.default_rng(202)
dates = pd.bdate_range("2022-06-01", periods=440)
n = len(dates)

base = np.linspace(3100, 3500, n)

correction = np.zeros(n)
correction[180:220] = -np.sin(np.linspace(0, np.pi, 40)) * 110

rally = np.zeros(n)
rally[-90:] = np.linspace(0, 1200, 90)

accel = np.zeros(n)
accel[-20:] = np.linspace(0, 500, 20)

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
summarise("Dataset 2 — Export surge (HOLD)", prices_2)


# ============================================================== DATASET 3 ====
# Story:  Bumper harvest across the maize belt. Rainfall is strong all
#         season, the rand strengthens against the dollar, fuel stays
#         flat. The market prices in a large crop and the price drifts
#         down for the whole second half of the series, accelerating
#         into a sharp fall over the last 45 business days.
# Expected outcome: SELL

rng = np.random.default_rng(303)
dates = pd.bdate_range("2022-06-01", periods=440)
n = len(dates)

# Slowly rising baseline that then rolls over
base = np.linspace(3400, 3600, n)

# The long slide: gentle decline over the middle 120 days
slide = np.zeros(n)
slide[200:320] = -np.linspace(0, 300, 120)

# The sharp fall: last 45 days
fall = np.zeros(n)
fall[-45:] = -np.linspace(0, 850, 45)

# A small dead-cat bounce so it isn't a straight line
bounce = np.zeros(n)
bounce[280:300] = np.sin(np.linspace(0, np.pi, 20)) * 60

prices_3 = np.round(base + slide + fall + bounce + rng.normal(0, 14, n), 2)

safex_3 = pd.DataFrame({"Date": dates, "Close": prices_3})
factors_3 = pd.DataFrame({
    "Date":        dates,
    # Rand strengthens (number falls) — bearish for local maize
    "ZAR_USD":     np.round(17.0 - np.linspace(0, 2.2, n) + rng.normal(0, 0.02, n), 4),
    # Strong, steady rainfall — bumper crop
    "Rainfall_mm": np.round(np.clip(rng.gamma(3.5, 7.0, n), 0.1, None), 2),
    # Fuel flat / slightly lower
    "Fuel_Price":  np.round(21.0 - np.linspace(0, 1.5, n) + rng.normal(0, 0.02, n), 2),
})

safex_3.to_csv("test_sell_1_safex_prices.csv", index=False)
factors_3.to_csv("test_sell_1_sa_factors.csv", index=False)
summarise("Dataset 3 — Bumper harvest (SELL)", prices_3)


# ============================================================== DATASET 4 ====
# Story:  Global grain supply recovers, export demand cools, the rand
#         holds firm. Prices trade sideways for a while, then break
#         down hard over the last 30 days as the new crop comes in.
#         Rainfall is heavy and fuel drifts lower.
# Expected outcome: SELL

rng = np.random.default_rng(404)
dates = pd.bdate_range("2022-06-01", periods=440)
n = len(dates)

# Essentially flat baseline with a mild drift up, then collapse
base = np.linspace(3300, 3450, n)

# Mild mid-series correction to add texture
correction = np.zeros(n)
correction[150:190] = -np.sin(np.linspace(0, np.pi, 40)) * 70

# The breakdown: last 30 days
breakdown = np.zeros(n)
breakdown[-30:] = -np.linspace(0, 1000, 30)

prices_4 = np.round(base + correction + breakdown + rng.normal(0, 13, n), 2)

safex_4 = pd.DataFrame({"Date": dates, "Close": prices_4})
factors_4 = pd.DataFrame({
    "Date":        dates,
    "ZAR_USD":     np.round(16.2 + rng.normal(0, 0.4, n).cumsum() * 0.02, 4),  # rand holds firm, no weakening
    "Rainfall_mm": np.round(np.clip(rng.gamma(4.0, 8.0, n), 0.1, None), 2),    # heavy, steady rain
    "Fuel_Price":  np.round(20.5 - np.linspace(0, 2.0, n) + rng.normal(0, 0.02, n), 2),  # fuel eases
})

safex_4.to_csv("test_sell_2_safex_prices.csv", index=False)
factors_4.to_csv("test_sell_2_sa_factors.csv", index=False)
summarise("Dataset 4 — Supply recovery breakdown (SELL)", prices_4)


print("\nDone. Eight CSVs written to the current folder:")
print("  test_hold_1_safex_prices.csv  /  test_hold_1_sa_factors.csv")
print("  test_hold_2_safex_prices.csv  /  test_hold_2_sa_factors.csv")
print("  test_sell_1_safex_prices.csv  /  test_sell_1_sa_factors.csv")
print("  test_sell_2_safex_prices.csv  /  test_sell_2_sa_factors.csv")