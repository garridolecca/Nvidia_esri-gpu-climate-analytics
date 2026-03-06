"""Generate realistic US weather station data."""
import numpy as np, pandas as pd, os

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA, exist_ok=True)
np.random.seed(42)

print("Generating synthetic NOAA-like weather station data...")

# ~500 stations across CONUS
n = 500
lons = np.random.uniform(-125, -67, n)
lats = np.random.uniform(25, 49, n)
# Temperature varies with latitude (colder north) + elevation proxy
base_temp_f = 70 - (lats - 25) * 1.8 + np.random.normal(0, 5, n)
# January adjustment (winter)
jan_adj = -15 - (lats - 25) * 0.5
avg_temp = base_temp_f + jan_adj

stations = pd.DataFrame({
    "station_id": [f"US{i:06d}" for i in range(n)],
    "name": [f"Station_{i}" for i in range(n)],
    "lat": np.round(lats, 4), "lon": np.round(lons, 4),
    "elevation_m": np.round(np.random.uniform(0, 2000, n) * np.abs(lons + 100) / 30, 0),
    "tmax_f": np.round(avg_temp + np.random.uniform(5, 15, n), 1),
    "tmin_f": np.round(avg_temp - np.random.uniform(5, 15, n), 1),
    "tavg_f": np.round(avg_temp, 1),
    "precip_in": np.round(np.random.exponential(0.3, n), 2),
    "snow_in": np.round(np.where(avg_temp < 32, np.random.exponential(2, n), 0), 1),
    "wind_mph": np.round(np.random.uniform(2, 25, n), 1),
    "humidity_pct": np.round(np.random.uniform(20, 95, n), 0),
})
stations["tmax_c"] = np.round((stations["tmax_f"] - 32) * 5/9, 1)
stations["tmin_c"] = np.round((stations["tmin_f"] - 32) * 5/9, 1)
stations["tavg_c"] = np.round((stations["tavg_f"] - 32) * 5/9, 1)
stations["temp_range_c"] = stations["tmax_c"] - stations["tmin_c"]

stations.to_csv(os.path.join(DATA, "stations.csv"), index=False)
print(f"  Generated {n} stations")
print(f"  Temp range: {stations['tavg_c'].min():.1f}C to {stations['tavg_c'].max():.1f}C")
print("Done!")
