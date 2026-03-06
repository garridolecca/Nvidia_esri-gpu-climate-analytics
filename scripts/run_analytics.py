"""GPU-Accelerated Climate Analytics Pipeline."""
import os, json, time, numpy as np, pandas as pd

try:
    import cupy as cp; GPU = True; print(f"GPU: CuPy {cp.__version__}")
except ImportError: GPU = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "output"); WEB = os.path.join(BASE, "webapp", "data")
os.makedirs(OUT, exist_ok=True); os.makedirs(WEB, exist_ok=True)

def save(data, name):
    for d in [OUT, WEB]:
        with open(os.path.join(d, name), "w") as f:
            json.dump(data, f)

df = pd.read_csv(os.path.join(DATA, "stations.csv"))
print(f"Loaded {len(df)} stations")

# Station GeoJSON
print("\n=== Station Export ===")
feats = []
for _, r in df.iterrows():
    feats.append({"type":"Feature","geometry":{"type":"Point","coordinates":[float(r.lon),float(r.lat)]},
                 "properties":{k: float(r[k]) if isinstance(r[k], (np.floating, float)) else r[k]
                              for k in ["station_id","name","tavg_c","tmax_c","tmin_c","temp_range_c",
                                       "precip_in","snow_in","wind_mph","humidity_pct","elevation_m"]}})
save({"type":"FeatureCollection","features":feats}, "stations.geojson")

# GPU IDW Temperature Interpolation
print("\n=== GPU IDW Temperature Interpolation ===")
t0 = time.time()
grid_nx, grid_ny = 120, 60
gx = np.linspace(-125, -67, grid_nx)
gy = np.linspace(25, 49, grid_ny)

if GPU:
    sx = cp.asarray(df.lon.values); sy = cp.asarray(df.lat.values); sv = cp.asarray(df.tavg_c.values)
    gxf = cp.asarray(np.tile(gx, grid_ny)); gyf = cp.asarray(np.repeat(gy, grid_nx))
    dx = gxf[:, None] - sx[None, :]; dy = gyf[:, None] - sy[None, :]
    dist = cp.sqrt(dx**2 + dy**2); dist = cp.maximum(dist, 0.01)
    w = 1.0 / dist**2
    temp_grid = cp.asnumpy(cp.sum(w * sv[None, :], axis=1) / cp.sum(w, axis=1)).reshape(grid_ny, grid_nx)
    # Also interpolate precip
    pv = cp.asarray(df.precip_in.values)
    precip_grid = cp.asnumpy(cp.sum(w * pv[None, :], axis=1) / cp.sum(w, axis=1)).reshape(grid_ny, grid_nx)
else:
    gxf = np.tile(gx, grid_ny); gyf = np.repeat(gy, grid_nx)
    dx = gxf[:, None] - df.lon.values[None, :]; dy = gyf[:, None] - df.lat.values[None, :]
    dist = np.maximum(np.sqrt(dx**2 + dy**2), 0.01)
    w = 1.0 / dist**2
    temp_grid = (np.sum(w * df.tavg_c.values[None, :], axis=1) / np.sum(w, axis=1)).reshape(grid_ny, grid_nx)
    precip_grid = (np.sum(w * df.precip_in.values[None, :], axis=1) / np.sum(w, axis=1)).reshape(grid_ny, grid_nx)

print(f"  IDW grid: {grid_nx}x{grid_ny} ({time.time()-t0:.1f}s)")

# Export temperature surface
t_feats = []
for i in range(grid_ny):
    for j in range(grid_nx):
        t_feats.append({"type":"Feature","geometry":{"type":"Point","coordinates":[float(gx[j]),float(gy[i])]},
                       "properties":{"temp_c":round(float(temp_grid[i,j]),1)}})
save({"type":"FeatureCollection","features":t_feats}, "temp_surface.geojson")

p_feats = []
for i in range(grid_ny):
    for j in range(grid_nx):
        p_feats.append({"type":"Feature","geometry":{"type":"Point","coordinates":[float(gx[j]),float(gy[i])]},
                       "properties":{"precip_in":round(float(precip_grid[i,j]),3)}})
save({"type":"FeatureCollection","features":p_feats}, "precip_surface.geojson")

# GPU Anomaly Detection
print("\n=== GPU Temperature Anomaly Detection ===")
t0 = time.time()
# Expected temp by latitude: linear model
lats = df.lat.values
expected = 20 - (lats - 25) * 1.0  # rough expected Jan temp
anomaly = df.tavg_c.values - expected
df["anomaly_c"] = np.round(anomaly, 1)
df["anomaly_class"] = np.where(anomaly > 5, "Warm Anomaly", np.where(anomaly < -5, "Cold Anomaly", "Normal"))

anom_feats = []
for _, r in df[df.anomaly_class != "Normal"].iterrows():
    anom_feats.append({"type":"Feature","geometry":{"type":"Point","coordinates":[float(r.lon),float(r.lat)]},
                      "properties":{"station_id":r.station_id,"tavg_c":float(r.tavg_c),
                                   "anomaly_c":float(r.anomaly_c),"class":r.anomaly_class}})
save({"type":"FeatureCollection","features":anom_feats}, "anomalies.geojson")
print(f"  {len(anom_feats)} anomaly stations ({time.time()-t0:.1f}s)")

# GPU K-Means Climate Clustering
print("\n=== GPU Climate Clustering ===")
t0 = time.time()
clust_feats = np.column_stack([df.tavg_c, df.temp_range_c, df.precip_in, df.humidity_pct, df.lat/10]).astype(np.float64)
# Normalize
clust_feats = (clust_feats - clust_feats.mean(axis=0)) / (clust_feats.std(axis=0) + 1e-10)

k = 6
if GPU:
    X = cp.asarray(clust_feats)
    centroids = X[cp.random.choice(len(X), k, replace=False)].copy()
    for _ in range(100):
        dists = cp.stack([cp.sum((X - centroids[c])**2, axis=1) for c in range(k)], axis=1)
        labels = cp.argmin(dists, axis=1)
        new_c = cp.stack([X[labels==c].mean(axis=0) if cp.any(labels==c) else centroids[c] for c in range(k)])
        if float(cp.max(cp.abs(new_c - centroids))) < 1e-6: break
        centroids = new_c
    cluster_labels = cp.asnumpy(labels)
else:
    from sklearn.cluster import KMeans
    cluster_labels = KMeans(k, random_state=42).fit_predict(clust_feats)

zone_names = ["Continental Cold","Maritime Mild","Humid Subtropical","Arid Desert","Mountain","Pacific Coast"]
cl_feats = []
for idx, r in df.iterrows():
    cl = int(cluster_labels[idx])
    cl_feats.append({"type":"Feature","geometry":{"type":"Point","coordinates":[float(r.lon),float(r.lat)]},
                    "properties":{"station_id":r.station_id,"cluster":cl,"zone":zone_names[cl],
                                 "tavg_c":float(r.tavg_c),"precip_in":float(r.precip_in)}})
save({"type":"FeatureCollection","features":cl_feats}, "climate_clusters.geojson")
print(f"  {k} climate zones ({time.time()-t0:.1f}s)")

# Summary
summary = {"stations":len(df),"temp_range":[round(float(df.tavg_c.min()),1),round(float(df.tavg_c.max()),1)],
           "avg_temp_c":round(float(df.tavg_c.mean()),1),"avg_precip_in":round(float(df.precip_in.mean()),2),
           "warm_anomalies":int((df.anomaly_class=="Warm Anomaly").sum()),
           "cold_anomalies":int((df.anomaly_class=="Cold Anomaly").sum()),
           "climate_zones":k,"gpu":GPU,"grid_size":[grid_nx,grid_ny],
           "location":"Continental United States","period":"January 2024"}
save(summary, "summary.json")
print("\nComplete!")
for k2,v in summary.items(): print(f"  {k2}: {v}")
