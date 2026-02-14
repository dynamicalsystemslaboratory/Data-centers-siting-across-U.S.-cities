import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# PATHS
MSA_SHP = "tl_2024_us_cbsa/tl_2024_us_cbsa.shp"      # polygons
DC_SHP  = "/Users/camilla/PycharmProjects/datacenters/DC_Shapefiles_ex/Data_centers_exact_location.shp"      # points

MSA_ID_COL = "CBSAFP"
CONTIGUOUS = {
    "Alabama","Arizona","Arkansas","California","Colorado","Connecticut","Delaware",
    "District of Columbia","Florida","Georgia","Idaho","Illinois","Indiana","Iowa","Kansas",
    "Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan","Minnesota",
    "Mississippi","Missouri","Montana","Nebraska","Nevada","New Hampshire","New Jersey",
    "New Mexico","New York","North Carolina","North Dakota","Ohio","Oklahoma","Oregon",
    "Pennsylvania","Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah",
    "Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"
}
# OUTPUTS
FIG_DIR = "Figures"
OUT_FIG = os.path.join(FIG_DIR, "pan_dist_clustering.png")
OUT_TEX = "updated_numbers_snippet.tex"
OUT_MSA_STATS_CSV = "msa_clustering_stats.csv"

# PARAMETERS
K_MIN = 2
K_MAX = 6
N_INIT = 25
RANDOM_STATE = 0

CRS_CONUS = "EPSG:5070"   # meters for CONUS


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def km(m):
    return m / 1000.0

def pairwise_centroid_distances(centroids):
    k = centroids.shape[0]
    if k < 2:
        return np.array([])
    d = []
    for i in range(k):
        for j in range(i+1, k):
            d.append(np.linalg.norm(centroids[i] - centroids[j]))
    return np.asarray(d)

def choose_k_by_silhouette(X, k_min=2, k_max=6):
    n = X.shape[0]
    k_max_eff = min(k_max, n - 1)
    if k_max_eff < k_min:
        return None, None, None

    best_k, best_model, best_sil = None, None, -np.inf
    for k in range(k_min, k_max_eff + 1):
        model = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE)
        labels = model.fit_predict(X)
        sil = silhouette_score(X, labels, metric="euclidean")
        if sil > best_sil:
            best_k, best_model, best_sil = k, model, sil
    return best_k, best_model, best_sil


def main():
    if not os.path.exists(MSA_SHP):
        raise FileNotFoundError(f"MSA shapefile not found: {MSA_SHP}")
    if not os.path.exists(DC_SHP):
        raise FileNotFoundError(f"Datacenter shapefile not found: {DC_SHP}")

    msa = gpd.read_file(MSA_SHP)
    dc  = gpd.read_file(DC_SHP)

    if MSA_ID_COL not in msa.columns:
        raise ValueError(
            f"MSA shapefile missing '{MSA_ID_COL}'. "
            f"Available columns: {msa.columns.tolist()}"
        )

    if dc.geometry.isna().any():
        dc = dc.dropna(subset=["geometry"]).copy()

    # Project to EPSG:5070 (meters)
    msa = msa.to_crs(CRS_CONUS)
    dc  = dc.to_crs(CRS_CONUS)
    STATE_NAME_COL = "state"

    dc[STATE_NAME_COL] = dc[STATE_NAME_COL].astype(str).str.strip()
    dc = dc[dc[STATE_NAME_COL].isin(CONTIGUOUS)].copy()

    # Spatial join: assign each datacenter point to an MSA polygon
    joined = gpd.sjoin(
        dc,
        msa[[MSA_ID_COL, "geometry"]],
        how="left",
        predicate="within"
    )

    # If you suspect boundary issues, fallback to intersects for those not assigned
    n_missing = joined[MSA_ID_COL].isna().sum()
    if n_missing > 0:
        dc_missing = joined[joined[MSA_ID_COL].isna()].drop(columns=[MSA_ID_COL, "index_right"], errors="ignore")
        joined2 = gpd.sjoin(
            dc_missing,
            msa[[MSA_ID_COL, "geometry"]],
            how="left",
            predicate="intersects"
        )
        joined.loc[joined[MSA_ID_COL].isna(), MSA_ID_COL] = joined2[MSA_ID_COL].values

    joined = joined.dropna(subset=[MSA_ID_COL]).copy()

    # Keep only MSAs with >= 3 facilities
    msa_counts = joined.groupby(MSA_ID_COL).size()
    keep_msas = msa_counts[msa_counts >= 3].index.tolist()
    joined = joined[joined[MSA_ID_COL].isin(keep_msas)].copy()

    rows = []
    for msa_id, sub in joined.groupby(MSA_ID_COL):
        coords = np.vstack([sub.geometry.x.values, sub.geometry.y.values]).T
        n = coords.shape[0]
        if n < 3:
            continue

        k_hat, model, sil = choose_k_by_silhouette(coords, K_MIN, K_MAX)
        if model is None:
            continue

        labels = model.labels_
        centroids = model.cluster_centers_

        # intra distances: each point -> assigned centroid
        d = np.linalg.norm(coords - centroids[labels], axis=1)  # meters

        intra_mean = km(d.mean())
        intra_median = km(np.median(d))
        intra_max = km(d.max())
        intra_sum = km(d.sum())

        cd = pairwise_centroid_distances(centroids)  # meters
        cent_mean = km(cd.mean()) if len(cd) else np.nan
        cent_median = km(np.median(cd)) if len(cd) else np.nan

        rows.append({
            "MSA_ID": str(msa_id),
            "n_facilities": int(n),
            "k_hat": int(k_hat),
            "silhouette": float(sil),
            "intra_mean_km": float(intra_mean),
            "intra_median_km": float(intra_median),
            "intra_max_km": float(intra_max),
            "intra_sum_km": float(intra_sum),
            "centroid_mean_km": float(cent_mean),
            "centroid_median_km": float(cent_median),
        })

    stats = gpd.pd.DataFrame(rows)
    if stats.empty:
        raise RuntimeError("No MSA was clustered. Check the join + MSA_ID_COL + geometry validity.")

    intra_dist = stats["intra_mean_km"].dropna().values
    cent_dist  = stats["centroid_mean_km"].dropna().values

    intra_mean_all = float(np.mean(intra_dist))
    intra_median_all = float(np.median(intra_dist))
    cent_mean_all = float(np.mean(cent_dist))
    cent_median_all = float(np.median(cent_dist))

    # Save stats per MSA
    stats.to_csv(OUT_MSA_STATS_CSV, index=False)
    print(f"[OK] Wrote per-MSA clustering stats: {OUT_MSA_STATS_CSV}")

    # Plot figure (two panels)
    ensure_dir(FIG_DIR)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))

    axes[0].hist(intra_dist, bins=40)
    axes[0].axvline(intra_mean_all, linestyle="--", linewidth=1)
    axes[0].axvline(intra_median_all, linestyle=":", linewidth=1)
    axes[0].set_xlabel("Average within-cluster distance (km)")
    axes[0].set_ylabel("MSA count")

    axes[1].hist(cent_dist, bins=40)
    axes[1].axvline(cent_mean_all, linestyle="--", linewidth=1)
    axes[1].axvline(cent_median_all, linestyle=":", linewidth=1)
    axes[1].set_xlabel("Average centroid-to-centroid distance (km)")
    axes[1].set_ylabel("MSA count")

    plt.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_FIG}")
    print(f"Clustered MSAs: {len(stats)}")
    print(f"Intra mean/median across MSAs (km): {intra_mean_all:.2f} / {intra_median_all:.2f}")
    print(f"Centroid mean/median across MSAs (km): {cent_mean_all:.2f} / {cent_median_all:.2f}")

    # Write Overleaf snippet
    tex = []
    tex.append("% Auto-generated clustering stats (MSA k-means)")
    tex.append(rf"\newcommand{{\NMSAClustered}}{{{len(stats)}}}")
    tex.append(rf"\newcommand{{\KMax}}{{{K_MAX}}}")
    tex.append(rf"\newcommand{{\IntraMeanKM}}{{{intra_mean_all:.2f}}}")
    tex.append(rf"\newcommand{{\IntraMedianKM}}{{{intra_median_all:.2f}}}")
    tex.append(rf"\newcommand{{\CentMeanKM}}{{{cent_mean_all:.2f}}}")
    tex.append(rf"\newcommand{{\CentMedianKM}}{{{cent_median_all:.2f}}}")

    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(tex))


if __name__ == "__main__":
    main()
