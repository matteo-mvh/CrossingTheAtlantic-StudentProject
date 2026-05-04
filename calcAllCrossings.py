import numpy as np
import xarray as xr
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
from pathlib import Path
import cartopy.feature as cfeature

from matplotlib.patches import Rectangle

from parameters import (
    ROUTE_POINTS,
    DEPARTURE_START,
    DEPARTURE_END,
    DEPARTURE_FREQ,
    DT_HOURS,
    MAX_DAYS,
    WAYPOINT_TOLERANCE_M,
    PLOT_MAX_TRACKS,
    HEATMAP_MAX_PLOT_DAYS,
    HEATMAP_TIME_DOWNWARD,
    OUT_DIR,
    LEG_MARGIN_LON,
    LEG_MARGIN_LAT,
    DOWNLOAD_EXTRA,
    LOW_ANGLE_EXTRAPOLATING,
    HIGH_ANGLE_EXTRAPOLATING,
    TEST_RUN,
    TEST_RUN_DAYS,
    CHECKPOINT_EVERY,
    INTERPOLATE_IN_TIME,
    WAVE_MODERATE_THRESHOLD,
    WAVE_CRITICAL_THRESHOLD,
    WAVE_MODERATE_WEIGHT,
    WAVE_CRITICAL_WEIGHT,
    FINAL_SCORE_WEIGHT_TIME,
    FINAL_SCORE_WEIGHT_SAFETY,
    FINAL_SCORE_WEIGHT_WAVE,
    FINAL_SCORE_WEIGHT_WIND,
)

# ============================================================
# BOAT SPEED FUNCTION
# ============================================================
from Polar_Diagram import hr50_boat_speed

# ============================================================
# GEOGRAPHIC HELPERS
# ============================================================
R_EARTH = 6371000.0  # m


def haversine_distance(lon1, lat1, lon2, lat2):
    """Distance in meters."""
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R_EARTH * c


def initial_bearing(lon1, lat1, lon2, lat2):
    """
    Bearing from point 1 to point 2 in degrees clockwise from north.
    """
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1

    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    bearing = np.rad2deg(np.arctan2(x, y)) % 360
    return bearing


def destination_point(lon, lat, bearing_deg, distance_m):
    """
    Move from lon/lat along a bearing for a given distance.
    Returns new lon, lat in degrees.
    """
    bearing = np.deg2rad(bearing_deg)
    lon1 = np.deg2rad(lon)
    lat1 = np.deg2rad(lat)

    ang_dist = distance_m / R_EARTH

    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(ang_dist)
        + np.cos(lat1) * np.sin(ang_dist) * np.cos(bearing)
    )

    lon2 = lon1 + np.arctan2(
        np.sin(bearing) * np.sin(ang_dist) * np.cos(lat1),
        np.cos(ang_dist) - np.sin(lat1) * np.sin(lat2)
    )

    lon2 = (lon2 + np.pi) % (2 * np.pi) - np.pi

    return np.rad2deg(lon2), np.rad2deg(lat2)


def speed_bearing_to_uv(speed, bearing_deg):
    """
    Convert speed + bearing (clockwise from north) to eastward/northward velocity.
    """
    bearing_rad = np.deg2rad(bearing_deg)
    u = speed * np.sin(bearing_rad)   # eastward
    v = speed * np.cos(bearing_rad)   # northward
    return float(u), float(v)


def uv_to_speed_and_bearing(u, v):
    """
    Convert eastward/northward velocity to speed and bearing.
    Bearing is clockwise from north.
    """
    speed = np.sqrt(u**2 + v**2)
    bearing = (np.rad2deg(np.arctan2(u, v)) + 360) % 360
    return float(speed), float(bearing)


def sanity_check_strict_order(results_df):
    """
    Check that boats arrive in the same order as they departed.
    Prints a brief result only.
    """
    finished = results_df[results_df["finished"]].copy()

    if len(finished) < 2:
        print("Sanity check skipped (not enough finished runs).")
        return

    finished = finished.sort_values("departure_time").reset_index(drop=True)
    arrivals = pd.to_datetime(finished["arrival_time"])

    if (arrivals.diff().dropna() >= pd.Timedelta(0)).all():
        print("Sanity check OK: arrival order matches departure order.")
    else:
        print("Sanity check FAILED: arrival order does NOT match departure order.")


# ============================================================
# TEST RUN HELPER
# ============================================================
def get_effective_departure_window(start_date, end_date, test_run=False, test_run_days=14):
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    if not test_run:
        return start_ts, end_ts

    effective_end = start_ts + pd.Timedelta(days=test_run_days - 1)
    effective_end = min(effective_end, end_ts)

    print("\n============================================================")
    print("TEST RUN MODE ENABLED")
    print(f"Original departure window : {start_ts.date()} -> {end_ts.date()}")
    print(f"Effective departure window: {start_ts.date()} -> {effective_end.date()}")
    print("============================================================")

    return start_ts, effective_end


# ============================================================
# LEG FILE / BOX HELPERS
# ============================================================
def save_checkpoint(results_df, checkpoint_file):
    checkpoint_file = Path(checkpoint_file)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_pickle(checkpoint_file)


def load_checkpoint(checkpoint_file):
    checkpoint_file = Path(checkpoint_file)
    if checkpoint_file.exists():
        df = pd.read_pickle(checkpoint_file)
        print(f"Loaded checkpoint: {checkpoint_file}")
        print(f"Recovered {len(df)} completed departures.")
        return df
    return pd.DataFrame()


def build_checkpoint_filename(out_dir, departure_start, departure_end, departure_freq):
    start_tag = pd.Timestamp(departure_start).strftime("%Y%m%d")
    end_tag = pd.Timestamp(departure_end).strftime("%Y%m%d")
    return Path(out_dir) / f"crossing_results_{start_tag}_{end_tag}_{departure_freq}.pkl"


def coord_tag(x):
    """
    Turn a coordinate into a filename-safe string.
    Example:
    -19.0325 -> m19p03
     32.0785 -> p32p08
    """
    sign = "m" if x < 0 else "p"
    x_abs = abs(x)
    return f"{sign}{x_abs:.2f}".replace(".", "p")


def build_leg_boxes(route_points, margin_lon=3.0, margin_lat=3.0):
    """
    Build one bounding box per route leg.
    Each leg is between consecutive route points.
    """
    if len(route_points) < 2:
        raise ValueError("Need at least two route points.")

    boxes = []
    for i in range(len(route_points) - 1):
        lon1, lat1 = route_points[i]
        lon2, lat2 = route_points[i + 1]

        box = {
            "leg_index": i,
            "start_point": route_points[i],
            "end_point": route_points[i + 1],
            "min_lon": min(lon1, lon2) - margin_lon,
            "max_lon": max(lon1, lon2) + margin_lon,
            "min_lat": min(lat1, lat2) - (margin_lat * 0.3),
            "max_lat": max(lat1, lat2) + margin_lat,
        }
        boxes.append(box)

    return boxes


def build_leg_file_list(
    prefix,
    route_points,
    out_dir,
    margin_lon,
    margin_lat,
    departure_start,
    departure_end,
    download_extra,
):
    """
    Recreate the expected leg filenames from the same logic as the downloader.
    """
    start_dt = pd.Timestamp(departure_start).strftime("%Y-%m-%dT00:00:00")
    end_dt = (
        pd.Timestamp(departure_end) + pd.Timedelta(days=download_extra)
    ).strftime("%Y-%m-%dT00:00:00")

    start_day_str = pd.Timestamp(start_dt).strftime("%Y%m%d")
    end_day_str = pd.Timestamp(end_dt).strftime("%Y%m%d")

    boxes = build_leg_boxes(
        route_points,
        margin_lon=margin_lon,
        margin_lat=margin_lat,
    )

    leg_files = []
    for box in boxes:
        lon_min_tag = coord_tag(box["min_lon"])
        lon_max_tag = coord_tag(box["max_lon"])
        lat_min_tag = coord_tag(box["min_lat"])
        lat_max_tag = coord_tag(box["max_lat"])

        leg_filename = (
            f"{prefix}_leg_{box['leg_index']:02d}_"
            f"{start_day_str}_{end_day_str}_"
            f"lon_{lon_min_tag}_{lon_max_tag}_"
            f"lat_{lat_min_tag}_{lat_max_tag}.nc"
        )

        leg_path = out_dir / leg_filename
        leg_files.append(leg_path)

    return leg_files, boxes


# ============================================================
# DATA HELPERS
# ============================================================
def get_coord_name(ds, candidates):
    for name in candidates:
        if name in ds.coords:
            return name
        if name in ds.variables:
            return name
    raise KeyError(f"Could not find coordinate among: {candidates}")


def get_var_name(ds, candidates):
    for name in candidates:
        if name in ds.data_vars:
            return name
        if name in ds.variables:
            return name
    raise KeyError(f"Could not find variable among: {candidates}")


def prepare_forcing_dataset(nc_file, var_candidates):
    """
    Open one leg dataset and identify coordinate/variable names.
    Uses no dask chunking for faster repeated point access.
    """
    ds = xr.open_dataset(
        nc_file,
        cache=False,
    ).squeeze(drop=True)

    lon_name = get_coord_name(ds, ["longitude", "lon", "LONGITUDE", "LON"])
    lat_name = get_coord_name(ds, ["latitude", "lat", "LATITUDE", "LAT"])
    time_name = get_coord_name(ds, ["time", "TIME"])

    ds = ds.sortby(lon_name).sortby(lat_name)
    if time_name in ds.coords:
        ds = ds.sortby(time_name)

    time_vals = pd.to_datetime(ds[time_name].values)
    time_array_np = time_vals.values

    var_names = {}
    for key, candidates in var_candidates.items():
        var_names[key] = get_var_name(ds, candidates)

    prepared = {
        "path": str(nc_file),
        "ds": ds,
        "lon_name": lon_name,
        "lat_name": lat_name,
        "time_name": time_name,
        "time_vals": time_vals,
        "time_array_np": time_array_np,
        "lon_min": float(np.min(ds[lon_name].values)),
        "lon_max": float(np.max(ds[lon_name].values)),
        "lat_min": float(np.min(ds[lat_name].values)),
        "lat_max": float(np.max(ds[lat_name].values)),
        "last_data_time": pd.Timestamp(time_vals[-1]),
        "var_names": var_names,
    }
    return prepared


def prepare_wind_dataset(nc_file):
    return prepare_forcing_dataset(
        nc_file,
        {
            "u": ["eastward_wind", "u10", "u"],
            "v": ["northward_wind", "v10", "v"],
        }
    )


def prepare_current_dataset(nc_file):
    return prepare_forcing_dataset(
        nc_file,
        {
            "u": ["uo", "eastward_sea_water_velocity", "u"],
            "v": ["vo", "northward_sea_water_velocity", "v"],
        }
    )


def prepare_wave_dataset(nc_file):
    return prepare_forcing_dataset(
        nc_file,
        {
            "wave": ["VHM0", "swh", "wave_height"],
        }
    )


def get_time_slice(prepared, var_name, current_time, warning_state, label):
    """
    Get one 2D field in time.

    If INTERPOLATE_IN_TIME is True:
        use linear interpolation in time.
    If INTERPOLATE_IN_TIME is False:
        use an exact time match when possible, otherwise nearest neighbour in time.

    Reuses first/last timestamp when needed and warns once per data type.
    """
    ds = prepared["ds"]
    time_name = prepared["time_name"]
    time_vals = prepared["time_vals"]
    time_array_np = prepared["time_array_np"]
    last_data_time = prepared["last_data_time"]

    if len(time_vals) == 1:
        if not warning_state.get(label, False):
            warnings.warn(
                f"Too little {label} data to cover the full crossing. "
                f"Only one timestamp is available, so it will be reused."
            )
            warning_state[label] = True

        return ds[var_name].isel({time_name: 0}), pd.Timestamp(time_vals[0]), True

    if current_time > last_data_time:
        if not warning_state.get(label, False):
            warnings.warn(
                f"Too little {label} data to cover the full crossing. "
                f"Using the last available timestamp for the remaining simulation."
            )
            warning_state[label] = True

        return ds[var_name].isel({time_name: -1}), last_data_time, True

    t64 = np.datetime64(current_time)

    if INTERPOLATE_IN_TIME:
        return ds[var_name].interp(
            {time_name: t64},
            method="linear"
        ), current_time, False

    idx = np.searchsorted(time_array_np, t64)

    if idx < len(time_array_np) and time_array_np[idx] == t64:
        nearest_idx = idx
    elif idx == 0:
        nearest_idx = 0
    elif idx >= len(time_array_np):
        nearest_idx = len(time_array_np) - 1
    else:
        left = time_array_np[idx - 1]
        right = time_array_np[idx]
        nearest_idx = idx - 1 if abs(t64 - left) <= abs(right - t64) else idx

    return ds[var_name].isel({time_name: nearest_idx}), pd.Timestamp(time_vals[nearest_idx]), False


def safe_scalar_from_interp(da, lon_name, lat_name, lon, lat):
    """
    Interpolate a 2D field in space and return a scalar.
    Tries linear first, then nearest.
    """
    try:
        val = da.interp(
            {lon_name: lon, lat_name: lat},
            method="linear"
        ).values.item()

        if np.isnan(val):
            raise ValueError("Linear spatial interpolation returned NaN.")
        return float(val)

    except Exception:
        val = da.interp(
            {lon_name: lon, lat_name: lat},
            method="nearest"
        ).values.item()

        if np.isnan(val):
            raise ValueError("Nearest spatial interpolation returned NaN.")
        return float(val)


def wind_uv_to_speed_and_from_direction(u, v):
    """
    u = eastward wind
    v = northward wind

    Returns:
    - wind speed
    - wind direction FROM which wind comes, in degrees clockwise from north
    """
    speed = np.sqrt(u**2 + v**2)
    dir_to = (np.rad2deg(np.arctan2(u, v)) + 360) % 360
    dir_from = (dir_to + 180) % 360
    return float(speed), float(dir_from)


def compute_wave_safety_score(
    track_df,
    dt_hours,
    moderate_threshold,
    critical_threshold,
    moderate_weight,
    critical_weight,
):
    """
    Safety score based on time spent in moderate and critical wave conditions.
    0 means fully safe.
    """
    if track_df is None or track_df.empty or "wave_height" not in track_df.columns:
        return {
            "safety_score": 0.0,
            "moderate_hours": 0.0,
            "critical_hours": 0.0,
        }

    waves = track_df["wave_height"].to_numpy(dtype=float)

    moderate_mask = (waves >= moderate_threshold) & (waves < critical_threshold)
    critical_mask = waves >= critical_threshold

    moderate_hours = float(np.sum(moderate_mask) * dt_hours)
    critical_hours = float(np.sum(critical_mask) * dt_hours)

    score = (
        moderate_hours * moderate_weight
        + critical_hours * critical_weight
    )

    return {
        "safety_score": float(score),
        "moderate_hours": moderate_hours,
        "critical_hours": critical_hours,
    }


def summarize_track_metrics(track_df, dt_hours):
    """
    Extract summary metrics from one track.
    """
    if track_df is None or track_df.empty:
        return {
            "safety_score": np.nan,
            "moderate_hours": np.nan,
            "critical_hours": np.nan,
            "max_wind_speed": np.nan,
            "max_wave_height": np.nan,
        }

    safety = compute_wave_safety_score(
        track_df=track_df,
        dt_hours=dt_hours,
        moderate_threshold=WAVE_MODERATE_THRESHOLD,
        critical_threshold=WAVE_CRITICAL_THRESHOLD,
        moderate_weight=WAVE_MODERATE_WEIGHT,
        critical_weight=WAVE_CRITICAL_WEIGHT,
    )

    return {
        "safety_score": safety["safety_score"],
        "moderate_hours": safety["moderate_hours"],
        "critical_hours": safety["critical_hours"],
        "max_wind_speed": float(track_df["wind_speed"].max()) if "wind_speed" in track_df.columns else np.nan,
        "max_wave_height": float(track_df["wave_height"].max()) if "wave_height" in track_df.columns else np.nan,
    }


def write_departure_summary_txt(results_df, outfile):
    """
    Write a plain-text summary of all calculated departures.
    """
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    header = (
        f"{'departure_day':<14}"
        f"{'arrival_day':<14}"
        f"{'crossing_days':<16}"
        f"{'safety_score':<14}"
        f"{'moderate_h':<14}"
        f"{'critical_h':<14}"
        f"{'max_wind_mps':<16}"
        f"{'max_wave_m':<14}"
        f"{'status':<12}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for _, row in results_df.iterrows():
        dep_day = pd.Timestamp(row["departure_time"]).strftime("%Y-%m-%d")

        if pd.notnull(row["arrival_time"]):
            arr_day = pd.Timestamp(row["arrival_time"]).strftime("%Y-%m-%d")
        else:
            arr_day = "NA"

        crossing_days = (
            f"{row['crossing_time_days']:.3f}"
            if pd.notnull(row["crossing_time_days"]) else "NA"
        )

        safety_score = (
            f"{row['safety_score']:.3f}"
            if "safety_score" in row and pd.notnull(row["safety_score"]) else "NA"
        )

        moderate_hours = (
            f"{row['moderate_hours']:.3f}"
            if "moderate_hours" in row and pd.notnull(row["moderate_hours"]) else "NA"
        )

        critical_hours = (
            f"{row['critical_hours']:.3f}"
            if "critical_hours" in row and pd.notnull(row["critical_hours"]) else "NA"
        )

        max_wind = (
            f"{row['max_wind_speed']:.3f}"
            if "max_wind_speed" in row and pd.notnull(row["max_wind_speed"]) else "NA"
        )

        max_wave = (
            f"{row['max_wave_height']:.3f}"
            if "max_wave_height" in row and pd.notnull(row["max_wave_height"]) else "NA"
        )

        status = "finished" if bool(row["finished"]) else "failed"

        lines.append(
            f"{dep_day:<14}"
            f"{arr_day:<14}"
            f"{crossing_days:<16}"
            f"{safety_score:<14}"
            f"{moderate_hours:<14}"
            f"{critical_hours:<14}"
            f"{max_wind:<16}"
            f"{max_wave:<14}"
            f"{status:<12}"
        )

    outfile.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved departure summary text file to: {outfile}")


def minmax_normalize(series):
    """
    Min-max normalize a pandas Series.
    If all valid values are equal, returns 0 for valid entries.
    """
    s = pd.to_numeric(series, errors="coerce")
    s_min = s.min(skipna=True)
    s_max = s.max(skipna=True)

    if pd.isna(s_min) or pd.isna(s_max):
        return pd.Series(np.nan, index=series.index)

    if s_max == s_min:
        out = pd.Series(np.nan, index=series.index)
        out[s.notna()] = 0.0
        return out

    return (s - s_min) / (s_max - s_min)


def add_final_score(results_df):
    """
    Add normalized ranking terms and a final weighted score.
    Lower score = better.
    Failed runs get NaN score and are placed last.
    """
    df = results_df.copy()

    finished_mask = df["finished"].astype(bool)

    df["norm_crossing_time"] = np.nan
    df["norm_safety_score"] = np.nan
    df["norm_max_wave"] = np.nan
    df["norm_max_wind"] = np.nan
    df["final_score"] = np.nan

    if finished_mask.any():
        finished_df = df.loc[finished_mask].copy()

        finished_df["norm_crossing_time"] = minmax_normalize(finished_df["crossing_time_days"])
        finished_df["norm_safety_score"] = minmax_normalize(finished_df["safety_score"])
        finished_df["norm_max_wave"] = minmax_normalize(finished_df["max_wave_height"])
        finished_df["norm_max_wind"] = minmax_normalize(finished_df["max_wind_speed"])

        finished_df["final_score"] = (
            FINAL_SCORE_WEIGHT_TIME * finished_df["norm_crossing_time"]
            + FINAL_SCORE_WEIGHT_SAFETY * finished_df["norm_safety_score"]
            + FINAL_SCORE_WEIGHT_WAVE * finished_df["norm_max_wave"]
            + FINAL_SCORE_WEIGHT_WIND * finished_df["norm_max_wind"]
        )

        df.loc[finished_mask, "norm_crossing_time"] = finished_df["norm_crossing_time"]
        df.loc[finished_mask, "norm_safety_score"] = finished_df["norm_safety_score"]
        df.loc[finished_mask, "norm_max_wave"] = finished_df["norm_max_wave"]
        df.loc[finished_mask, "norm_max_wind"] = finished_df["norm_max_wind"]
        df.loc[finished_mask, "final_score"] = finished_df["final_score"]

    return df


def write_ranked_departure_summary_txt(results_df, outfile):
    """
    Write a second txt file sorted by final score (best first).
    Failed runs are listed last.
    """
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    ranked_df = add_final_score(results_df)

    ranked_df = ranked_df.sort_values(
        by=["finished", "final_score", "crossing_time_days"],
        ascending=[False, True, True],
        na_position="last"
    ).reset_index(drop=True)

    lines = []
    lines.append("Final ranking score:")
    lines.append(
        "final_score = "
        f"{FINAL_SCORE_WEIGHT_TIME:.2f} * norm(crossing_time_days) + "
        f"{FINAL_SCORE_WEIGHT_SAFETY:.2f} * norm(safety_score) + "
        f"{FINAL_SCORE_WEIGHT_WAVE:.2f} * norm(max_wave_height) + "
        f"{FINAL_SCORE_WEIGHT_WIND:.2f} * norm(max_wind_speed)"
    )
    lines.append("Lower score is better.")
    lines.append("")

    header = (
        f"{'rank':<6}"
        f"{'departure_day':<14}"
        f"{'arrival_day':<14}"
        f"{'crossing_days':<16}"
        f"{'safety_score':<14}"
        f"{'moderate_h':<14}"
        f"{'critical_h':<14}"
        f"{'max_wind_mps':<16}"
        f"{'max_wave_m':<14}"
        f"{'final_score':<14}"
        f"{'status':<12}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    rank_counter = 1
    for _, row in ranked_df.iterrows():
        dep_day = pd.Timestamp(row["departure_time"]).strftime("%Y-%m-%d")

        if pd.notnull(row["arrival_time"]):
            arr_day = pd.Timestamp(row["arrival_time"]).strftime("%Y-%m-%d")
        else:
            arr_day = "NA"

        crossing_days = f"{row['crossing_time_days']:.3f}" if pd.notnull(row["crossing_time_days"]) else "NA"
        safety_score = f"{row['safety_score']:.3f}" if pd.notnull(row["safety_score"]) else "NA"
        moderate_hours = f"{row['moderate_hours']:.3f}" if pd.notnull(row["moderate_hours"]) else "NA"
        critical_hours = f"{row['critical_hours']:.3f}" if pd.notnull(row["critical_hours"]) else "NA"
        max_wind = f"{row['max_wind_speed']:.3f}" if pd.notnull(row["max_wind_speed"]) else "NA"
        max_wave = f"{row['max_wave_height']:.3f}" if pd.notnull(row["max_wave_height"]) else "NA"
        final_score = f"{row['final_score']:.3f}" if pd.notnull(row["final_score"]) else "NA"
        status = "finished" if bool(row["finished"]) else "failed"

        rank_str = str(rank_counter) if bool(row["finished"]) and pd.notnull(row["final_score"]) else "-"
        if rank_str != "-":
            rank_counter += 1

        lines.append(
            f"{rank_str:<6}"
            f"{dep_day:<14}"
            f"{arr_day:<14}"
            f"{crossing_days:<16}"
            f"{safety_score:<14}"
            f"{moderate_hours:<14}"
            f"{critical_hours:<14}"
            f"{max_wind:<16}"
            f"{max_wave:<14}"
            f"{final_score:<14}"
            f"{status:<12}"
        )

    outfile.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved ranked departure summary text file to: {outfile}")


# ============================================================
# PLOT HELPERS
# ============================================================
def format_annual_date_axis(ax):
    """
    Force annual date axis from Jan 1 to Dec 31 and show full month names.
    """
    year = pd.Timestamp(DEPARTURE_START).year
    ax.set_xlim(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%B"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

def print_crossing_time_statistics(results_df):
    """
    Print brief statistics for successful crossings.
    """
    finished = results_df[results_df["finished"]].copy()

    if finished.empty:
        print("\n===== CROSSING TIME STATISTICS =====")
        print("No finished runs available.")
        return

    finished = finished.sort_values("crossing_time_days").reset_index(drop=True)

    shortest_row = finished.iloc[0]
    longest_row = finished.iloc[-1]

    min_time = float(finished["crossing_time_days"].min())
    max_time = float(finished["crossing_time_days"].max())
    mean_time = float(finished["crossing_time_days"].mean())

    print("\n===== CROSSING TIME STATISTICS =====")
    print(f"Minimum crossing time : {min_time:.3f} days")
    print(f"Maximum crossing time : {max_time:.3f} days")
    print(f"Mean crossing time    : {mean_time:.3f} days")
    print(
        f"Shortest departure day: {pd.Timestamp(shortest_row['departure_time']).date()} "
        f"(arrival: {pd.Timestamp(shortest_row['arrival_time']).date()}, "
        f"{shortest_row['crossing_time_days']:.3f} days)"
    )
    print(
        f"Longest departure day : {pd.Timestamp(longest_row['departure_time']).date()} "
        f"(arrival: {pd.Timestamp(longest_row['arrival_time']).date()}, "
        f"{longest_row['crossing_time_days']:.3f} days)"
    )

# ============================================================
# PLOTS
# ============================================================
def plot_domain_and_tracks_multileg(results_df, route_points=None, boxes=None, max_tracks=1):
    """
    Plot route boxes, fastest track, and slowest finished track.
    """
    fig, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=4)

    if boxes is not None:
        for box in boxes:
            rect = Rectangle(
                (box["min_lon"], box["min_lat"]),
                box["max_lon"] - box["min_lon"],
                box["max_lat"] - box["min_lat"],
                fill=False,
                edgecolor="black",
                linewidth=1.5,
                linestyle=":",
                transform=ccrs.PlateCarree(),
                zorder=5
            )
            ax.add_patch(rect)

            ax.text(
                box["min_lon"] + 0.2,
                box["max_lat"] - 0.2,
                f"Leg {box['leg_index']}",
                transform=ccrs.PlateCarree(),
                fontsize=10,
                color="black",
                zorder=6,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5)
            )

    valid = results_df[results_df["finished"]].copy()

    if not valid.empty:
        best_idx = valid["crossing_time_days"].idxmin()
        worst_idx = valid["crossing_time_days"].idxmax()

        best_track = valid.loc[best_idx, "track"]
        worst_track = valid.loc[worst_idx, "track"]

        ax.plot(
            best_track["lon"],
            best_track["lat"],
            color="red",
            linewidth=3,
            label="Fastest finished track",
            transform=ccrs.PlateCarree(),
            zorder=7
        )

        ax.plot(
            worst_track["lon"],
            worst_track["lat"],
            color="blue",
            linewidth=3,
            label="Slowest finished track",
            transform=ccrs.PlateCarree(),
            zorder=7
        )

        ax.scatter(
            best_track["lon"].iloc[0],
            best_track["lat"].iloc[0],
            color="green",
            s=220,
            edgecolor="black",
            label="Start",
            transform=ccrs.PlateCarree(),
            zorder=8
        )

        ax.scatter(
            best_track["lon"].iloc[-1],
            best_track["lat"].iloc[-1],
            color="purple",
            s=220,
            edgecolor="black",
            label="End",
            transform=ccrs.PlateCarree(),
            zorder=8
        )

    if route_points is not None and len(route_points) >= 2:
        rp_lons = [p[0] for p in route_points]
        rp_lats = [p[1] for p in route_points]

        ax.plot(
            rp_lons,
            rp_lats,
            linestyle=":",
            linewidth=2,
            marker="s",
            markersize=6,
            color="darkorange",
            label="Waypoints",
            transform=ccrs.PlateCarree(),
            zorder=6
        )

        ax.set_extent(
            [
                min(rp_lons) - 5,
                max(rp_lons) + 5,
                min(rp_lats) - 13,
                max(rp_lats) + 13
            ],
            crs=ccrs.PlateCarree()
        )

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_crossing_times(results_df):
    if results_df.empty:
        print("Results dataframe is empty.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        results_df["departure_time"],
        results_df["crossing_time_days"],
        marker="o",
        linestyle="-"
    )

    failed = results_df[~results_df["finished"]]
    if not failed.empty:
        ax.scatter(
            failed["departure_time"],
            np.zeros(len(failed)),
            marker="x",
            s=80,
            label="Failed crossing"
        )
        ax.legend()

    ax.set_xlabel("Departure time")
    ax.set_ylabel("Crossing time [days]")
    ax.grid(alpha=0.3)
    format_annual_date_axis(ax)
    plt.tight_layout()
    plt.show()


def plot_final_score_timeseries(results_df, rolling_window_days=14):
    """
    Plot final score over departure date and mark:
    1) best single departure
    2) best rolling-average departure period
    """
    ranked_df = add_final_score(results_df).copy()

    finished = ranked_df[ranked_df["finished"] & ranked_df["final_score"].notna()].copy()

    if finished.empty:
        print("No finished runs with valid final_score available for final score plot.")
        return

    finished = finished.sort_values("departure_time").reset_index(drop=True)
    finished["departure_time"] = pd.to_datetime(finished["departure_time"])

    rolling_mean = (
        finished["final_score"]
        .rolling(window=rolling_window_days, min_periods=rolling_window_days)
        .mean()
    )

    best_idx = finished["final_score"].idxmin()
    best_row = finished.loc[best_idx]

    if rolling_mean.notna().any():
        best_roll_idx = rolling_mean.idxmin()
        best_roll_end = finished.loc[best_roll_idx, "departure_time"]
        best_roll_start = finished.loc[best_roll_idx - rolling_window_days + 1, "departure_time"]
    else:
        best_roll_idx = None
        best_roll_start = None
        best_roll_end = None

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        finished["departure_time"],
        finished["final_score"],
        marker="o",
        linestyle="None",
        label="Final score"
    )

    if rolling_mean.notna().any():
        ax.plot(
            finished["departure_time"],
            rolling_mean,
            linewidth=2.5,
            label=f"{rolling_window_days}-day rolling mean"
        )

    ax.scatter(
        best_row["departure_time"],
        best_row["final_score"],
        s=160,
        marker="*",
        zorder=5,
        label=f"Best departure: {best_row['departure_time'].date()}"
    )

    if best_roll_start is not None:
        ax.axvspan(
            best_roll_start,
            best_roll_end,
            alpha=0.2,
            label=(
                f"Best {rolling_window_days}-day period: "
                f"{best_roll_start.date()} to {best_roll_end.date()}"
            )
        )

    ax.set_xlabel("Departure date")
    ax.set_ylabel("Final score [-]")
    ax.grid(alpha=0.3)
    ax.legend()
    format_annual_date_axis(ax)
    plt.tight_layout()
    plt.show()


def print_fastest_departure(results_df):
    """
    Print fastest successful departure by crossing time.
    """
    finished = results_df[results_df["finished"]].copy()
    if finished.empty:
        print("No finished runs available.")
        return

    fastest_idx = finished["crossing_time_days"].idxmin()
    fastest_row = finished.loc[fastest_idx]

    print("\n===== FASTEST SUCCESSFUL DEPARTURE =====")
    print(
        f"Departure: {pd.Timestamp(fastest_row['departure_time']).date()} | "
        f"Arrival: {pd.Timestamp(fastest_row['arrival_time']).date()} | "
        f"Crossing time: {fastest_row['crossing_time_days']:.3f} days"
    )


def print_best_departure_and_period(results_df, rolling_window_days=14):
    """
    Print best single departure and best rolling-average period.
    """
    ranked_df = add_final_score(results_df).copy()

    finished = ranked_df[ranked_df["finished"] & ranked_df["final_score"].notna()].copy()

    if finished.empty:
        print("No finished runs with valid final_score available.")
        return

    finished = finished.sort_values("departure_time").reset_index(drop=True)
    finished["departure_time"] = pd.to_datetime(finished["departure_time"])

    rolling_mean = (
        finished["final_score"]
        .rolling(window=rolling_window_days, min_periods=rolling_window_days)
        .mean()
    )

    best_idx = finished["final_score"].idxmin()
    best_row = finished.loc[best_idx]

    print("\n===== BEST SINGLE DEPARTURE =====")
    print(
        f"Departure: {best_row['departure_time'].date()} | "
        f"Final score: {best_row['final_score']:.3f} | "
        f"Crossing time: {best_row['crossing_time_days']:.3f} days | "
        f"Safety score: {best_row['safety_score']:.3f}"
    )

    if rolling_mean.notna().any():
        best_roll_idx = rolling_mean.idxmin()
        best_roll_end = finished.loc[best_roll_idx, "departure_time"]
        best_roll_start = finished.loc[best_roll_idx - rolling_window_days + 1, "departure_time"]
        best_roll_value = rolling_mean.loc[best_roll_idx]

        print(f"\n===== BEST {rolling_window_days}-DAY AVERAGE DEPARTURE PERIOD =====")
        print(
            f"From {best_roll_start.date()} to {best_roll_end.date()} | "
            f"Mean final score: {best_roll_value:.3f}"
        )


def plot_track_speed_and_wave_profile(track_df, title_prefix="Track"):
    if track_df is None or track_df.empty:
        print(f"{title_prefix}: empty track.")
        return

    time_days = (pd.to_datetime(track_df["time"]) - pd.to_datetime(track_df["time"].iloc[0])).dt.total_seconds() / 86400.0

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    axes[0].plot(time_days, track_df["boat_speed"], label="Boat speed through water")
    axes[0].plot(time_days, track_df["current_speed"], label="Current speed")
    axes[0].plot(time_days, track_df["total_speed"], label="Total ground speed")
    axes[0].set_ylabel("Speed [m/s]")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(time_days, track_df["wave_height"], label="Wave height")
    axes[1].set_xlabel("Elapsed time since departure [days]")
    axes[1].set_ylabel("Wave height [m]")
    axes[1].grid(alpha=0.3)
    
    max_time = np.nanmax(time_days)
    axes[1].set_xlim(0, max_time)

    plt.tight_layout()
    plt.show()


def plot_variable_heatmap(
    results_df,
    column_name,
    colorbar_label,
    title,
    dt_hours=1.0,
    max_plot_days=None,
    time_downward=True
):
    if results_df.empty:
        print("Results dataframe is empty.")
        return

    valid = results_df[
        results_df["track"].apply(lambda x: isinstance(x, pd.DataFrame) and not x.empty and column_name in x.columns)
    ].copy()

    if valid.empty:
        print(f"No valid tracks found for heatmap column '{column_name}'.")
        return

    max_steps_found = max(len(track) for track in valid["track"])

    if max_plot_days is not None:
        max_steps = int(np.floor(max_plot_days * 24 / dt_hours)) + 1
        max_steps = min(max_steps, max_steps_found)
    else:
        max_steps = max_steps_found

    n_departures = len(valid)
    value_grid = np.full((max_steps, n_departures), np.nan)

    departure_times = pd.to_datetime(valid["departure_time"]).reset_index(drop=True)

    for j, (_, row) in enumerate(valid.reset_index(drop=True).iterrows()):
        track = row["track"]
        n = min(len(track), max_steps)
        value_grid[:n, j] = track[column_name].values[:n]

    elapsed_days = np.arange(max_steps) * dt_hours / 24.0
    x_num = mdates.date2num(departure_times)

    if len(x_num) > 1:
        dx = np.median(np.diff(x_num))
    else:
        dx = 1.0

    x_edges = np.concatenate(([x_num[0] - dx / 2], x_num + dx / 2))
    dy = dt_hours / 24.0
    y_edges = np.concatenate(([elapsed_days[0] - dy / 2], elapsed_days + dy / 2))

    fig, ax = plt.subplots(figsize=(12, 6))

    pcm = ax.pcolormesh(
        x_edges,
        y_edges,
        value_grid,
        shading="auto"
    )

    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_xlabel("Departure date")
    ax.set_ylabel("Elapsed time since departure [days]")

    ax.xaxis_date()
    fig.autofmt_xdate()

    if time_downward:
        ax.invert_yaxis()

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN CROSSING FUNCTION
# ============================================================
def crossing_time_route_multileg(
    prepared_wind_legs,
    prepared_current_legs,
    prepared_wave_legs,
    route_points,
    departure_time,
    dt_hours=1.0,
    max_days=60,
    waypoint_tolerance_m=5000,
):
    if len(route_points) < 2:
        return {
            "finished": False,
            "message": "route_points must contain at least a start and an end point.",
            "track": pd.DataFrame(),
            "crossing_time_hours": np.nan,
            "crossing_time_days": np.nan,
            "arrival_time": None,
            "used_last_timestamp": False,
        }

    n_legs = len(route_points) - 1
    if not (
        len(prepared_wind_legs) == n_legs and
        len(prepared_current_legs) == n_legs and
        len(prepared_wave_legs) == n_legs
    ):
        return {
            "finished": False,
            "message": "Prepared wind/current/wave datasets must match number of route legs.",
            "track": pd.DataFrame(),
            "crossing_time_hours": np.nan,
            "crossing_time_days": np.nan,
            "arrival_time": None,
            "used_last_timestamp": False,
        }

    departure_time = pd.Timestamp(departure_time)

    lon, lat = route_points[0]
    current_target_idx = 1

    dt_seconds = dt_hours * 3600.0
    max_steps = int(max_days * 24 / dt_hours)

    used_last_timestamp = False
    warning_state = {"wind": False, "current": False, "wave": False}
    track = []
    total_time_hours = 0.0

    dt_timedelta = np.timedelta64(int(dt_hours * 3600), "s")
    current_time_np = np.datetime64(departure_time)

    for step in range(max_steps):
        current_time = pd.Timestamp(current_time_np)

        leg_idx = current_target_idx - 1
        wind_prepared = prepared_wind_legs[leg_idx]
        current_prepared = prepared_current_legs[leg_idx]
        wave_prepared = prepared_wave_legs[leg_idx]

        wind_lon_name = wind_prepared["lon_name"]
        wind_lat_name = wind_prepared["lat_name"]
        wind_u_name = wind_prepared["var_names"]["u"]
        wind_v_name = wind_prepared["var_names"]["v"]

        current_lon_name = current_prepared["lon_name"]
        current_lat_name = current_prepared["lat_name"]
        current_u_name = current_prepared["var_names"]["u"]
        current_v_name = current_prepared["var_names"]["v"]

        wave_lon_name = wave_prepared["lon_name"]
        wave_lat_name = wave_prepared["lat_name"]
        wave_name = wave_prepared["var_names"]["wave"]

        lon_target, lat_target = route_points[current_target_idx]
        remaining_distance = haversine_distance(lon, lat, lon_target, lat_target)

        if remaining_distance < waypoint_tolerance_m:
            if current_target_idx == len(route_points) - 1:
                arrival_time = current_time
                return {
                    "crossing_time_hours": total_time_hours,
                    "crossing_time_days": total_time_hours / 24.0,
                    "arrival_time": arrival_time,
                    "used_last_timestamp": used_last_timestamp,
                    "track": pd.DataFrame(track),
                    "finished": True,
                    "message": "Route completed.",
                }
            current_target_idx += 1
            continue

        lon_min = wind_prepared["lon_min"]
        lon_max = wind_prepared["lon_max"]
        lat_min = wind_prepared["lat_min"]
        lat_max = wind_prepared["lat_max"]

        if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
            return {
                "finished": False,
                "message": (
                    f"Boat position left leg {leg_idx} domain at "
                    f"lon={lon:.3f}, lat={lat:.3f}."
                ),
                "track": pd.DataFrame(track),
                "crossing_time_hours": np.nan,
                "crossing_time_days": np.nan,
                "arrival_time": None,
                "used_last_timestamp": used_last_timestamp,
            }

        boat_bearing = initial_bearing(lon, lat, lon_target, lat_target)

        wind_u_slice, wind_sample_time, wind_used_last = get_time_slice(
            wind_prepared, wind_u_name, current_time, warning_state, "wind"
        )
        wind_v_slice, _, _ = get_time_slice(
            wind_prepared, wind_v_name, current_time, warning_state, "wind"
        )
        used_last_timestamp = used_last_timestamp or wind_used_last

        try:
            wind_u = safe_scalar_from_interp(
                wind_u_slice, wind_lon_name, wind_lat_name, lon, lat
            )
            wind_v = safe_scalar_from_interp(
                wind_v_slice, wind_lon_name, wind_lat_name, lon, lat
            )
        except Exception as exc:
            return {
                "finished": False,
                "message": (
                    f"Wind interpolation failed on leg {leg_idx} at "
                    f"lon={lon:.3f}, lat={lat:.3f}, time={wind_sample_time}. Error: {exc}"
                ),
                "track": pd.DataFrame(track),
                "crossing_time_hours": np.nan,
                "crossing_time_days": np.nan,
                "arrival_time": None,
                "used_last_timestamp": used_last_timestamp,
            }

        current_u_slice, current_sample_time, current_used_last = get_time_slice(
            current_prepared, current_u_name, current_time, warning_state, "current"
        )
        current_v_slice, _, _ = get_time_slice(
            current_prepared, current_v_name, current_time, warning_state, "current"
        )
        used_last_timestamp = used_last_timestamp or current_used_last

        try:
            current_u = safe_scalar_from_interp(
                current_u_slice, current_lon_name, current_lat_name, lon, lat
            )
            current_v = safe_scalar_from_interp(
                current_v_slice, current_lon_name, current_lat_name, lon, lat
            )
        except Exception as exc:
            return {
                "finished": False,
                "message": (
                    f"Current interpolation failed on leg {leg_idx} at "
                    f"lon={lon:.3f}, lat={lat:.3f}, time={current_sample_time}. Error: {exc}"
                ),
                "track": pd.DataFrame(track),
                "crossing_time_hours": np.nan,
                "crossing_time_days": np.nan,
                "arrival_time": None,
                "used_last_timestamp": used_last_timestamp,
            }

        wave_slice, wave_sample_time, wave_used_last = get_time_slice(
            wave_prepared, wave_name, current_time, warning_state, "wave"
        )
        used_last_timestamp = used_last_timestamp or wave_used_last

        try:
            wave_height = safe_scalar_from_interp(
                wave_slice, wave_lon_name, wave_lat_name, lon, lat
            )
        except Exception as exc:
            return {
                "finished": False,
                "message": (
                    f"Wave interpolation failed on leg {leg_idx} at "
                    f"lon={lon:.3f}, lat={lat:.3f}, time={wave_sample_time}. Error: {exc}"
                ),
                "track": pd.DataFrame(track),
                "crossing_time_hours": np.nan,
                "crossing_time_days": np.nan,
                "arrival_time": None,
                "used_last_timestamp": used_last_timestamp,
            }

        wind_speed, wind_dir_from = wind_uv_to_speed_and_from_direction(wind_u, wind_v)

        boat_speed = hr50_boat_speed(
            wind_speed,
            wind_dir_from,
            boat_bearing,
            low_angle_mode=LOW_ANGLE_EXTRAPOLATING,
            high_angle_mode=HIGH_ANGLE_EXTRAPOLATING
        )

        if np.isnan(boat_speed):
            return {
                "finished": False,
                "message": (
                    f"Boat speed became NaN on leg {leg_idx} at "
                    f"lon={lon:.3f}, lat={lat:.3f}, time={wind_sample_time}."
                ),
                "track": pd.DataFrame(track),
                "crossing_time_hours": np.nan,
                "crossing_time_days": np.nan,
                "arrival_time": None,
                "used_last_timestamp": used_last_timestamp,
            }

        boat_speed = max(float(boat_speed), 0.05)

        boat_u, boat_v = speed_bearing_to_uv(boat_speed, boat_bearing)

        total_u = boat_u + current_u
        total_v = boat_v + current_v
        total_speed, total_bearing = uv_to_speed_and_bearing(total_u, total_v)
        current_speed = float(np.sqrt(current_u**2 + current_v**2))

        travel_distance = min(total_speed * dt_seconds, remaining_distance)

        if travel_distance > 0:
            new_lon, new_lat = destination_point(lon, lat, total_bearing, travel_distance)
        else:
            new_lon, new_lat = lon, lat

        track.append(
            {
                "step": step,
                "leg_idx": leg_idx,
                "time": current_time,
                "sample_time_used_wind": wind_sample_time,
                "sample_time_used_current": current_sample_time,
                "sample_time_used_wave": wave_sample_time,
                "lon": lon,
                "lat": lat,
                "target_lon": lon_target,
                "target_lat": lat_target,
                "target_index": current_target_idx,
                "boat_bearing_deg": boat_bearing,
                "ground_bearing_deg": total_bearing,
                "wind_u": wind_u,
                "wind_v": wind_v,
                "wind_speed": wind_speed,
                "wind_dir_from_deg": wind_dir_from,
                "boat_u": boat_u,
                "boat_v": boat_v,
                "boat_speed": boat_speed,
                "current_u": current_u,
                "current_v": current_v,
                "current_speed": current_speed,
                "total_u": total_u,
                "total_v": total_v,
                "total_speed": total_speed,
                "wave_height": wave_height,
                "travel_distance_m": travel_distance,
                "remaining_distance_m": remaining_distance,
            }
        )

        lon, lat = new_lon, new_lat
        total_time_hours += dt_hours
        current_time_np = current_time_np + dt_timedelta

    return {
        "crossing_time_hours": np.nan,
        "crossing_time_days": np.nan,
        "arrival_time": None,
        "used_last_timestamp": used_last_timestamp,
        "track": pd.DataFrame(track),
        "finished": False,
        "message": "Route not completed within max_days.",
    }


def crossing_times_for_departure_range_multileg(
    wind_leg_files,
    current_leg_files,
    wave_leg_files,
    route_points,
    departure_start,
    departure_end,
    departure_freq="1D",
    dt_hours=1.0,
    max_days=60,
    waypoint_tolerance_m=5000,
    checkpoint_file=None,
):
    departure_dates = pd.date_range(
        start=pd.Timestamp(departure_start),
        end=pd.Timestamp(departure_end),
        freq=departure_freq
    )

    if len(departure_dates) == 0:
        return pd.DataFrame()

    if checkpoint_file is None:
        checkpoint_file = build_checkpoint_filename(
            OUT_DIR,
            departure_start,
            departure_end,
            departure_freq
        )

    checkpoint_file = Path(checkpoint_file)

    print(f"Running {len(departure_dates)} departures...")
    print(f"From {departure_dates[0]} to {departure_dates[-1]}")
    print(f"Checkpoint file: {checkpoint_file}")

    existing_results = load_checkpoint(checkpoint_file)

    completed_departures = set()
    if not existing_results.empty and "departure_time" in existing_results.columns:
        completed_departures = set(pd.to_datetime(existing_results["departure_time"]))

    all_results = []
    if not existing_results.empty:
        all_results = existing_results.to_dict("records")

    prepared_wind_legs = [prepare_wind_dataset(fp) for fp in wind_leg_files]
    prepared_current_legs = [prepare_current_dataset(fp) for fp in current_leg_files]
    prepared_wave_legs = [prepare_wave_dataset(fp) for fp in wave_leg_files]

    try:
        for i, dep_time in enumerate(departure_dates, start=1):
            dep_time = pd.Timestamp(dep_time)

            if dep_time in completed_departures:
                print(f"\n[{i}/{len(departure_dates)}] Departure: {dep_time} -> already done, skipping")
                continue

            print(f"\n[{i}/{len(departure_dates)}] Departure: {dep_time}")

            result = crossing_time_route_multileg(
                prepared_wind_legs=prepared_wind_legs,
                prepared_current_legs=prepared_current_legs,
                prepared_wave_legs=prepared_wave_legs,
                route_points=route_points,
                departure_time=dep_time,
                dt_hours=dt_hours,
                max_days=max_days,
                waypoint_tolerance_m=waypoint_tolerance_m,
            )

            track_df = result.get("track", pd.DataFrame())
            metrics = summarize_track_metrics(track_df, dt_hours=dt_hours)

            row = {
                "departure_time": dep_time,
                "finished": result["finished"],
                "message": result.get("message", ""),
                "crossing_time_hours": result.get("crossing_time_hours", np.nan),
                "crossing_time_days": result.get("crossing_time_days", np.nan),
                "arrival_time": result.get("arrival_time", None),
                "used_last_timestamp": result.get("used_last_timestamp", False),
                "safety_score": metrics["safety_score"],
                "moderate_hours": metrics["moderate_hours"],
                "critical_hours": metrics["critical_hours"],
                "max_wind_speed": metrics["max_wind_speed"],
                "max_wave_height": metrics["max_wave_height"],
                "track": track_df,
            }

            all_results.append(row)

            if (i % CHECKPOINT_EVERY == 0) or (i == len(departure_dates)):
                temp_df = pd.DataFrame(all_results)
                temp_df = temp_df.sort_values("departure_time").reset_index(drop=True)

                save_checkpoint(temp_df, checkpoint_file)
                print(f"Saved checkpoint after departure {dep_time}")

    finally:
        for prepared in prepared_wind_legs:
            prepared["ds"].close()
        for prepared in prepared_current_legs:
            prepared["ds"].close()
        for prepared in prepared_wave_legs:
            prepared["ds"].close()

    final_df = pd.DataFrame(all_results)
    final_df = final_df.sort_values("departure_time").reset_index(drop=True)
    save_checkpoint(final_df, checkpoint_file)

    return final_df


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    effective_start, effective_end = get_effective_departure_window(
        DEPARTURE_START,
        DEPARTURE_END,
        test_run=TEST_RUN,
        test_run_days=TEST_RUN_DAYS
    )

    wind_leg_files, boxes = build_leg_file_list(
        prefix="wind",
        route_points=ROUTE_POINTS,
        out_dir=OUT_DIR,
        margin_lon=LEG_MARGIN_LON,
        margin_lat=LEG_MARGIN_LAT,
        departure_start=DEPARTURE_START,
        departure_end=DEPARTURE_END,
        download_extra=DOWNLOAD_EXTRA,
    )

    current_leg_files, _ = build_leg_file_list(
        prefix="current",
        route_points=ROUTE_POINTS,
        out_dir=OUT_DIR,
        margin_lon=LEG_MARGIN_LON,
        margin_lat=LEG_MARGIN_LAT,
        departure_start=DEPARTURE_START,
        departure_end=DEPARTURE_END,
        download_extra=DOWNLOAD_EXTRA,
    )

    wave_leg_files, _ = build_leg_file_list(
        prefix="wave",
        route_points=ROUTE_POINTS,
        out_dir=OUT_DIR,
        margin_lon=LEG_MARGIN_LON,
        margin_lat=LEG_MARGIN_LAT,
        departure_start=DEPARTURE_START,
        departure_end=DEPARTURE_END,
        download_extra=DOWNLOAD_EXTRA,
    )

    print("\nExpected wind leg files:")
    for fp in wind_leg_files:
        print(fp)

    print("\nExpected current leg files:")
    for fp in current_leg_files:
        print(fp)

    print("\nExpected wave leg files:")
    for fp in wave_leg_files:
        print(fp)

    checkpoint_file = build_checkpoint_filename(
        OUT_DIR,
        effective_start,
        effective_end,
        DEPARTURE_FREQ
    )

    results_df = crossing_times_for_departure_range_multileg(
        wind_leg_files=wind_leg_files,
        current_leg_files=current_leg_files,
        wave_leg_files=wave_leg_files,
        route_points=ROUTE_POINTS,
        departure_start=effective_start,
        departure_end=effective_end,
        departure_freq=DEPARTURE_FREQ,
        dt_hours=DT_HOURS,
        max_days=MAX_DAYS,
        waypoint_tolerance_m=WAYPOINT_TOLERANCE_M,
        checkpoint_file=checkpoint_file,
    )

    sanity_check_strict_order(results_df)

    print("\n===== STATUS SUMMARY =====")
    print(
        results_df[
            ["departure_time", "crossing_time_days", "arrival_time", "finished", "message"]
        ].to_string(index=False)
    )

    summary_txt_file = OUT_DIR / f"departure_summary_{pd.Timestamp(effective_start).strftime('%Y%m%d')}_{pd.Timestamp(effective_end).strftime('%Y%m%d')}.txt"
    write_departure_summary_txt(results_df, summary_txt_file)

    ranked_summary_txt_file = OUT_DIR / f"departure_summary_ranked_{pd.Timestamp(effective_start).strftime('%Y%m%d')}_{pd.Timestamp(effective_end).strftime('%Y%m%d')}.txt"
    write_ranked_departure_summary_txt(results_df, ranked_summary_txt_file)

    plot_crossing_times(results_df)

    finished = results_df[results_df["finished"]].copy()

    if not finished.empty:
        fastest_idx = finished["crossing_time_days"].idxmin()
        slowest_idx = finished["crossing_time_days"].idxmax()

        fastest_row = finished.loc[fastest_idx]
        slowest_row = finished.loc[slowest_idx]

        plot_track_speed_and_wave_profile(
            fastest_row["track"],
            title_prefix=f"Fastest finished run ({pd.Timestamp(fastest_row['departure_time']).date()})"
        )

        plot_track_speed_and_wave_profile(
            slowest_row["track"],
            title_prefix=f"Slowest finished run ({pd.Timestamp(slowest_row['departure_time']).date()})"
        )
    else:
        print("No finished runs available for fastest/slowest profile plots.")

    ranked_df = add_final_score(results_df)
    ranked_finished = ranked_df[ranked_df["finished"] & ranked_df["final_score"].notna()].copy()

    if not ranked_finished.empty:
        best_rank_idx = ranked_finished["final_score"].idxmin()
        best_rank_row = ranked_finished.loc[best_rank_idx]

        plot_track_speed_and_wave_profile(
            best_rank_row["track"],
            title_prefix=f"Highest ranking departure ({pd.Timestamp(best_rank_row['departure_time']).date()})"
        )
        # Lowest ranking (worst final score)
        worst_rank_idx = ranked_finished["final_score"].idxmax()
        worst_rank_row = ranked_finished.loc[worst_rank_idx]
        
        plot_track_speed_and_wave_profile(
            worst_rank_row["track"],
            title_prefix=f"Lowest ranking departure ({pd.Timestamp(worst_rank_row['departure_time']).date()})"
        )

    plot_domain_and_tracks_multileg(
        results_df=results_df,
        route_points=ROUTE_POINTS,
        boxes=boxes,
        max_tracks=PLOT_MAX_TRACKS
    )

    plot_variable_heatmap(
        results_df=results_df,
        column_name="boat_speed",
        colorbar_label="Boat speed [m/s]",
        title="Boat speed heatmap by departure date",
        dt_hours=DT_HOURS,
        max_plot_days=HEATMAP_MAX_PLOT_DAYS,
        time_downward=HEATMAP_TIME_DOWNWARD
    )

    plot_variable_heatmap(
        results_df=results_df,
        column_name="current_speed",
        colorbar_label="Current speed [m/s]",
        title="Current speed heatmap by departure date",
        dt_hours=DT_HOURS,
        max_plot_days=HEATMAP_MAX_PLOT_DAYS,
        time_downward=HEATMAP_TIME_DOWNWARD
    )

    plot_variable_heatmap(
        results_df=results_df,
        column_name="total_speed",
        colorbar_label="Total speed [m/s]",
        title="Total ground speed heatmap by departure date",
        dt_hours=DT_HOURS,
        max_plot_days=HEATMAP_MAX_PLOT_DAYS,
        time_downward=HEATMAP_TIME_DOWNWARD
    )

    plot_variable_heatmap(
        results_df=results_df,
        column_name="wave_height",
        colorbar_label="Wave height [m]",
        title="Wave height heatmap by departure date",
        dt_hours=DT_HOURS,
        max_plot_days=HEATMAP_MAX_PLOT_DAYS,
        time_downward=HEATMAP_TIME_DOWNWARD
    )

    ranked_preview = add_final_score(results_df)
    ranked_preview = ranked_preview.sort_values(
        by=["finished", "final_score", "crossing_time_days"],
        ascending=[False, True, True],
        na_position="last"
    )

    print_fastest_departure(results_df)

    print("\n===== BEST DEPARTURES BY FINAL SCORE =====")
    print(
        ranked_preview[
            [
                "departure_time",
                "arrival_time",
                "crossing_time_days",
                "safety_score",
                "max_wind_speed",
                "max_wave_height",
                "final_score",
                "finished",
            ]
        ].head(10).to_string(index=False)
    )

    plot_final_score_timeseries(results_df, rolling_window_days=14)
    print_best_departure_and_period(results_df, rolling_window_days=14)
    
    
    print_crossing_time_statistics(results_df)