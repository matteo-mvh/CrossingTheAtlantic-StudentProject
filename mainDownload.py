import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import copernicusmarine
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle

from parameters import (
    CMEMS_USERNAME,
    CMEMS_PASSWORD,
    OUT_DIR,

    # wind
    DATASET_ID,
    DATASET_VERSION,
    VARIABLES,

    # waves
    WAVE_DATASET_ID,
    WAVE_DATASET_VERSION,
    WAVE_VARIABLES,

    # currents
    CURRENT_DATASET_ID,
    CURRENT_DATASET_VERSION,
    CURRENT_VARIABLES,
    CURRENT_DEPTH,

    # route / timing
    ROUTE_POINTS,
    LEG_MARGIN_LON,
    LEG_MARGIN_LAT,
    DEPARTURE_START,
    DEPARTURE_END,
    DOWNLOAD_EXTRA,
    FORCE_DOWNLOAD,
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

START_DT = pd.Timestamp(DEPARTURE_START).strftime("%Y-%m-%dT00:00:00")
END_DT = (
    pd.Timestamp(DEPARTURE_END) + pd.Timedelta(days=DOWNLOAD_EXTRA)
).strftime("%Y-%m-%dT00:00:00")


def login_if_needed():
    if CMEMS_USERNAME is not None and CMEMS_PASSWORD is not None:
        print("Logging in with username/password from parameters.py ...")
        copernicusmarine.login(
            username=CMEMS_USERNAME,
            password=CMEMS_PASSWORD,
            force_overwrite=True
        )
    else:
        print("No username/password set in parameters.py")
        print("Using existing Copernicus Marine credentials if available.")


def coord_tag(x):
    sign = "m" if x < 0 else "p"
    x_abs = abs(x)
    return f"{sign}{x_abs:.2f}".replace(".", "p")


def guess_coord_name(ds, candidates):
    for name in candidates:
        if name in ds.coords:
            return name
        if name in ds.variables:
            return name
    raise KeyError(f"None of these names found in dataset: {candidates}")


def guess_var_name(ds, candidates):
    for name in candidates:
        if name in ds.data_vars:
            return name
        if name in ds.variables:
            return name
    raise KeyError(f"None of these variable names found in dataset: {candidates}")


def build_leg_boxes(route_points, margin_lon=3.0, margin_lat=3.0):
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


def build_leg_filename(prefix, leg_index, lon_min, lon_max, lat_min, lat_max):
    start_day_str = pd.Timestamp(START_DT).strftime("%Y%m%d")
    end_day_str = pd.Timestamp(END_DT).strftime("%Y%m%d")

    lon_min_tag = coord_tag(lon_min)
    lon_max_tag = coord_tag(lon_max)
    lat_min_tag = coord_tag(lat_min)
    lat_max_tag = coord_tag(lat_max)

    return (
        f"{prefix}_leg_{leg_index:02d}_"
        f"{start_day_str}_{end_day_str}_"
        f"lon_{lon_min_tag}_{lon_max_tag}_"
        f"lat_{lat_min_tag}_{lat_max_tag}.nc"
    )


def existing_file_covers_request(nc_path, requested_start, requested_end):
    if not nc_path.exists():
        return False

    try:
        with xr.open_dataset(nc_path) as ds_existing:
            time_name = guess_coord_name(ds_existing, ["time", "TIME"])
            time_vals = pd.to_datetime(ds_existing[time_name].values)

            existing_start = pd.Timestamp(time_vals.min())
            existing_end = pd.Timestamp(time_vals.max())

            return existing_start <= requested_start and existing_end >= requested_end
    except Exception as exc:
        print(f"Could not validate existing file {nc_path.name}: {exc}")
        return False


def download_dataset_for_leg(
    box,
    prefix,
    dataset_id,
    dataset_version,
    variables,
    force_download=False,
    minimum_depth=None,
    maximum_depth=None,
):
    leg_index = box["leg_index"]

    filename = build_leg_filename(
        prefix=prefix,
        leg_index=leg_index,
        lon_min=box["min_lon"],
        lon_max=box["max_lon"],
        lat_min=box["min_lat"],
        lat_max=box["max_lat"],
    )

    nc_path = OUT_DIR / filename

    requested_start = pd.Timestamp(START_DT)
    requested_end = pd.Timestamp(END_DT)

    if not force_download and existing_file_covers_request(nc_path, requested_start, requested_end):
        print("\n============================================================")
        print(f"Skipping {prefix} leg {leg_index} (already downloaded)")
        print(f"File      : {nc_path}")
        print(f"Time range: {requested_start} to {requested_end}")
        print("============================================================")
        return nc_path

    print("\n============================================================")
    print(f"Downloading {prefix} leg {leg_index}")
    print(f"  Start point: {box['start_point']}")
    print(f"  End point  : {box['end_point']}")
    print(f"  Lon range  : {box['min_lon']} to {box['max_lon']}")
    print(f"  Lat range  : {box['min_lat']} to {box['max_lat']}")
    print(f"  Time range : {START_DT} to {END_DT}")
    if minimum_depth is not None and maximum_depth is not None:
        print(f"  Depth      : {minimum_depth} to {maximum_depth}")
    print(f"  Variables  : {variables}")
    print(f"  File       : {filename}")
    print("============================================================")

    subset_kwargs = dict(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        variables=variables,
        minimum_longitude=box["min_lon"],
        maximum_longitude=box["max_lon"],
        minimum_latitude=box["min_lat"],
        maximum_latitude=box["max_lat"],
        start_datetime=START_DT,
        end_datetime=END_DT,
        coordinates_selection_method="strict-inside",
        output_directory=str(OUT_DIR),
        output_filename=filename,
        file_format="netcdf",
        netcdf_compression_level=1,
        disable_progress_bar=True,
        overwrite=True,
    )

    if minimum_depth is not None:
        subset_kwargs["minimum_depth"] = minimum_depth
    if maximum_depth is not None:
        subset_kwargs["maximum_depth"] = maximum_depth

    response = copernicusmarine.subset(**subset_kwargs)

    print("Download finished.")
    print(response)

    return nc_path


def load_and_prepare_wind(nc_path):
    print(f"\nOpening wind dataset: {nc_path}")
    ds = xr.open_dataset(nc_path)

    print("\nDataset summary:")
    print(ds)

    lon_name = guess_coord_name(ds, ["longitude", "lon", "LONGITUDE", "LON"])
    lat_name = guess_coord_name(ds, ["latitude", "lat", "LATITUDE", "LAT"])
    u_name = guess_var_name(ds, ["eastward_wind", "u10", "u", "eastward"])
    v_name = guess_var_name(ds, ["northward_wind", "v10", "v", "northward"])

    lons = ds[lon_name].values
    lats = ds[lat_name].values

    u = ds[u_name]
    v = ds[v_name]

    if "time" in u.dims:
        u = u.isel(time=0)
    if "time" in v.dims:
        v = v.isel(time=0)

    u = u.squeeze()
    v = v.squeeze()

    return ds, lons, lats, u.values, v.values, u_name, v_name


def plot_flow_field(lons, lats, u, v, u_name, v_name, route_points=None, boxes=None):
    Lon2D, Lat2D = np.meshgrid(lons, lats)

    fig, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=4)

    if boxes is not None:
        for i, box in enumerate(boxes):
            rect = Rectangle(
                (box["min_lon"], box["min_lat"]),
                box["max_lon"] - box["min_lon"],
                box["max_lat"] - box["min_lat"],
                fill=False,
                edgecolor="black",
                linewidth=1.5,
                linestyle=":",
                transform=ccrs.PlateCarree(),
                zorder=6
            )
            ax.add_patch(rect)

            ax.text(
                box["min_lon"] + 0.2,
                box["max_lat"] - 0.2,
                f"Leg {i}",
                transform=ccrs.PlateCarree(),
                fontsize=10,
                color="black",
                zorder=7,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5)
            )

    if route_points is not None and len(route_points) >= 2:
        rp_lons = [p[0] for p in route_points]
        rp_lats = [p[1] for p in route_points]

        ax.plot(
            rp_lons, rp_lats,
            linestyle="--", marker="o", linewidth=2, color="red",
            label="Route waypoints",
            transform=ccrs.PlateCarree(),
            zorder=7
        )
        ax.legend(loc="upper left")

        ax.set_extent(
            [
                min(rp_lons) - 5,
                max(rp_lons) + 5,
                min(rp_lats) - 13,
                max(rp_lats) + 13
            ],
            crs=ccrs.PlateCarree()
        )
    else:
        ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    plt.tight_layout()
    plt.show()


def main():
    login_if_needed()

    boxes = build_leg_boxes(
        ROUTE_POINTS,
        margin_lon=LEG_MARGIN_LON,
        margin_lat=LEG_MARGIN_LAT
    )

    wind_files = []
    wave_files = []
    current_files = []

    for box in boxes:
        wind_path = download_dataset_for_leg(
            box=box,
            prefix="wind",
            dataset_id=DATASET_ID,
            dataset_version=DATASET_VERSION,
            variables=VARIABLES,
            force_download=FORCE_DOWNLOAD,
        )
        wind_files.append(wind_path)

        wave_path = download_dataset_for_leg(
            box=box,
            prefix="wave",
            dataset_id=WAVE_DATASET_ID,
            dataset_version=WAVE_DATASET_VERSION,
            variables=WAVE_VARIABLES,
            force_download=FORCE_DOWNLOAD,
        )
        wave_files.append(wave_path)

        current_path = download_dataset_for_leg(
            box=box,
            prefix="current",
            dataset_id=CURRENT_DATASET_ID,
            dataset_version=CURRENT_DATASET_VERSION,
            variables=CURRENT_VARIABLES,
            force_download=FORCE_DOWNLOAD,
            minimum_depth=CURRENT_DEPTH,
            maximum_depth=CURRENT_DEPTH,
        )
        current_files.append(current_path)

    print("\nDownloaded wind files:")
    for fp in wind_files:
        print(fp)

    print("\nDownloaded wave files:")
    for fp in wave_files:
        print(fp)

    print("\nDownloaded current files:")
    for fp in current_files:
        print(fp)

    # Preview first wind file
    ds, lons, lats, u, v, u_name, v_name = load_and_prepare_wind(wind_files[0])
    plot_flow_field(
        lons, lats, u, v, u_name, v_name,
        route_points=ROUTE_POINTS,
        boxes=boxes
    )
    ds.close()


if __name__ == "__main__":
    main()