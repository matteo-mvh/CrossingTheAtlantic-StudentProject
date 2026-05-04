import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# HALLBERG-RASSY 50 POLAR DATA
# ============================================================

TWS_GRID = np.array([6, 8, 10, 12, 14, 16, 20, 25, 30], dtype=float)
TWA_GRID = np.array([40, 45, 52, 60, 70, 80, 90, 100, 110, 120, 135, 150], dtype=float)

# Rows = TWA, columns = TWS
BOAT_SPEED_TABLE = np.array([
    [4.47, 5.64, 6.41, 6.90, 7.23, 7.45, 7.69, 7.83, 7.84],   # 40
    [4.99, 6.18, 6.96, 7.43, 7.72, 7.90, 8.12, 8.23, 8.28],   # 45
    [5.55, 6.74, 7.54, 7.94, 8.16, 8.29, 8.51, 8.67, 8.75],   # 52
    [5.99, 7.20, 7.94, 8.30, 8.49, 8.63, 8.85, 9.02, 9.13],   # 60
    [6.30, 7.51, 8.19, 8.57, 8.80, 8.94, 9.17, 9.36, 9.50],   # 70
    [6.55, 7.82, 8.32, 8.69, 8.98, 9.18, 9.43, 9.66, 9.84],   # 80
    [6.89, 8.07, 8.55, 8.82, 9.04, 9.29, 9.66, 9.94, 10.16],  # 90
    [7.00, 8.14, 8.68, 9.00, 9.23, 9.42, 9.76, 10.20, 10.45], # 100
    [6.87, 8.05, 8.66, 9.08, 9.39, 9.60, 9.97, 10.34, 10.73], # 110
    [6.52, 7.81, 8.50, 8.98, 9.36, 9.70, 10.17, 10.60, 10.99],# 120
    [5.66, 7.01, 7.97, 8.54, 8.99, 9.38, 10.15, 10.99, 11.61],# 135
    [4.42, 5.76, 6.88, 7.80, 8.39, 8.84, 9.59, 10.55, 11.53], # 150
], dtype=float)


def _normalize_relative_angle(wind_from_deg: float, boat_heading_deg: float) -> float:
    delta = abs(wind_from_deg - boat_heading_deg) % 360
    if delta > 180:
        delta = 360 - delta
    return delta


def _interp_1d(x, xp, fp):
    return np.interp(np.clip(x, xp[0], xp[-1]), xp, fp)


def _speed_row_interpolated_over_tws(wind_speed):
    """
    For a given wind speed, interpolate the whole TWA-speed curve
    from the table. Returns one speed value for each TWA row.
    """
    tws = np.clip(wind_speed, TWS_GRID[0], TWS_GRID[-1])
    return np.array([
        _interp_1d(tws, TWS_GRID, BOAT_SPEED_TABLE[i, :])
        for i in range(len(TWA_GRID))
    ])


def _linear_extrapolate(x, x1, y1, x2, y2):
    """
    Linear extrapolation using points (x1,y1) and (x2,y2).
    """
    slope = (y2 - y1) / (x2 - x1)
    return y1 + slope * (x - x1)


def hr50_speed_from_tws_twa(
    wind_speed,
    twa,
    low_angle_mode="scale",
    high_angle_mode="trend",
):
    """
    Interpolate/extrapolate boat speed from true wind speed and true wind angle.

    Parameters
    ----------
    wind_speed : float
        True wind speed [kn]
    twa : float
        True wind angle [deg], expected in [0, 180]
    low_angle_mode : str
        Handling for TWA < 40:
        - "zero"  : return 0 below 40
        - "clip"  : use speed at 40
        - "scale" : scale linearly from 0 at 0 deg to speed at 40
        - "trend" : extrapolate using slope from 40 to 45
    high_angle_mode : str
        Handling for TWA > 150:
        - "clip"  : use speed at 150
        - "scale" : scale linearly from speed at 150 to 0 at 180
        - "trend" : extrapolate using slope from 135 to 150

    Returns
    -------
    float
        Estimated boat speed [kn]
    """
    twa = np.clip(twa, 0, 180)
    speeds_vs_twa = _speed_row_interpolated_over_tws(wind_speed)

    # -------------------------
    # BELOW 40 DEG
    # -------------------------
    if twa < TWA_GRID[0]:
        s40 = speeds_vs_twa[0]
        s45 = speeds_vs_twa[1]

        if low_angle_mode == "zero":
            return 0.0

        elif low_angle_mode == "clip":
            return float(s40)

        elif low_angle_mode == "scale":
            return float(s40 * twa / TWA_GRID[0])

        elif low_angle_mode == "trend":
            val = _linear_extrapolate(twa, TWA_GRID[0], s40, TWA_GRID[1], s45)
            return float(max(0.0, val))

        else:
            raise ValueError("low_angle_mode must be 'zero', 'clip', 'scale', or 'trend'")

    # -------------------------
    # ABOVE 150 DEG
    # -------------------------
    if twa > TWA_GRID[-1]:
        s135 = speeds_vs_twa[-2]
        s150 = speeds_vs_twa[-1]

        if high_angle_mode == "clip":
            return float(s150)

        elif high_angle_mode == "scale":
            # linearly reduce to zero at 180
            val = s150 * (180 - twa) / (180 - 150)
            return float(max(0.0, val))

        elif high_angle_mode == "trend":
            val = _linear_extrapolate(twa, TWA_GRID[-2], s135, TWA_GRID[-1], s150)
            return float(max(0.0, val))

        else:
            raise ValueError("high_angle_mode must be 'clip', 'scale', or 'trend'")

    # -------------------------
    # INSIDE TABULATED RANGE
    # -------------------------
    return float(np.interp(twa, TWA_GRID, speeds_vs_twa))

def knots_to_mps(speed_kn):
    return speed_kn * 0.514444

def hr50_boat_speed(
    wind_speed,
    wind_direc_from,
    boat_direc,
    low_angle_mode="scale",
    high_angle_mode="trend",
):
    twa = _normalize_relative_angle(wind_direc_from, boat_direc)

    speed = hr50_speed_from_tws_twa(wind_speed=wind_speed,
                                    twa=twa,
                                    low_angle_mode=low_angle_mode,
                                    high_angle_mode=high_angle_mode,
                                    )
    return knots_to_mps(speed)



def plot_hr50_polar(
    wind_speed=12,
    half=False,
    low_angle_mode="scale",
    high_angle_mode="trend",
):
    if half:
        headings = np.arange(0, 181)
    else:
        headings = np.arange(0, 360)

    wind_from = 0.0
    speeds = np.array([
        hr50_boat_speed(
            wind_speed,
            wind_from,
            hdg,
            low_angle_mode=low_angle_mode,
            high_angle_mode=high_angle_mode,
        )
        for hdg in headings
    ])

    angles_rad = np.deg2rad(headings)

    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles_rad, speeds, linewidth=2)

    if half:
        ax.set_thetamin(0)
        ax.set_thetamax(180)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(
        f"HR50 polar at TWS = {wind_speed} kn\n"
        f"low={low_angle_mode}, high={high_angle_mode}"
    )

    plt.show()

def plot_hr50_cartesian(
    wind_speeds=(6, 10, 14, 20, 30),
    low_angle_mode="trend",
    high_angle_mode="trend",
):
    """
    Horizontal plot: TWA (deg) vs boat speed (kn)
    """

    twa_vals = np.linspace(0, 180, 361)

    plt.figure(figsize=(10, 5))

    for ws in wind_speeds:
        speeds = [
            hr50_speed_from_tws_twa(
                wind_speed=ws,
                twa=twa,
                low_angle_mode=low_angle_mode,
                high_angle_mode=high_angle_mode,
            )
            for twa in twa_vals
        ]

        plt.plot(twa_vals, speeds, label=f"{ws} kn")

    plt.xlabel("True Wind Angle (deg)")
    plt.ylabel("Boat Speed (kn)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim([0,180])

    plt.tight_layout()
    plt.show()
    
def plot_hr50_polars_multiple(
    wind_speeds=(6, 10, 14, 20, 30),
    half=True,
    low_angle_mode="trend",
    high_angle_mode="trend",
):
    headings = np.arange(0, 181 if half else 360)
    angles_rad = np.deg2rad(headings)

    # --- Wider, less tall ---
    fig = plt.figure(figsize=(12, 6))

    # --- Move plot to the right ---
    ax = fig.add_axes([0.35, 0.1, 0.6, 0.8], polar=True)
    # [left, bottom, width, height]

    all_speeds = []

    for ws in wind_speeds:
        speeds = np.array([
            hr50_boat_speed(
                ws,
                wind_direc_from=0.0,
                boat_direc=hdg,
                low_angle_mode=low_angle_mode,
                high_angle_mode=high_angle_mode,
            )
            for hdg in headings
        ])

        all_speeds.append(speeds)
        ax.plot(angles_rad, speeds, label=f"{ws} kn")

    if half:
        ax.set_thetamin(0)
        ax.set_thetamax(180)

    # Rotation
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)

    # Radial ticks
    max_speed = np.nanmax(np.concatenate(all_speeds))
    ticks = np.arange(0, np.ceil(max_speed) + 1, 3)
    ax.set_rticks(ticks)
    ax.set_yticklabels([f"{t:.0f} kn" for t in ticks])

    # Legend sits nicely in left empty space
    ax.legend(
        title="Wind speed [kn]",
        loc="center left",
        bbox_to_anchor=(-0.3, 0.5)
    )

    plt.show()


if __name__ == "__main__":

    plot_hr50_polars_multiple()
    plot_hr50_cartesian()
    
    speed = hr50_boat_speed(
        wind_speed= 60,
        wind_direc_from = 4000 ,
        boat_direc = 340.4,
        low_angle_mode="trend",
        high_angle_mode="trend",
    )
    print(speed)