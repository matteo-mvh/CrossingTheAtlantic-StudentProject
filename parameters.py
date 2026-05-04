from pathlib import Path

# ============================================================
# OPTIONAL LOGIN IN SCRIPT FOR COPERNICUS
# Leave as None to use saved credentials / environment variables
# ============================================================
CMEMS_USERNAME = "USERNAME"  # Copernicus Marine username
CMEMS_PASSWORD = "PASSWORD"  # Copernicus Marine password

# ============================================================
# TEST RUN SETTINGS
# ============================================================
TEST_RUN = False           # if True, only simulate a short test period
TEST_RUN_DAYS = 11        # [days] number of departure days to simulate from DEPARTURE_START

# ============================================================
# ROUTE SETTINGS
# IMPORTANT: route points must be (lon, lat)
# ============================================================
ROUTE_POINTS = [
    (-6.3504050, 36.565077),   # Kadyks (start)
    (-16.042313, 28.200391),   # Tenerife (waypoint)
    (-25.627700, 17.172334),   # Cabo Verde (waypoint)
    (-61.507460, 14.847829),   # Martinique (waypoint)
    (-66.736060, 17.811485),   # Puerto Rico (end)
]

# Margin around each route leg bounding box
LEG_MARGIN_LON = 1.5   # [deg] longitude buffer around route legs
LEG_MARGIN_LAT = 2.5   # [deg] latitude buffer around route legs

# ============================================================
# FINAL RANKING SCORE
# Lower score = better departure
# ============================================================
FINAL_SCORE_WEIGHT_TIME   = 0.50
FINAL_SCORE_WEIGHT_SAFETY = 0.25
FINAL_SCORE_WEIGHT_WAVE   = 0.15
FINAL_SCORE_WEIGHT_WIND   = 0.10

# ============================================================
# DEPARTURE SETTINGS
# ============================================================
DEPARTURE_START = "2025-01-01"  # first departure date
DEPARTURE_END   = "2025-12-31"  # last departure date
DEPARTURE_FREQ  = "1D"          # departure interval (daily)

# Number of extra days to download after the last departure day
DOWNLOAD_EXTRA  = 40  # [days] ensures full coverage for late departures

# ============================================================
# OUTPUT SETTINGS
# ============================================================
OUT_DIR = Path("cmems_wind_test_output")  # directory for downloaded data and results
OUTPUT_FILENAME = "north_atlantic_wind_test.nc"  # (optional) combined output filename

# ============================================================
# DATASET SETTINGS
# ============================================================

# WIND DATA (Surface Winds)
DATASET_ID = "cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H"  # wind dataset ID
DATASET_VERSION = None  # specify if needed
VARIABLES = ["eastward_wind", "northward_wind"]  # wind components (u, v)

# WAVE DATA (Significant wave height)
WAVE_DATASET_ID = "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i"  # wave dataset ID
WAVE_DATASET_VERSION = None  # specify if needed
WAVE_VARIABLES = ["VHM0"]  # significant wave height [m]

# OCEAN CURRENTS (Surface currents)
CURRENT_DATASET_ID = "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m"  # current dataset ID
CURRENT_DATASET_VERSION = None  # specify if needed
CURRENT_VARIABLES = ["uo", "vo"]  # current velocity components [m/s]

# Depth selection for currents (surface layer)
CURRENT_DEPTH = 0.49402499198913574  # [m] surface depth level for currents

# Thresholds for significant wave height (Hs = VHM0)
WAVE_MODERATE_THRESHOLD = 4.0   # [m] moderate sea state (reduced comfort)
WAVE_CRITICAL_THRESHOLD = 8.0   # [m] rough conditions (safety concerns)

# Weights for safety score (per hour in each regime)
WAVE_MODERATE_WEIGHT = 1.0      # [-] penalty for moderate waves
WAVE_CRITICAL_WEIGHT = 3.0      # [-] stronger penalty for critical waves

# ============================================================
# DOWNLOAD CONTROL
# ============================================================
FORCE_DOWNLOAD = False  # if True, overwrite existing downloaded files

# ============================================================
# MODEL SETTINGS
# ============================================================
INTERPOLATE_IN_TIME = False  
# True: linear time interpolation, False: use exact/nearest time slice
CHECKPOINT_EVERY = 5   # save every 5 departures
DT_HOURS = 1  # [hours] simulation time step
MAX_DAYS = DOWNLOAD_EXTRA  # [days] maximum allowed crossing duration
WAYPOINT_TOLERANCE_M = 5000  # [m] distance threshold to consider waypoint reached

# Extrapolation options for boat polar diagram
LOW_ANGLE_EXTRAPOLATING = "trend"   # behavior at low wind angles
HIGH_ANGLE_EXTRAPOLATING = "trend"  # behavior at high wind angles

# ============================================================
# PLOT SETTINGS
# ============================================================
PLOT_MAX_TRACKS = 1  # number of trajectories shown in map plots
HEATMAP_MAX_PLOT_DAYS = 100  # max time range shown in heatmaps
HEATMAP_TIME_DOWNWARD = False  # if True, time axis is inverted