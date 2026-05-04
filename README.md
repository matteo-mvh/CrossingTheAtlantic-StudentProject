# CrossingTheAtlantic-StudentProject

This project models and evaluates Atlantic crossings using weather/ocean data and a boat performance model.

## Project Structure

- `parameters.py`  
  Central configuration file. Use this to set:
  - general run parameters,
  - model/grid settings,
  - Copernicus credentials and related download options.

- `Polar_Diagram.py`  
  Contains boat performance data (polar diagram) and helper functions used to estimate sailing performance under different conditions.

- `mainDownload.py`  
  Downloads and prepares required environmental input data. This is the first processing step after parameters are configured.

- `calcAllCrossings.py`  
  Runs crossing calculations using the prepared data and model setup.

## How to Use

Run the workflow in this order:

1. **Set parameters and credentials** in `parameters.py`.
2. **Download data** by running:
   ```bash
   python mainDownload.py
   ```
3. After the download step is complete, **calculate crossings**:
   ```bash
   python calcAllCrossings.py
   ```

## Notes

- Make sure your Copernicus credentials in `parameters.py` are valid before starting downloads.
- `calcAllCrossings.py` depends on outputs created by `mainDownload.py`, so do not skip step 2.
