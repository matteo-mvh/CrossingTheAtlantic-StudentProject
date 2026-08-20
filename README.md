# Atlantic Crossing Route Model

> **Project type:** University Student Project  
> **Field:** Ocean data · weather routing · numerical modelling · sailing performance  
> **Language:** Python  
> **Data:** Copernicus Marine

## Overview

This project models and evaluates **Atlantic sailing crossings** by combining environmental ocean/weather data with a numerical representation of sailing-boat performance.

The workflow downloads environmental forcing, represents vessel performance through a **polar diagram**, and calculates potential Atlantic crossings under different conditions.

The project demonstrates how environmental data and engineering models can be combined to evaluate real-world marine routes.

## Main Idea

The achievable speed and route of a sailing vessel depend strongly on environmental conditions including wind and ocean state.

This project combines:

1. environmental data
2. boat-performance information
3. numerical route calculations

to investigate possible Atlantic crossings.

## Data

Environmental inputs are downloaded using **Copernicus Marine** products.

The data-processing workflow prepares the required model fields before they are used by the route calculations.

Copernicus credentials and download options are configured in `parameters.py`.

## Boat Performance

Boat performance is represented using a **polar diagram**.

A polar diagram describes the expected boat speed as a function of sailing conditions and relative direction.

The implementation is contained in:

```text
Polar_Diagram.py
```

and is used by the crossing model to translate environmental conditions into estimated sailing performance.

## Repository Structure

```text
CrossingTheAtlantic-StudentProject/
├── parameters.py
├── Polar_Diagram.py
├── mainDownload.py
├── calcAllCrossings.py
└── README.md
```

### `parameters.py`

Central configuration file for:

- model parameters
- spatial/grid settings
- data-download settings
- Copernicus configuration
- environmental input settings

### `mainDownload.py`

Downloads and prepares the environmental data required by the model.

This should be run before performing crossing calculations.

### `Polar_Diagram.py`

Contains boat-performance data and functions used to estimate sailing speed under different conditions.

### `calcAllCrossings.py`

Runs the crossing calculations using the prepared environmental data and vessel-performance model.

## Running the Project

### 1. Configure parameters

Set the required configuration in:

```text
parameters.py
```

including valid Copernicus credentials where required.

### 2. Download environmental data

Run:

```bash
python mainDownload.py
```

### 3. Calculate crossings

After the required data have been prepared, run:

```bash
python calcAllCrossings.py
```

## Workflow

```text
Copernicus environmental data
            ↓
      Data preparation
            ↓
     Boat polar diagram
            ↓
  Sailing-performance model
            ↓
   Crossing calculations
            ↓
      Route evaluation
```

## What This Project Demonstrates

The project provides experience in:

- marine environmental data
- Copernicus Marine products
- scientific Python
- automated environmental-data retrieval
- numerical route modelling
- sailing-performance models
- data preprocessing
- spatial ocean data
- combining observations/models with engineering calculations

## Status

Completed university student project.

The repository is intended as an educational modelling project rather than an operational marine-routing service.
