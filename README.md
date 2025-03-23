# Impressed Current Cathodic Protection (ICCP) Simulation

This repository contains a Python simulation for Impressed Current Cathodic Protection systems for pig stations in pipelines.

## Overview

The simulation models the cathodic protection system for a pipeline pig station, including:
- Pig launcher and receiver sections
- Underground pipeline
- Vertical connections
- MMO-Ti anodes placement and performance

The project compares an original design with an optimized design to demonstrate improvements in protection coverage and efficiency.

## Features

- Simulation of original and optimized anode placement designs
- Visualization of potential distribution in soil environment
- Current distribution analysis
- Protection metrics calculation and comparison
- Interactive plots with hover functionality for detailed inspection

## Results

The simulation generates several visualizations:
- Station layout with anode placement
- Soil CP potential distribution
- Current flow patterns
- Protection potential distribution along the pipeline

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- Pandas

## Usage

```python
# Run the simulation and comparison
python "Impressed Current Cathodic Protection pig.py"

# Generate all result files
python generate_results.py
```
Results
Original Design

<img width="1484" alt="Screenshot 2025-03-23 at 4 43 13 pm" src="https://github.com/user-attachments/assets/345b7a1f-1c08-4198-98c3-9300fcf66630" />
<img width="1484" alt="Screenshot 2025-03-23 at 4 43 34 pm" src="https://github.com/user-attachments/assets/7fe737ac-6052-4edc-ac9c-3e540018fbeb" />
<img width="1484" alt="Screenshot 2025-03-23 at 4 44 42 pm" src="https://github.com/user-attachments/assets/ce5f74e7-4a1d-42e0-befb-bfac7e68c930" />
<img width="1483" alt="Screenshot 2025-03-23 at 4 46 15 pm" src="https://github.com/user-attachments/assets/05688d75-df3b-4eee-97e8-b8bd16942dfc" />

Optimized Design

<img width="1483" alt="Screenshot 2025-03-23 at 4 46 30 pm" src="https://github.com/user-attachments/assets/e8f7a5c7-7cd6-4655-82f4-4ad2f8ae6ba6" />
<img width="1483" alt="Screenshot 2025-03-23 at 4 46 40 pm" src="https://github.com/user-attachments/assets/1fdec265-7528-46ec-8520-5aef74330608" />
<img width="1483" alt="Screenshot 2025-03-23 at 4 59 04 pm" src="https://github.com/user-attachments/assets/899c78b6-0a8f-4588-8ba7-91beb089b94f" />
<img width="1483" alt="Screenshot 2025-03-23 at 4 59 58 pm" src="https://github.com/user-attachments/assets/9ee7fa30-f524-44d4-a314-1d65d1428368" />
<img width="1496" alt="Screenshot 2025-03-23 at 5 00 36 pm" src="https://github.com/user-attachments/assets/75e8c949-4091-46fb-9329-c8bdb2513a03" />
Potential distribution Comparison

<img width="1496" alt="Screenshot 2025-03-23 at 5 01 20 pm" src="https://github.com/user-attachments/assets/7018c37f-e595-4f5b-9a8f-9a27c7712bd6" />







