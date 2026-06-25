
"""
processNew_no_gui.py

- ไม่มี GUI
- ใช้ main_pipeline(data) แทน main()
- บันทึกผลเป็น geojson, json summary, CSV และ shapefile
"""

import os
import logging
import json
import shapefile
import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import mapping, LineString, Point

import networkx as nx

# ======== Placeholder for custom functions ========
# extractMeterData, extractLineData, auto_determine_snap_tolerance, etc.
# ให้แทรกที่นี่หากแยก module
# ================================================

def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def save_geojson(filepath, features):
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

def get_voltage_text(idx, result_df, phases):
    connected_phases = phases[idx].upper().strip()
    voltage_text = ''
    for ph in ['A', 'B', 'C']:
        if ph in connected_phases:
            colname = f'Final Voltage {ph} (V)'
            if colname in result_df.columns:
                vval = result_df.iloc[idx][colname]
                voltage_text += f'{ph}:{vval:.1f}V\n'
            else:
                voltage_text += f'{ph}:N/A\n'
    return voltage_text.strip()

def process_shapefiles(project_id, output_dir, result_df, meterLocations, phases,
                       group1_indices, group2_indices,
                       initialTransformerLocation, splitting_point_coords,
                       optimizedTransformerLocationGroup1, optimizedTransformerLocationGroup2):
    ensure_folder_exists(output_dir)

    proj_tm3 = pyproj.CRS.from_proj4("+proj=utm +zone=47 +datum=WGS84 +units=m +no_defs")
    proj_wgs84 = pyproj.CRS("EPSG:4326")
    transformer = pyproj.Transformer.from_crs(proj_tm3, proj_wgs84, always_xy=True)

    features = []
    for idx in group1_indices:
        x, y = meterLocations[idx]
        lon, lat = transformer.transform(x, y)
        voltage_text = get_voltage_text(idx, result_df, phases)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"group": 1, "voltage_text": voltage_text}
        })
    for idx in group2_indices:
        x, y = meterLocations[idx]
        lon, lat = transformer.transform(x, y)
        voltage_text = get_voltage_text(idx, result_df, phases)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"group": 2, "voltage_text": voltage_text}
        })
    save_geojson(os.path.join(output_dir, "meter_groups.geojson"), features)

    tf_features = []
    if initialTransformerLocation is not None:
        lon, lat = transformer.transform(*initialTransformerLocation)
        tf_features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"name": "Initial Transformer"}
        })
    if splitting_point_coords is not None:
        lon, lat = transformer.transform(*splitting_point_coords)
        tf_features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"name": "Splitting Point"}
        })
    if optimizedTransformerLocationGroup1 is not None:
        lon, lat = transformer.transform(*optimizedTransformerLocationGroup1)
        tf_features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"name": "Group 1 Transformer"}
        })
    if optimizedTransformerLocationGroup2 is not None:
        lon, lat = transformer.transform(*optimizedTransformerLocationGroup2)
        tf_features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"name": "Group 2 Transformer"}
        })
    save_geojson(os.path.join(output_dir, "feature_groups.geojson"), tf_features)


