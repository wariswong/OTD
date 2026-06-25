
def main_pipeline(data):
    import networkx as nx
    import numpy as np
    import pandas as pd
    from shapely.geometry import Point

    # Unpack data from shapefile readers
    meterData = data['meterData']
    lvData = data['lvData']
    mvData = data['mvData']
    transformerData = data['transformerData']
    eserviceData = data['eserviceData']
    project_id = data['project_id']
    output_dir = data['output_dir']

    # 1. Extract data
    meterLocations, initialVoltages, totalLoads, phase_loads, peano, phases = extractMeterData(meterData)
    lvLines = extractLineData(lvData)
    mvLines = extractLineData(mvData)
    eserviceLines = extractLineData(eserviceData)

    # 2. Calculate Snap Tolerance
    tolerance = auto_determine_snap_tolerance(meterLocations)

    # 3. Create Network
    G = create_graph_from_lines(lvLines, tolerance)

    # 4. Find initial Transformer
    initialTransformerLocation = find_initial_transformer_location(transformerData)
    G = connect_meters_to_graph(G, meterLocations, tolerance)

    # 5. Calculate Voltage Profiles
    result_df = calculate_voltage_profiles(G, meterLocations, initialVoltages, totalLoads, phase_loads, phases)

    # 6. Group Meters (Split)
    group1_indices, group2_indices, splitting_point_coords = split_groups_by_voltage(G, meterLocations, result_df)

    # 7. Optimize Transformer Location per Group
    optimizedTransformerLocationGroup1 = optimize_transformer_location([meterLocations[i] for i in group1_indices])
    optimizedTransformerLocationGroup2 = optimize_transformer_location([meterLocations[i] for i in group2_indices])

    # 8. Save GeoJSON
    process_shapefiles(
        project_id,
        output_dir,
        result_df,
        meterLocations,
        phases,
        group1_indices,
        group2_indices,
        initialTransformerLocation,
        splitting_point_coords,
        optimizedTransformerLocationGroup1,
        optimizedTransformerLocationGroup2
    )

    # 9. Save CSV
    result_df.to_csv(os.path.join(output_dir, 'result_meters.csv'), index=False)

    # 10. Return Summary JSON
    return {
        'project_id': project_id,
        'group1_count': len(group1_indices),
        'group2_count': len(group2_indices),
        'group1_tx': list(optimizedTransformerLocationGroup1) if optimizedTransformerLocationGroup1 else None,
        'group2_tx': list(optimizedTransformerLocationGroup2) if optimizedTransformerLocationGroup2 else None,
        'splitting_point': list(splitting_point_coords) if splitting_point_coords else None
    }
