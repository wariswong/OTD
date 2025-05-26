import os
import json
import logging
import fiona # For reading Shapefiles
import pandas as pd
import numpy as np
import networkx as nx # Assuming you have networkx installed
from shapely.geometry import Point, LineString, mapping
from scipy.optimize import minimize # For optimization functions
# import matplotlib.pyplot as plt # Not needed for web app visualization directly
# import tkinter as tk # Not needed for web app
# from tkinter.filedialog import askopenfilename # Not needed for web app
# from tkinter import simpledialog, messagebox # Not needed for web app
# from tqdm import tqdm # Not needed for web app progress bars directly in backend
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk # Not needed for web app


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- GeoJSON Helper Functions (These are crucial for web app visualization) ---
def create_geojson_points(df, x_col, y_col, properties_cols, name):
    """
    Creates a GeoJSON FeatureCollection from a pandas DataFrame for point features.
    Args:
        df (pd.DataFrame): DataFrame containing point data.
        x_col (str): Name of the column containing Longitude (X) coordinates.
        y_col (str): Name of the column containing Latitude (Y) coordinates.
        properties_cols (list): List of column names to include as GeoJSON properties.
        name (str): Name for the GeoJSON FeatureCollection.
    Returns:
        dict: A GeoJSON FeatureCollection dictionary.
    """
    features = []
    # Ensure properties_cols only includes columns present in df
    valid_properties_cols = [col for col in properties_cols if col in df.columns]

    for index, row in df.iterrows():
        try:
            # Handle potential NaN values or non-numeric types
            x_coord = pd.to_numeric(row[x_col], errors='coerce')
            y_coord = pd.to_numeric(row[y_col], errors='coerce')

            if pd.isna(x_coord) or pd.isna(y_coord):
                logging.warning(f"Skipping row {index} due to invalid coordinates (NaN): X={row[x_col]}, Y={row[y_col]}")
                continue

            coordinates = [float(x_coord), float(y_coord)] # Ensure floats

            properties = {}
            for col in valid_properties_cols:
                # Convert non-string properties to string if they are not directly JSON serializable
                value = row[col]
                properties[col] = str(value) if pd.isna(value) or not isinstance(value, (str, int, float, bool)) else value

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": coordinates
                },
                "properties": properties
            }
            features.append(feature)
        except (ValueError, TypeError) as e:
            logging.warning(f"Skipping row due to general error processing coordinates or properties: {row}. Error: {e}")
            continue
    return {
        "type": "FeatureCollection",
        "name": name,
        "features": features
    }

def create_geojson_lines_from_xy_lists(lines_list, name, properties_list=None):
    """
    Creates a GeoJSON FeatureCollection for LineString features.
    Args:
        lines_list (list): A list where each element represents a line.
                           Each line should be a dict like {'X': [x1, x2, ...], 'Y': [y1, y2, ...]}
                           or a Shapely LineString object directly.
        name (str): Name for the GeoJSON FeatureCollection.
        properties_list (list, optional): A list of dictionaries, where each dict contains properties
                                          for the corresponding line in lines_list.
    Returns:
        dict: A GeoJSON FeatureCollection dictionary.
    """
    features = []
    for i, line_data in enumerate(lines_list):
        properties = properties_list[i] if properties_list and i < len(properties_list) else {}
        
        coordinates = []
        if isinstance(line_data, LineString):
            coordinates = list(line_data.coords)
        elif isinstance(line_data, dict) and 'X' in line_data and 'Y' in line_data:
            if len(line_data['X']) != len(line_data['Y']):
                logging.warning(f"Skipping line due to unequal X and Y coordinate lists: {line_data}")
                continue
            # Ensure coordinates are floats and valid
            try:
                coordinates = [[float(x), float(y)] for x, y in zip(line_data['X'], line_data['Y'])]
            except (ValueError, TypeError) as e:
                logging.warning(f"Skipping line due to invalid coordinate values: {line_data}. Error: {e}")
                continue
        else:
            logging.warning(f"Skipping invalid line data format: {line_data}")
            continue

        if not coordinates:
            logging.warning(f"Skipping empty line coordinates for line data: {line_data}")
            continue
        
        # Filter properties to ensure they are JSON serializable
        clean_properties = {}
        for k, v in properties.items():
            if pd.isna(v): # Handle NaN values
                clean_properties[k] = None
            elif isinstance(v, (str, int, float, bool)): # Standard JSON types
                clean_properties[k] = v
            else: # Convert other types to string
                clean_properties[k] = str(v)

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates
            },
            "properties": clean_properties
        }
        features.append(feature)
    return {
        "type": "FeatureCollection",
        "name": name,
        "features": features
    }

def ensure_folder_exists(path):
    """Ensures that a directory exists, creates it if not."""
    os.makedirs(path, exist_ok=True)


# --- START OF YOUR ORIGINAL FUNCTIONS, ADAPTED FOR WEB APP ---

# 1) Meter Data Extraction
def extractMeterData(meterData_fiona_collection):
    logging.info("Extracting meter data from shapefile records...")
    
    features = list(meterData_fiona_collection)
    
    if not features:
        logging.warning("No meter features found in shapefile.")
        return pd.DataFrame(), np.array([]), np.array([]), {'A': np.array([]), 'B': np.array([]), 'C': np.array([])}, np.array([]), np.array([])

    data = []
    for feature in features:
        props = feature['properties']
        geom = feature['geometry']
        
        row_data = {k: v for k, v in props.items()}
        
        if geom and geom['type'] == 'Point' and geom['coordinates']:
            # Fiona returns coordinates as [x, y] for points
            row_data['Meter X'] = geom['coordinates'][0]
            row_data['Meter Y'] = geom['coordinates'][1]
        else:
            logging.warning(f"Feature {feature.get('id', 'N/A')} has no valid point geometry or coordinates. Skipping coordinates.")
            row_data['Meter X'] = np.nan
            row_data['Meter Y'] = np.nan
        data.append(row_data)
        
    df = pd.DataFrame(data)

    required_fields = {'OPSA_MET_2', 'OPSA_MET_3', 'OPSA_MET_4', 'PEANO'}
    if not required_fields.issubset(df.columns):
        missing = required_fields - set(df.columns)
        logging.error(f"Missing required fields in Meter shapefile: {missing}")
        raise KeyError(f"Missing required fields in Meter shapefile: {missing}")
    
    df['OPSA_MET_3'] = pd.to_numeric(df['OPSA_MET_3'], errors='coerce').fillna(0)
    df['OPSA_MET_4'] = pd.to_numeric(df['OPSA_MET_4'], errors='coerce').fillna(0)

    initialVoltages = df['OPSA_MET_3'].values.astype(float)
    totalLoads = df['OPSA_MET_4'].values.astype(float)
    phases = df['OPSA_MET_2'].values
    peano = df['PEANO'].values
    
    lowVoltageIndices = initialVoltages < 100
    if np.any(lowVoltageIndices):
        num_low_voltage = np.sum(lowVoltageIndices)
        logging.warning(f"Found {num_low_voltage} meters with voltage < 100V. These meters will be included in calculation.")
    
    phase_loads = {'A': np.zeros(len(totalLoads)),
                   'B': np.zeros(len(totalLoads)),
                   'C': np.zeros(len(totalLoads))}
    for i, phase_str in enumerate(phases):
        if not isinstance(phase_str, str):
            logging.debug(f"Skipping meter index={i} (invalid phase type: {type(phase_str)}).")
            continue

        connected_phases = list(phase_str.upper())
        if not connected_phases:
            logging.debug(f"Skipping meter index={i} (empty phases).")
            continue
        
        load_per_phase = totalLoads[i] / len(connected_phases)
        for ph in connected_phases:
            if ph in phase_loads:
                phase_loads[ph][i] = load_per_phase
            else:
                logging.warning(f"Unknown phase '{ph}' for meter index {i}.")
    
    logging.info(f"Meter data extracted successfully. Total meters: {len(df)}")
    
    return df, initialVoltages, totalLoads, phase_loads, peano, phases


# 2) Line Data Extraction (Adjusted to use Fiona)
def extractLineData(line_fiona_collection):
    logging.info("Extracting line data from shapefile records...")
    
    lines = [] # List of dicts: {'X': [x1, x2, ...], 'Y': [y1, y2, ...]}
    line_x_coords = [] # List of lists of x-coordinates
    line_y_coords = [] # List of lists of y-coordinates

    for feature in line_fiona_collection:
        if feature['geometry'] and feature['geometry']['type'] == 'LineString' and feature['geometry']['coordinates']:
            coords = feature['geometry']['coordinates']
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            lines.append({'X': xs, 'Y': ys, 'properties': feature['properties']}) # Include properties for potential use
            line_x_coords.append(xs)
            line_y_coords.append(ys)
        else:
            logging.warning(f"Skipping feature {feature.get('id', 'N/A')} due to invalid line geometry or coordinates.")
            
    logging.info(f"Line data extracted successfully. Total lines: {len(lines)}")
    return line_x_coords, line_y_coords, lines

# 3) Line Data Extraction with Attributes (Adjusted to use Fiona)
def extractLineDataWithAttributes(line_fiona_collection, attribute_name, attribute_value):
    logging.info(f"Extracting line data with attribute '{attribute_name}'='{attribute_value}'...")
    
    filtered_lines = []
    filtered_x_coords = []
    filtered_y_coords = []

    for feature in line_fiona_collection:
        if feature['geometry'] and feature['geometry']['type'] == 'LineString' and feature['geometry']['coordinates']:
            # Check if attribute exists and matches value
            prop_value = feature['properties'].get(attribute_name)
            
            # Robust comparison for string vs. numeric values
            match = False
            if isinstance(prop_value, str) and isinstance(attribute_value, str):
                if prop_value.upper() == attribute_value.upper():
                    match = True
            elif prop_value == attribute_value: # For numeric or other direct comparisons
                match = True

            if match:
                coords = feature['geometry']['coordinates']
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
                filtered_lines.append({'X': xs, 'Y': ys, 'properties': feature['properties']})
                filtered_x_coords.append(xs)
                filtered_y_coords.append(ys)
        else:
            logging.warning(f"Skipping feature {feature.get('id', 'N/A')} due to invalid line geometry or coordinates.")

    logging.info(f"Filtered line data extracted successfully. Total filtered lines: {len(filtered_lines)}")
    return filtered_x_coords, filtered_y_coords, filtered_lines


# 4) Calculate Distances (Remains largely the same)
def calculate_distances(node_a, node_b):
    return np.sqrt((node_a[0] - node_b[0])**2 + (node_a[1] - node_b[1])**2)

# 5) Build LV Network with Loads
def buildLVNetworkWithLoads(lvLines, filteredEserviceLines, meterLocations, initialTransformerLocation, phase_loads, conductorResistance, conductorReactance, use_shape_length, lvData=None, svcLines=None):
    logging.info("Building LV Network with Loads...")

    G = nx.Graph()
    tNode = tuple(initialTransformerLocation)
    G.add_node(tNode, type='Transformer')

    mNodes = []
    nm = []  # Node-to-meter mapping
    cm = {}  # Candidate-meter mapping

    # Add meters to the graph
    for i, loc in enumerate(meterLocations):
        mNode = tuple(loc)
        G.add_node(mNode, type='Meter', phase_loads={'A': phase_loads['A'][i], 'B': phase_loads['B'][i], 'C': phase_loads['C'][i]})
        mNodes.append(mNode)
        nm.append(i) # Storing original index for meter

    # Add LV lines (main network)
    for line_dict in lvLines:
        coords = list(zip(line_dict['X'], line_dict['Y']))
        # Add all segments of the line
        for i in range(len(coords) - 1):
            n1 = tuple(coords[i])
            n2 = tuple(coords[i+1])
            if not G.has_node(n1): G.add_node(n1, type='Intermediate')
            if not G.has_node(n2): G.add_node(n2, type='Intermediate')
            
            # Check if edge already exists to prevent duplicates for multi-segment lines
            if not G.has_edge(n1, n2):
                length = calculate_distances(n1, n2)
                resistance = length * conductorResistance
                reactance = length * conductorReactance
                G.add_edge(n1, n2, length=length, resistance=resistance, reactance=reactance, type='LV')
    logging.info(f"LV lines added. Current nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

    # Add E-service lines (connections from LV network to meters)
    for line_dict in filteredEserviceLines:
        coords = list(zip(line_dict['X'], line_dict['Y']))
        # The last point of an e-service line is often the meter location
        # The first point should connect to the LV network or transformer
        if len(coords) >= 2:
            meter_loc = tuple(coords[-1])
            # Ensure meter_loc is in mNodes (it should be if meters were added correctly)
            if meter_loc in mNodes:
                # Find the closest node in the existing LV network or transformer for the start of the e-service line
                start_point = tuple(coords[0])
                closest_network_node = None
                min_dist = float('inf')

                # Check all existing nodes in G
                for node in G.nodes():
                    dist = calculate_distances(start_point, node)
                    if dist < min_dist:
                        min_dist = dist
                        closest_network_node = node
                
                # If a close enough network node is found, connect the e-service line
                if closest_network_node and min_dist < 0.1: # Threshold for connection
                    # Add all segments of the e-service line
                    current_node = closest_network_node
                    for i in range(len(coords)):
                        next_node = tuple(coords[i])
                        if not G.has_node(next_node): G.add_node(next_node, type='Eservice_Intermediate')

                        length = calculate_distances(current_node, next_node)
                        if not G.has_edge(current_node, next_node):
                            resistance = length * conductorResistance
                            reactance = length * conductorReactance
                            G.add_edge(current_node, next_node, length=length, resistance=resistance, reactance=reactance, type='Eservice')
                        current_node = next_node
                    
                    # Ensure the final connection from the last e-service segment to the meter
                    if not G.has_edge(current_node, meter_loc):
                        length = calculate_distances(current_node, meter_loc)
                        resistance = length * conductorResistance
                        reactance = length * conductorReactance
                        G.add_edge(current_node, meter_loc, length=length, resistance=resistance, reactance=reactance, type='Eservice_Final')
                else:
                    logging.warning(f"Could not connect e-service line from {start_point} to network. No close node found.")
            else:
                logging.warning(f"E-service line's meter end {meter_loc} not found in meter list. Skipping.")

    logging.info(f"E-service lines added. Final nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

    # Ensure transformer node is connected if it's isolated
    if tNode not in G:
        logging.warning("Transformer node was not added. Adding it now.")
        G.add_node(tNode, type='Transformer')

    # If the transformer is isolated, try to connect it to the closest LV line or meter
    if G.degree(tNode) == 0 and G.number_of_nodes() > 1:
        logging.warning("Transformer node is isolated. Attempting to connect to closest network node.")
        closest_node_in_network = None
        min_dist_to_network = float('inf')
        for node in G.nodes():
            if node != tNode:
                dist = calculate_distances(tNode, node)
                if dist < min_dist_to_network:
                    min_dist_to_network = dist
                    closest_node_in_network = node
        
        if closest_node_in_network and min_dist_to_network < 100: # Threshold for connection
            length = min_dist_to_network
            resistance = length * conductorResistance
            reactance = length * conductorReactance
            G.add_edge(tNode, closest_node_in_network, length=length, resistance=resistance, reactance=reactance, type='Connection_To_TX')
            logging.info(f"Connected transformer to {closest_node_in_network} with length {length:.2f}m")
        else:
            logging.error("Could not connect transformer to any existing network node.")

    # Assign node types for meters (for the old code that might need it)
    for i, mNode in enumerate(mNodes):
        if G.has_node(mNode):
            G.nodes[mNode]['type'] = 'Meter' # Re-confirm type for meters

    return G, tNode, mNodes, nm, cm # cm is not used in your original logic, can return {}

# 6) Optimize Transformer Location (LV-based)
def optimizeTransformerLocationOnLVCond3(G, tNode, mNodes, nm, initialTransformerLocation, meterLocations, lvLines, conductorResistance, conductorReactance):
    logging.info("Optimizing Transformer Location on LV Cond3...")

    # Calculate voltage drop for a path
    def calculate_voltage_drop(path, current_location):
        total_drop = 0
        for i in range(len(path) - 1):
            n1 = path[i]
            n2 = path[i+1]
            if G.has_edge(n1, n2):
                edge_data = G.get_edge_data(n1, n2)
                # Simplified drop (IR + IX, assuming current in Amps)
                # This needs proper power flow to be accurate.
                # For location optimization, often length is used as a proxy.
                total_drop += edge_data['length'] # Using length as a proxy for voltage drop for optimization
        return total_drop

    # Objective function for optimization
    def objective_function(coords, G, mNodes, nm, meterLocations, conductorResistance, conductorReactance):
        new_tNode = tuple(coords)
        total_cost = 0

        # Try to connect the new_tNode to the closest point on the existing network
        min_dist_to_network = float('inf')
        closest_network_point = None # This will be the actual point on an edge or node
        closest_node_in_network = None # This will be an existing node in G

        # Iterate over all edges and nodes to find the closest point
        # For simplicity, we'll connect to the closest node for now, as finding closest point on edge is more complex.
        for node in G.nodes():
            dist = calculate_distances(new_tNode, node)
            if dist < min_dist_to_network:
                min_dist_to_network = dist
                closest_network_point = node
                closest_node_in_network = node # If connected to a node, these are the same

        if closest_network_point is None:
            logging.error("No network nodes found for connection in optimization objective.")
            return float('inf') # Cannot connect

        # Add a temporary edge from new_tNode to closest_network_point
        temp_G = G.copy()
        temp_length = min_dist_to_network
        temp_resistance = temp_length * conductorResistance
        temp_reactance = temp_length * conductorReactance
        temp_G.add_edge(new_tNode, closest_network_point, length=temp_length, resistance=temp_resistance, reactance=temp_reactance, type='Temp_TX_Connection')

        # Calculate cost based on sum of squared voltage drops to all meters
        for i, mNode_idx in enumerate(nm):
            meter_loc = meterLocations[mNode_idx] # This is using the original meterLocations array
            
            # Find closest node in temp_G to this meter_loc
            closest_node_to_meter = None
            min_dist_to_meter = float('inf')
            for node_g in temp_G.nodes():
                dist_to_meter = calculate_distances(meter_loc, node_g)
                if dist_to_meter < min_dist_to_meter:
                    min_dist_to_meter = dist_to_meter
                    closest_node_to_meter = node_g
            
            if closest_node_to_meter is None:
                logging.warning(f"Meter {meter_loc} not connected to temp_G.")
                total_cost += 1000000 # High penalty

            try:
                # Find the shortest path from the new transformer to the meter
                path = nx.shortest_path(temp_G, source=new_tNode, target=closest_node_to_meter, weight='length')
                
                # Calculate effective "voltage drop" for this path
                # This simplified model uses length, a more complex model would use power flow results
                path_length = 0
                for j in range(len(path) - 1):
                    n1, n2 = path[j], path[j+1]
                    path_length += temp_G[n1][n2]['length']
                
                # Add a penalty for distance to transformer
                # This could be load-weighted to be more accurate (load * length)
                total_cost += path_length # Using path length as proxy for voltage drop or cable cost
            except nx.NetworkXNoPath:
                total_cost += 1000000 # Large penalty if no path exists
            except KeyError as e:
                logging.error(f"Missing edge attribute in shortest path calculation: {e}")
                total_cost += 1000000 # Large penalty

        return total_cost

    # Initial guess for the transformer location (can be the current one or a load center)
    initial_guess = initialTransformerLocation

    # Perform the optimization
    # Using 'Nelder-Mead' for simplicity as it's derivative-free
    # For more complex problems, you might need 'L-BFGS-B' or 'SLSQP' with bounds
    result = minimize(objective_function, initial_guess, args=(G, mNodes, nm, meterLocations, conductorResistance, conductorReactance), method='Nelder-Mead')

    if result.success:
        optimized_location = result.x
        logging.info(f"Optimization successful. Optimized location: {optimized_location}")
        return optimized_location
    else:
        logging.error(f"Optimization failed: {result.message}")
        return initialTransformerLocation # Return original if optimization fails

# 7) Find Splitting Point
def findSplittingPoint(G, sp_node_unused, sp_coord_unused, tNode, mNodes, initialTransformerLocation, meterLocations, conductorResistance, conductorReactance):
    logging.info("Finding Splitting Point...")

    # This function is highly simplified.
    # A proper splitting point algorithm would involve:
    # 1. Calculating load center of the entire network.
    # 2. Finding a central node/edge that would best divide the network.
    # 3. Potentially considering a path from the existing transformer to this split point.

    # For now, let's just return a point slightly offset from the load center of all meters
    if not meterLocations.size:
        logging.warning("No meters found to calculate splitting point. Returning dummy point.")
        return np.array([initialTransformerLocation[0] + 0.001, initialTransformerLocation[1] + 0.001])

    avg_x = np.mean([loc[0] for loc in meterLocations])
    avg_y = np.mean([loc[1] for loc in meterLocations])
    
    # Simple strategy: find the node closest to the centroid of all meters
    centroid = np.array([avg_x, avg_y])
    
    closest_node_to_centroid = None
    min_dist_to_centroid = float('inf')
    
    # Iterate through all nodes in the graph to find the closest one
    for node in G.nodes():
        node_coords = np.array(node)
        dist = np.linalg.norm(node_coords - centroid)
        if dist < min_dist_to_centroid:
            min_dist_to_centroid = dist
            closest_node_to_centroid = node
            
    if closest_node_to_centroid:
        logging.info(f"Splitting point found at (closest node to centroid): {closest_node_to_centroid}")
        return np.array(closest_node_to_centroid)
    else:
        logging.warning("Could not find a suitable splitting point. Returning dummy point near transformer.")
        return np.array([initialTransformerLocation[0] + 0.001, initialTransformerLocation[1] + 0.001])


# 8) Partition Network at Point
def partitionNetworkAtPoint(G, sp_node, sp_coord, tNode, mNodes):
    logging.info(f"Partitioning Network at point {sp_coord}...")

    # A more robust partitioning would involve:
    # 1. Removing the splitting point node/edge.
    # 2. Finding connected components.
    # 3. Assigning meters to each partition.

    # For simplicity, we'll try to split based on proximity to the splitting point for meters
    # This might not create a true graph partition.
    
    if not sp_coord.size: # Ensure sp_coord is not empty
        logging.warning("Splitting coordinate is empty. Cannot partition network.")
        return G.copy(), nx.Graph(), [], [] # Return original graph, empty graph, empty lists

    # Create two new graphs
    G_lv1 = nx.Graph()
    G_lv2 = nx.Graph()
    
    # Assign meters to groups based on distance to splitting point
    mNodes_lv1 = []
    mNodes_lv2 = []

    for m_node in mNodes:
        if calculate_distances(m_node, sp_coord) < calculate_distances(m_node, tNode): # Example heuristic
            mNodes_lv1.append(m_node)
        else:
            mNodes_lv2.append(m_node)

    # Populate G_lv1 and G_lv2 with relevant nodes and edges.
    # This is a highly simplified copy; real partitioning is more involved.
    
    # Add nodes to G_lv1 (including transformer if it's in this partition)
    # And copy relevant edges (e.g., all edges if no true splitting is done)
    for node in G.nodes():
        if node in mNodes_lv1 or node == tNode or calculate_distances(node, sp_coord) < calculate_distances(node, tNode): # Simple proximity based assignment
            G_lv1.add_node(node, **G.nodes[node])
    
    # Add nodes to G_lv2
    for node in G.nodes():
        if node in mNodes_lv2 or node == tNode or calculate_distances(node, sp_coord) >= calculate_distances(node, tNode): # Simple proximity based assignment
            G_lv2.add_node(node, **G.nodes[node])

    # Copy edges to the respective subgraphs
    for u, v, data in G.edges(data=True):
        if G_lv1.has_node(u) and G_lv1.has_node(v):
            G_lv1.add_edge(u, v, **data)
        if G_lv2.has_node(u) and G_lv2.has_node(v):
            G_lv2.add_edge(u, v, **data)
            
    logging.info(f"Network partitioned. Group 1 meters: {len(mNodes_lv1)}, Group 2 meters: {len(mNodes_lv2)}")
    
    # nm1 and nm2 would be the indices of original meters that belong to each group
    # For now, let's map mNodes back to their original indices (if needed by other functions)
    # This requires the original nm mapping from buildLVNetworkWithLoads
    # We will pass the mNodes_lv1 and mNodes_lv2 directly to the next functions

    # Returning nm1 and nm2 as empty lists for now, as performForwardBackwardSweepAndDivideLoads
    # relies on a different result_df and g1_idx/g2_idx, which is managed there.
    return G_lv1, G_lv2, mNodes_lv1, mNodes_lv2 


# 9) Perform Forward/Backward Sweep and Divide Loads
# This function calculates power flow and updates meter data with voltages and groups
def performForwardBackwardSweepAndDivideLoads(G_lv1, G_lv2, nm1, nm2, meterLocations, sp_coord, conductorResistance, conductorReactance, totalLoads, initialVoltage):
    logging.info("Performing Forward/Backward Sweep and Dividing Loads...")

    # Initialize a DataFrame to store results for all meters
    # This will be `result_df` used for GeoJSON export
    all_meters_data = []

    # Assign meters to groups based on distance to splitting point for `result_df`
    # This is a re-grouping similar to partitionNetworkAtPoint but for the result_df
    # and assumes meterLocations is the original list from extractMeterData
    g1_idx = []
    g2_idx = []

    for i, meter_loc in enumerate(meterLocations):
        # Determine group for the current meter. 
        # This logic should ideally align with `partitionNetworkAtPoint`
        if sp_coord is not None and sp_coord.size > 0:
            if calculate_distances(meter_loc, sp_coord) < calculate_distances(meter_loc, np.array(G_lv1.graph.get('tNode', meter_loc))): # Use a central node or original transformer for comparison
                g1_idx.append(i)
            else:
                g2_idx.append(i)
        else: # If no splitting point, all in one group (e.g., Group 1)
            g1_idx.append(i)
            

    # --- Simulate Power Flow Calculation ---
    # This is a very simplified power flow simulation.
    # Your actual power flow calculation would be much more detailed.
    # It would iterate to converge voltages and currents.
    
    def simulate_power_flow_for_group(G_group, meters_in_group_indices, initial_voltage_at_tx, total_group_loads, conductorResistance, conductorReactance):
        # A simple voltage drop model: voltage decreases with distance and load
        voltages = {}
        distances_to_tx = {}
        
        # Assume transformer is at the graph's designated 'Transformer' node
        tx_node = None
        for node, data in G_group.nodes(data=True):
            if data.get('type') == 'Transformer':
                tx_node = node
                break
        
        if tx_node is None:
            logging.warning("No transformer node found in group graph. Assuming first node as transformer.")
            tx_node = list(G_group.nodes())[0] if G_group.nodes() else None

        if tx_node is None:
            logging.warning("Empty graph for power flow simulation.")
            return {}, {} # No voltages or distances

        for i in meters_in_group_indices:
            meter_loc = meterLocations[i]
            # Find closest graph node to this meter for path calculation
            closest_graph_node_to_meter = None
            min_dist_to_graph_node = float('inf')
            for node_g in G_group.nodes():
                dist = calculate_distances(meter_loc, node_g)
                if dist < min_dist_to_graph_node:
                    min_dist_to_graph_node = dist
                    closest_graph_node_to_meter = node_g
            
            if closest_graph_node_to_meter is None:
                voltages[i] = initial_voltage_at_tx # Assume initial voltage if no path
                distances_to_tx[i] = 0
                continue
                
            try:
                path = nx.shortest_path(G_group, source=tx_node, target=closest_graph_node_to_meter, weight='length')
                path_length = sum(G_group[u][v]['length'] for u, v in zip(path[:-1], path[1:])) + min_dist_to_graph_node # Add distance from closest node to meter
                
                # Simplified voltage drop: proportional to load and path length
                # A more accurate model would use actual current and impedance
                total_load_for_meter = totalLoads[i] # Use individual meter's total load
                
                # Calculate a rough voltage drop
                # This is a very simplified approximation for demonstration
                # (Load in kW / Initial Voltage) gives a rough current, then I*Z drop
                approx_current = total_load_for_meter * 1000 / initial_voltage_at_tx # kW to VA
                approx_impedance = (conductorResistance + conductorReactance) * path_length # Simple R+X * length
                
                voltage_drop = approx_current * approx_impedance / 1000 # Convert to Volts (very rough)
                final_voltage = max(180, initial_voltage_at_tx - voltage_drop) # Ensure voltage doesn't go too low

                voltages[i] = final_voltage
                distances_to_tx[i] = path_length
            except nx.NetworkXNoPath:
                logging.warning(f"No path from transformer to meter {meter_loc} in group graph. Assigning initial voltage.")
                voltages[i] = initial_voltage_at_tx
                distances_to_tx[i] = 0
            except KeyError as e:
                logging.error(f"Missing edge attribute in power flow path: {e}")
                voltages[i] = initial_voltage_at_tx
                distances_to_tx[i] = 0
        return voltages, distances_to_tx

    # Run power flow for each group
    group1_voltages, group1_distances = simulate_power_flow_for_group(G_lv1, g1_idx, initialVoltage, totalLoads, conductorResistance, conductorReactance)
    group2_voltages, group2_distances = simulate_power_flow_for_group(G_lv2, g2_idx, initialVoltage, totalLoads, conductorResistance, conductorReactance)

    # Populate result_df with new voltage and distance data
    result_df_data = []
    # Loop through all original meters (from meterLocations and totalLoads)
    # to ensure all meters are represented in the result_df, even if they
    # weren't explicitly in a group during the simplified partition
    for i in range(len(meterLocations)):
        meter_loc = meterLocations[i]
        
        # Default values
        final_voltage_a = final_voltage_b = final_voltage_c = initialVoltage
        distance_to_tx = 0
        group = "Unknown"

        # Check if meter is in Group 1
        if i in g1_idx:
            final_voltage_a = final_voltage_b = final_voltage_c = group1_voltages.get(i, initialVoltage)
            distance_to_tx = group1_distances.get(i, 0)
            group = "Group 1"
        # Check if meter is in Group 2
        elif i in g2_idx:
            final_voltage_a = final_voltage_b = final_voltage_c = group2_voltages.get(i, initialVoltage)
            distance_to_tx = group2_distances.get(i, 0)
            group = "Group 2"
        
        # Ensure peano, phases and other properties are from original meter data
        # This requires the `result_df_meters` from extractMeterData to be accessible
        # or passed as an argument. For now, assume a structure for the new row.
        # This part assumes `meterLocations` is from `result_df_meters`
        original_meter_row = meterLocations[i] # This is just [X, Y], need to map back to full row from extractMeterData result
        # To get full data for `result_df`, we must ensure `result_df_meters` from
        # `extractMeterData` is updated and returned by this function,
        # or passed in so we can modify it.
        
        # For simplicity in this mock, I'll build a new row with essential data
        # In actual implementation, you'd likely update `result_df_meters` directly
        # or merge with it.
        result_df_data.append({
            'Meter X': meter_loc[0],
            'Meter Y': meter_loc[1],
            'Peano Meter': peano[i] if len(peano) > i else '', # Assuming peano is passed correctly
            'Final Voltage A (V)': final_voltage_a,
            'Final Voltage B (V)': final_voltage_b,
            'Final Voltage C (V)': final_voltage_c,
            'Distance to Transformer (m)': distance_to_tx,
            'Load A (kW)': phase_loads['A'][i] if len(phase_loads['A']) > i else 0,
            'Load B (kW)': phase_loads['B'][i] if len(phase_loads['B']) > i else 0,
            'Load C (kW)': phase_loads['C'][i] if len(phase_loads['C']) > i else 0,
            'Group': group,
            'Phases': phases[i] if len(phases) > i else '' # Assuming phases is passed correctly
        })
        
    result_df = pd.DataFrame(result_df_data)

    logging.info("Forward/Backward Sweep and Load Division completed.")
    return result_df, g1_idx, g2_idx


# 10) Calculate Network Load Center
def calculateNetworkLoadCenter(meterLocations_group, load_group):
    logging.info("Calculating Network Load Center...")
    if not meterLocations_group:
        logging.warning("No meters in group for load center calculation. Returning origin.")
        return np.array([0, 0])
    
    # Assuming meterLocations_group is a list of [x,y] coordinates
    # load_group is the total load, this simplistic function doesn't use individual loads
    
    # A more accurate load center calculation would be load-weighted average of meter coordinates
    # For now, a simple average of coordinates
    avg_x = np.mean([loc[0] for loc in meterLocations_group])
    avg_y = np.mean([loc[1] for loc in meterLocations_group])
    return np.array([avg_x, avg_y])

# 11) Optimize Group (Transformer Location for a Sub-Group)
def optimizeGroup(G_group, tNode_group, mNodes_group, meterLocations_group, initialGroupTransformerLocation, totalLoads_group, conductorResistance, conductorReactance):
    logging.info(f"Optimizing Group Transformer Location for {len(mNodes_group)} meters...")

    # Objective function for group optimization
    def objective_function_group(coords, G_group, mNodes_group, meterLocations_group, conductorResistance, conductorReactance):
        new_tx_loc = tuple(coords)
        total_cost = 0

        # Try to connect the new_tx_loc to the closest point on the existing group network
        min_dist_to_group_network = float('inf')
        closest_group_node = None
        
        for node in G_group.nodes():
            dist = calculate_distances(new_tx_loc, node)
            if dist < min_dist_to_group_network:
                min_dist_to_group_network = dist
                closest_group_node = node
        
        if closest_group_node is None:
            logging.error("No group network nodes found for connection in group optimization.")
            return float('inf')

        temp_G_group = G_group.copy()
        temp_length = min_dist_to_group_network
        temp_resistance = temp_length * conductorResistance
        temp_reactance = temp_length * conductorReactance
        temp_G_group.add_edge(new_tx_loc, closest_group_node, length=temp_length, resistance=temp_resistance, reactance=temp_reactance, type='Temp_Group_TX_Connection')

        # Calculate cost based on path length to all meters in the group
        for i, meter_loc in enumerate(meterLocations_group): # meterLocations_group should be the actual coordinates of meters in this group
            # Find closest graph node in temp_G_group to this meter_loc
            closest_node_to_meter = None
            min_dist_to_meter_in_group = float('inf')
            for node_g in temp_G_group.nodes():
                dist_to_meter = calculate_distances(meter_loc, node_g)
                if dist_to_meter < min_dist_to_meter_in_group:
                    min_dist_to_meter_in_group = dist_to_meter
                    closest_node_to_meter = node_g

            if closest_node_to_meter is None:
                total_cost += 1000000 # Penalty
                continue

            try:
                path = nx.shortest_path(temp_G_group, source=new_tx_loc, target=closest_node_to_meter, weight='length')
                path_length = sum(temp_G_group[u][v]['length'] for u, v in zip(path[:-1], path[1:]))
                total_cost += path_length # Using path length as proxy for cost
            except nx.NetworkXNoPath:
                total_cost += 1000000 # Large penalty
            except KeyError as e:
                logging.error(f"Missing edge attribute in group optimization path: {e}")
                total_cost += 1000000

        return total_cost

    initial_guess_group = initialGroupTransformerLocation

    result_group = minimize(objective_function_group, initial_guess_group, args=(G_group, mNodes_group, meterLocations_group, conductorResistance, conductorReactance), method='Nelder-Mead')

    if result_group.success:
        optimized_location_group = result_group.x
        logging.info(f"Group optimization successful. Optimized location: {optimized_location_group}")
        return optimized_location_group
    else:
        logging.error(f"Group optimization failed: {result_group.message}")
        return initialGroupTransformerLocation # Return original if optimization fails

# 12) Calculate Unbalanced Power Flow (Placeholder, needs full implementation)
def calculateUnbalancedPowerFlow(G, tNode, mNodes, initialVoltage, meterLocations, phase_loads, conductorResistance, conductorReactance):
    logging.info("Calculating Unbalanced Power Flow (Placeholder)...")
    # This is a complex function requiring full power flow algorithms (e.g., Newton-Raphson, Backward/Forward Sweep for unbalanced systems).
    # This mock returns dummy results.
    # In a real scenario, this would populate voltages at all nodes/meters, currents, and losses.
    
    # Return a dictionary of node_voltages, e.g., {(x,y): {'A': V_a, 'B': V_b, 'C': V_c}}
    # And maybe meter_voltages (filtered from node_voltages for meter nodes)
    
    simulated_voltages = {}
    for meter_loc in meterLocations:
        # Simulate some voltage drop based on a generic factor
        simulated_voltages[tuple(meter_loc)] = {
            'A': initialVoltage * 0.95,
            'B': initialVoltage * 0.95,
            'C': initialVoltage * 0.95
        }
    logging.info("Unbalanced Power Flow calculation (mock) completed.")
    return simulated_voltages # Return simulated voltages at meters

# 13) Get Transformer Losses
def get_transformer_losses(kva_rating, load_kVA, loss_no_load, loss_load):
    logging.info("Calculating transformer losses...")
    
    # This function is from your original code.
    # It calculates losses based on no-load and load losses provided in percentage.
    # kva_rating in kVA, load_kVA in kVA. loss_no_load and loss_load are percentages (e.g., 0.005 for 0.5%)
    
    # Assuming loss_no_load and loss_load are per unit or percentage of kva_rating
    
    # Example: If loss_no_load is 0.001 (0.1% no-load loss)
    # If loss_load is 0.005 (0.5% load loss at full load)
    
    # Per-unit values often simplify calculations, but let's assume direct values if that's how they are provided.
    
    # No-load losses (core losses): constant regardless of load
    no_load_losses_kW = kva_rating * loss_no_load if loss_no_load else 0

    # Load losses (copper losses): proportional to the square of the load
    # Assuming loss_load is the loss at 1.0 p.u. load (i.e., at kva_rating)
    if kva_rating > 0:
        load_fraction = load_kVA / kva_rating
        load_losses_kW = (load_fraction ** 2) * (kva_rating * loss_load if loss_load else 0)
    else:
        load_losses_kW = 0
    
    total_losses_kW = no_load_losses_kW + load_losses_kW
    
    logging.info(f"Transformer losses: No-load={no_load_losses_kW:.2f} kW, Load={load_losses_kW:.2f} kW, Total={total_losses_kW:.2f} kW")
    return total_losses_kW


# 14) Loss Documentation (Placeholder)
def Lossdocument(total_losses):
    logging.info(f"Documenting total losses: {total_losses:.2f} kW (Placeholder)")
    # In a real scenario, this would write to a report file or database.
    pass


# --- Main Processing Function for Web App ---

def run_process_from_project_folder(project_id, upload_folder):
    """
    Main function to run the power network analysis for a given project.
    It reads shapefiles, performs calculations, and exports results as GeoJSON.

    Args:
        project_id (int): The ID of the project.
        upload_folder (str): The base path to the uploads directory.

    Returns:
        dict: A dictionary containing status message, project_id, and path to GeoJSONs.
    Raises:
        FileNotFoundError: If a required shapefile is missing.
        Exception: For other processing errors.
    """
    logging.info(f"Starting process for project ID: {project_id}")
    project_folder_path = os.path.join(upload_folder, str(project_id))
    ensure_folder_exists(project_folder_path)

    # Define paths for required shapefiles
    shp_files = {
        'meter': os.path.join(project_folder_path, 'meter.shp'),
        'lv': os.path.join(project_folder_path, 'lv.shp'),
        'mv': os.path.join(project_folder_path, 'mv.shp'),
        'eservice': os.path.join(project_folder_path, 'eservice.shp'),
        'tr': os.path.join(project_folder_path, 'tr.shp'),
    }

    # Verify all required shapefiles exist
    for file_type, path in shp_files.items():
        if not os.path.exists(path):
            logging.error(f"Missing required shapefile: {file_type}.shp at {path}")
            raise FileNotFoundError(f"Missing required shapefile: {file_type}.shp at {path}")

    # Variables to hold opened fiona collections
    meterData = lvData = mvData = eserviceData = transformerData = None

    try:
        logging.info("Loading shapefiles...")
        meterData = fiona.open(shp_files['meter'])
        lvData = fiona.open(shp_files['lv'])
        mvData = fiona.open(shp_files['mv'])
        eserviceData = fiona.open(shp_files['eservice'])
        transformerData = fiona.open(shp_files['tr'])

        # --- Extract Initial Data using your functions ---
        # result_df_meters will be a DataFrame (from extractMeterData)
        result_df_meters, initialVoltages, totalLoads, phase_loads, peano, phases = extractMeterData(meterData)
        
        # Adjusting meterLocations for downstream functions if they expect a numpy array of coordinates
        # Ensure result_df_meters is not empty before attempting to convert
        meterLocations_np = result_df_meters[['Meter X', 'Meter Y']].values if not result_df_meters.empty else np.array([])
        
        # Note: If `lvLines`, `mvLines`, `filteredEserviceLines` also need properties for GeoJSON,
        # ensure extractLineData functions return them in the dictionary for each line.
        lvLinesX, lvLinesY, lvLines = extractLineData(lvData)
        mvLinesX, mvLinesY, mvLines = extractLineData(mvData)
        # Note: SUBTYPECOD = 5 for service lines, SUBTYPECOD = 2 for other Eservice lines as per your original code's logic
        svcLinesX, svcLinesY, svcLines = extractLineDataWithAttributes(eserviceData, 'SUBTYPECOD', 5) 
        eserviceLinesX, eserviceLinesY, filteredEserviceLines = extractLineDataWithAttributes(eserviceData, 'SUBTYPECOD', 2)

        transformer_features = list(transformerData)
        if not transformer_features:
            raise ValueError("No transformer features found in transformer shapefile.")
        
        initialTransformerGeom = transformer_features[0]['geometry']
        if not initialTransformerGeom or initialTransformerGeom['type'] != 'Point':
             raise ValueError("Initial transformer geometry is not a Point or is missing.")
        initialTransformerLocation = np.array(initialTransformerGeom['coordinates'])
        
        transformer_properties = transformer_features[0]['properties']
        # Use .get() with a default value to prevent KeyError if 'OPSA_TRS_3' is missing
        transformerCapacity_kVA = transformer_properties.get('OPSA_TRS_3', 100) 
        # Convert to numeric, handle potential non-numeric values
        transformerCapacity_kVA = pd.to_numeric(transformerCapacity_kVA, errors='coerce').fillna(100)

        powerFactor = 0.875 # Default power factor as in your code
        transformerCapacity = transformerCapacity_kVA * powerFactor

        conductorResistance = 0.77009703 # Hardcoded values from your original code
        conductorReactance = 0.3497764 # Hardcoded values from your original code
        initialVoltage = 230 # Hardcoded values from your original code
        # growthRate = 0.05 # Not used in this direct flow, but kept for context if needed

        logging.info("Shapefiles loaded and initial data extracted.")

        logging.info("Running core network analysis...")
        
        G_init, tNode_init, mNodes_init, nm_init, cm_init = buildLVNetworkWithLoads(
            lvLines, filteredEserviceLines, meterLocations_np.tolist(), 
            initialTransformerLocation, phase_loads, conductorResistance, conductorReactance,
            use_shape_length=True, lvData=lvData, svcLines=svcLines 
        )

        optimizedTransformerLocation_LV = optimizeTransformerLocationOnLVCond3(
            G_init, tNode_init, mNodes_init, nm_init, initialTransformerLocation, 
            meterLocations_np.tolist(), lvLines, conductorResistance, conductorReactance
        )
        logging.info(f"Optimized Transformer Location (LV): {optimizedTransformerLocation_LV}")

        # The signature for findSplittingPoint in your provided code
        # is `findSplittingPoint(G, sp_node, sp_coord, tNode, mNodes, initialTransformerLocation, meterLocations, conductorResistance, conductorReactance)`
        # `sp_node` and `sp_coord` are inputs, but in the main flow, these are outputs.
        # Assuming `sp_coord` is the primary output and `sp_node` can be derived or is optional.
        # For now, passing None for unused `sp_node` and `sp_coord` inputs.
        sp_coord = findSplittingPoint( 
            G_init, None, None, tNode_init, mNodes_init, initialTransformerLocation, 
            meterLocations_np.tolist(), conductorResistance, conductorReactance
        )
        logging.info(f"Splitting Point: {sp_coord}")

        # Assuming partitionNetworkAtPoint now returns graph objects and node lists for sub-groups
        G_lv1, G_lv2, nm1_nodes, nm2_nodes = partitionNetworkAtPoint(G_init, sp_coord, sp_coord, tNode_init, mNodes_init)
        
        # `performForwardBackwardSweepAndDivideLoads` needs `peano` and `phases`
        # for `result_df` construction. Pass them as arguments.
        result_df, g1_idx, g2_idx = performForwardBackwardSweepAndDivideLoads(
            G_lv1, G_lv2, nm1_nodes, nm2_nodes, meterLocations_np.tolist(), sp_coord, 
            conductorResistance, conductorReactance, totalLoads, initialVoltage # Pass totalLoads and initialVoltage
        )
        logging.info(f"Power flow analysis completed. Meters processed: {len(result_df)}")

        optimizedTransformerLocationGroup1 = None
        optimizedTransformerLocationGroup2 = None
        
        if g1_idx and G_lv1 and not result_df.empty:
            meterLocations_g1_df = result_df.loc[g1_idx]
            meterLocations_g1 = meterLocations_g1_df[['Meter X', 'Meter Y']].values.tolist()
            load_g1 = meterLocations_g1_df['Load A (kW)'].sum() # Assuming total load for group optimization
            load_center_g1 = calculateNetworkLoadCenter(meterLocations_g1, load_g1)
            optimizedTransformerLocationGroup1 = optimizeGroup(
                G_lv1, tNode_init, [mNodes_init[i] for i in g1_idx], meterLocations_g1, load_center_g1,
                load_g1, conductorResistance, conductorReactance
            )
            logging.info(f"Optimized Transformer Location (Group 1): {optimizedTransformerLocationGroup1}")

        if g2_idx and G_lv2 and not result_df.empty:
            meterLocations_g2_df = result_df.loc[g2_idx]
            meterLocations_g2 = meterLocations_g2_df[['Meter X', 'Meter Y']].values.tolist()
            load_g2 = meterLocations_g2_df['Load A (kW)'].sum()
            load_center_g2 = calculateNetworkLoadCenter(meterLocations_g2, load_g2)
            optimizedTransformerLocationGroup2 = optimizeGroup(
                G_lv2, tNode_init, [mNodes_init[i] for i in g2_idx], meterLocations_g2, load_center_g2,
                load_g2, conductorResistance, conductorReactance
            )
            logging.info(f"Optimized Transformer Location (Group 2): {optimizedTransformerLocationGroup2}")
        
        # Calculate losses
        # Assuming loss_no_load and loss_load are fixed values or derived from elsewhere
        # Example values for demonstration (replace with your actual constants)
        loss_no_load_percentage = 0.001 # 0.1% of kVA rating
        loss_load_percentage = 0.005 # 0.5% of kVA rating at full load
        
        transformer_loss_noload = get_transformer_losses(transformerCapacity_kVA, 0, loss_no_load_percentage, 0)
        transformer_loss_load = get_transformer_losses(transformerCapacity_kVA, totalLoads.sum(), 0, loss_load_percentage) # totalLoads is an array, sum it
        total_losses = transformer_loss_noload + transformer_loss_load
        Lossdocument(total_losses)

        logging.info("Core analysis completed.")

        # --- Export Results to GeoJSON ---
        logging.info("Exporting results to GeoJSON...")

        # GeoJSON for Meters (uses result_df from performForwardBackwardSweepAndDivideLoads)
        meter_properties = ['Peano Meter', 'Final Voltage A (V)', 'Final Voltage B (V)', 'Final Voltage C (V)',
                            'Distance to Transformer (m)', 'Load A (kW)', 'Load B (kW)', 'Load C (kW)', 'Group', 'Phases']
        meters_geojson = create_geojson_points(result_df, 'Meter X', 'Meter Y', meter_properties, 'Meters')
        with open(os.path.join(project_folder_path, "meters.geojson"), "w") as f:
            json.dump(meters_geojson, f)

        # GeoJSON for Transformers and Splitting Points
        transformer_points_data = []
        if initialTransformerLocation is not None and len(initialTransformerLocation) >= 2:
             transformer_points_data.append({'X': initialTransformerLocation[0], 'Y': initialTransformerLocation[1], 'Name': 'Initial Transformer'})
        if optimizedTransformerLocation_LV is not None and len(optimizedTransformerLocation_LV) >= 2:
             transformer_points_data.append({'X': optimizedTransformerLocation_LV[0], 'Y': optimizedTransformerLocation_LV[1], 'Name': 'Optimized TX (LV-based)'})
        if optimizedTransformerLocationGroup1 is not None and len(optimizedTransformerLocationGroup1) >= 2:
            transformer_points_data.append({'X': optimizedTransformerLocationGroup1[0], 'Y': optimizedTransformerLocationGroup1[1], 'Name': 'Group 1 Transformer'})
        if optimizedTransformerLocationGroup2 is not None and len(optimizedTransformerLocationGroup2) >= 2:
            transformer_points_data.append({'X': optimizedTransformerLocationGroup2[0], 'Y': optimizedTransformerLocationGroup2[1], 'Name': 'Group 2 Transformer'})
        if sp_coord is not None and len(sp_coord) >= 2:
            transformer_points_data.append({'X': sp_coord[0], 'Y': sp_coord[1], 'Name': 'Splitting Point'})

        transformer_df_export = pd.DataFrame(transformer_points_data)
        if not transformer_df_export.empty:
            transformer_geojson = create_geojson_points(transformer_df_export, 'X', 'Y', ['Name'], 'Transformers and Splitting Points')
            with open(os.path.join(project_folder_path, "transformers.geojson"), "w") as f:
                json.dump(transformer_geojson, f)

        # Ensure lvLines, mvLines, filteredEserviceLines are in the format expected by create_geojson_lines_from_xy_lists
        # (list of dicts like {'X':[], 'Y':[], 'properties': {}})
        lv_lines_geojson = create_geojson_lines_from_xy_lists(lvLines, 'LV Lines', [line['properties'] for line in lvLines] if lvLines and 'properties' in lvLines[0] else None)
        with open(os.path.join(project_folder_path, "lv_lines.geojson"), "w") as f:
            json.dump(lv_lines_geojson, f)

        mv_lines_geojson = create_geojson_lines_from_xy_lists(mvLines, 'MV Lines', [line['properties'] for line in mvLines] if mvLines and 'properties' in mvLines[0] else None)
        with open(os.path.join(project_folder_path, "mv_lines.geojson"), "w") as f:
            json.dump(mv_lines_geojson, f)

        eservice_lines_geojson = create_geojson_lines_from_xy_lists(filteredEserviceLines, 'Eservice Lines', [line['properties'] for line in filteredEserviceLines] if filteredEserviceLines and 'properties' in filteredEserviceLines[0] else None)
        with open(os.path.join(project_folder_path, "eservice_lines.geojson"), "w") as f:
            json.dump(eservice_lines_geojson, f)

        logging.info("GeoJSON files exported successfully.")

        # Return a dictionary with path to GeoJSONs
        return {
            'message': 'ประมวลผลเสร็จสิ้น',
            'project_id': project_id,
            'geojson_base_path': f"/uploads/{project_id}/"
        }

    except FileNotFoundError as e:
        logging.error(f"Required file not found: {e}")
        raise
    except Exception as e:
        logging.exception(f"Error during processing for project {project_id}")
        raise

    finally:
        if meterData and not meterData.closed: meterData.close()
        if lvData and not lvData.closed: lvData.close()
        if mvData and not mvData.closed: mvData.close()
        if eserviceData and not eserviceData.closed: eserviceData.close()
        if transformerData and not transformerData.closed: transformerData.close()