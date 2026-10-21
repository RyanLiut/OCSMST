from flask import Flask, request, jsonify, send_from_directory
from SMM import SemanticMap
from utils import port

import numpy as np
import pandas as pd
import uuid
import traceback
import time
import re
import io

app = Flask(__name__, static_folder='./')


@app.route('/')
def index():
    return send_from_directory('./', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('./', path)


@app.route('/api/process-excel', methods=['POST'])
def process_excel():
    """
    Generate a semantic map based on the xlsx data parsed by the frontend.
    """
    try:
        # Fetch request data
        data = request.json.get('data', [])
        label= request.json.get('label', [])
        sheet3 = request.json.get('sheet3', [])
        calc_type = request.json.get('calc_type', 'J')
        blend_weight = request.json.get('blend_weight', 0.4)

        # Check data.
        if not data or not isinstance(data, list):
            return jsonify({'error': 'Invalid data format.'}), 400
        if not isinstance(label, list):
            label = []

        # Convert the data into a DataFrame.
        df = pd.DataFrame(data)
        df_label = pd.DataFrame(label)

        # Normalize column names and structure
        df, norm_result = normalize_columns(df)
        if norm_result != 'standard':
            return jsonify({'error': norm_result}), 400

        # Preprocess: detect and merge multi-row headers, forward-fill merged cells.
        df = preprocess_multirow_header(df)

        # Extract group info from non-numeric columns (e.g., '语组') before filtering them out
        form_groups = extract_form_groups(df)

        # Remove non-numeric feature columns (e.g., language group columns)
        df = filter_non_numeric_features(df)

        # Check whether the data is standardized.
        v_result = validate_data(df)
        # Validation failed; return error code.
        if v_result != 'standard':
            return jsonify({'error': v_result}), 400

        # Process and standardize the data.
        features, forms, co_occurrence_matrix, overlay_info = process_data(df)
        ground_truth = process_label(df_label, features)

        # Determine llm_sim source for calc_type='L'
        llm_sim_path = None
        llm_sim_data = None
        if calc_type == 'L':
            llm_sim_csv = request.json.get('llm_sim_csv', None)
            if llm_sim_csv:
                llm_sim_data = pd.read_csv(io.StringIO(llm_sim_csv), index_col=0)
            else:
                return jsonify({'error': 'calc_type=L requires an embedding similarity CSV (llm_sim_csv)'}), 400

        # Generate the adjacency matrix corresponding to the candidate tree.
        semantic_maps = SemanticMap(co_occurrence_matrix, features, forms, None, ground_truth, calc_type=calc_type, blend=(blend_weight, False), llm_sim_path=llm_sim_path, llm_sim_data=llm_sim_data, cooc_scale="minmax") # (blend_weight, True), 
        print(co_occurrence_matrix, features, forms, None, ground_truth, 0)
        matrix_data = semantic_maps.get_all_matrix()

        # Convert from matrix to graph format as required by the frontend.
        graph_data = list()
        node_labels = semantic_maps.unique_featNames

        # origin_matrix_data = matrix_data.get('origin_matrix')
        # origin_matrix_data = process_symmetric_matrix(origin_matrix_data)
        # origin_graph_data = convert_to_graph_data('Initial fully connected conceptual space ', origin_matrix_data, node_labels)
        # # 评价当前图
        # origin_evaluation_metric = semantic_maps.get_evaluation_metric(origin_matrix_data, range(len(semantic_maps.formNames)))
        # origin_graph_data['evaluation_metric'] = origin_evaluation_metric
        # graph_data.append(origin_graph_data)

        trees_matrix_data = matrix_data.get('trees')
        start_evaluation = time.time()

        # Compute GT rank (position of best-GT-matching tree among all spanning trees by weight)
        # gt_rank_result = None
        # if semantic_maps.GT_adj is not None:
        #     gt_rank_result = semantic_maps.get_gt_rank()
        #     print(f"[GT Rank] {gt_rank_result}")
        gt_rank_result = None

        for index, tree_matrix in enumerate(trees_matrix_data):
            # Process the symmetric adjacency matrix.
            tree_matrix = process_symmetric_matrix(tree_matrix)
            tree_graph_data = convert_to_graph_data('Candidate semantic map '+str(index+1), tree_matrix, node_labels)
            # Evaluate the current semantic map.
            tree_evaluation_metric = semantic_maps.get_evaluation_metric(tree_matrix, range(len(semantic_maps.formNames)))
            if gt_rank_result is not None:
                tree_evaluation_metric['rank'] = gt_rank_result[0]
                tree_evaluation_metric['rank_f1'] = gt_rank_result[1]
            tree_graph_data['evaluation_metric'] = tree_evaluation_metric
            graph_data.append(tree_graph_data)
        end_evaluation = time.time()
        print('Evaluation time elapsed: ', (end_evaluation-start_evaluation))

        # Build dedup-space GT matrix and append GT slide + gt_matrix for frontend comparison
        gt_dedup_matrix = None
        gt_graph_data = None
        if semantic_maps.GT_adj is not None:
            n_unique = len(semantic_maps.unique_featNames)
            gt_norm = semantic_maps.norm_matrix(semantic_maps.GT_adj)
            gt_dedup_matrix = np.zeros((n_unique, n_unique))
            # merge_feat_info contains (kept_idx, merged_idx) pairs where merged_idx is removed.
            # Build a correct origin->dedup index mapping by iterating original indices in order
            # and assigning dedup indices only to non-merged columns.
            merged_set = set()
            target_map = {}  # original kept_idx -> original merged_idx list
            for kept_idx, merged_idx in semantic_maps.merge_feat_info:
                merged_set.add(merged_idx)
                target_map.setdefault(kept_idx, []).append(merged_idx)
            origin_to_dedup = {}
            dedup_idx = 0
            for oi in range(gt_norm.shape[0]):
                if oi in merged_set:
                    continue
                origin_to_dedup[oi] = dedup_idx
                # Merged columns map to the same dedup index
                for mi in target_map.get(oi, []):
                    origin_to_dedup[mi] = dedup_idx
                dedup_idx += 1
            # Use the full adjacency matrix (dedup-space) to get computed weights for GT edges
            origin_matrix = matrix_data.get('origin_matrix')
            for oi in range(gt_norm.shape[0]):
                di = origin_to_dedup.get(oi, oi)
                for oj in range(gt_norm.shape[1]):
                    dj = origin_to_dedup.get(oj, oj)
                    if gt_norm[oi, oj] > 0:
                        # Use the computed weight from the full adjacency matrix
                        w = origin_matrix[di, dj] if origin_matrix is not None else 1.0
                        gt_dedup_matrix[di, dj] = max(gt_dedup_matrix[di, dj], w)
            gt_dedup_matrix = process_symmetric_matrix(gt_dedup_matrix)
            gt_graph_data = convert_to_graph_data('Ground Truth', gt_dedup_matrix, node_labels)
            gt_graph_data['is_gt'] = True
            gt_evaluation_metric = semantic_maps.get_evaluation_metric(gt_dedup_matrix, range(len(semantic_maps.formNames)))
            gt_graph_data['evaluation_metric'] = gt_evaluation_metric
            graph_data.append(gt_graph_data)

        # Retrieve the forms and the corresponding nodes for each form in the semantic map.
        feature_map = semantic_maps.tfM
        forms_with_nodes = get_forms_with_nodes(df, feature_map, overlay_info, semantic_maps)

        # Parse per-example feature label mapping from Sheet3
        per_example_labels = parse_sheet3_labels(sheet3, semantic_maps.unique_featNames)

        ret = {
            'graph_data': graph_data,
            'forms_with_nodes': forms_with_nodes,
            'form_groups': form_groups,
            'gt_matrix': gt_dedup_matrix.tolist() if gt_dedup_matrix is not None else None,
            'per_example_labels': per_example_labels
        }
        return jsonify(ret)

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/merge-edges', methods=['POST'])
def merge_edges():
    """
    Automatically add edges to make the semantic map fully connected.
    """
    try:
        # Fetch request data
        data = request.json.get('data', [])
        label = request.json.get('label', [])
        graph = request.json.get('graph', {})
        calc_type = request.json.get('calc_type', 'J')
        blend_weight = request.json.get('blend_weight', 0.4)

        if not data or not graph:
            return jsonify({'error': 'The backend failed to parse the data. Please check the file format and content.'}), 400

        # Convert the data into a DataFrame.
        df = pd.DataFrame(data)
        df_label = pd.DataFrame(label)

        # Normalize column names and structure (same as process_excel)
        df, norm_result = normalize_columns(df)
        if norm_result != 'standard':
            return jsonify({'error': norm_result}), 400
        df = preprocess_multirow_header(df)
        df = filter_non_numeric_features(df)

        # Check whether the data is standardized.
        v_result = validate_data(df)
        if v_result != 'standard':
            return jsonify({'error': v_result}), 400

        # Process and standardize the data.
        features, forms, co_occurrence_matrix, overlay_info = process_data(df)
        ground_truth = process_label(df_label, features)

        # Parse the current map information.
        adjacency_matrix, map_name = process_graph(graph)
        adjacency_matrix = process_symmetric_matrix(adjacency_matrix)

        # Merge edges
        llm_sim_path = None
        llm_sim_data = None
        if calc_type == 'L':
            llm_sim_csv = request.json.get('llm_sim_csv', None)
            if llm_sim_csv:
                llm_sim_data = pd.read_csv(io.StringIO(llm_sim_csv), index_col=0)
            else:
                return jsonify({'error': 'calc_type=L requires an embedding similarity CSV (llm_sim_csv)'}), 400
        semantic_maps = SemanticMap(co_occurrence_matrix, features, forms, None, ground_truth, calc_type=calc_type, blend=(blend_weight, True), llm_sim_path=llm_sim_path, llm_sim_data=llm_sim_data,cooc_scale="minmax")
        adjacency_matrix_merged, merged_edges_info = semantic_maps.merge_edge(adjacency_matrix)

        # Convert from edges to graph format as required by the frontend.
        graph_data = dict()
        edges = convert_to_graph_data_merged(merged_edges_info)
        graph_data['edges'] = edges

        # Evaluate the merged semantic map.
        evaluation_metric = semantic_maps.get_evaluation_metric(adjacency_matrix_merged, range(len(forms)))
        graph_data['evaluation_metric'] = evaluation_metric

        # Retrieve the forms and the corresponding nodes for each form in the semantic map.
        feature_map = semantic_maps.tfM # tfm and co_occurrence_matrix are not the same—deduplication may have been applied.
        forms_with_nodes = get_forms_with_nodes(df, feature_map, overlay_info, semantic_maps)

        ret = {
            'graph_data': graph_data,
            'forms_with_nodes': forms_with_nodes
        }
        return jsonify(ret)

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500



@app.route('/api/edge-modify', methods=['POST'])
def edge_modify():
    """
    Re-evaluate a semantic map after the user manually edits edges
    (add / edit / delete) in the frontend.
    """
    try:
        data = request.json.get('data', [])
        label = request.json.get('label', [])
        graph = request.json.get('graph', {})

        if not data or not graph:
            return jsonify({'error': 'Invalid request data.'}), 400

        df = pd.DataFrame(data)
        df_label = pd.DataFrame(label)

        df, norm_result = normalize_columns(df)
        if norm_result != 'standard':
            return jsonify({'error': norm_result}), 400
        df = preprocess_multirow_header(df)
        df = filter_non_numeric_features(df)

        v_result = validate_data(df)
        if v_result != 'standard':
            return jsonify({'error': v_result}), 400

        features, forms, co_occurrence_matrix, overlay_info = process_data(df)
        ground_truth = process_label(df_label, features)

        adjacency_matrix, map_name = process_graph(graph)
        adjacency_matrix = process_symmetric_matrix(adjacency_matrix)

        semantic_maps = SemanticMap(co_occurrence_matrix, features, forms, None, ground_truth, calc_type='J', blend=(0.1, True), cooc_scale="minmax")
        evaluation_metric = semantic_maps.get_evaluation_metric(adjacency_matrix, range(len(forms)))

        return jsonify(evaluation_metric)

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500



def normalize_columns(df):
    """
    Normalize column names and structure to the expected format:
    - Rename case-insensitive 'languages'/'forms' variants
    - If there is no 'forms' column, insert one from 'languages'
    - If the first column is neither 'languages' nor 'forms', treat it as a
      non-standard column and rename/insert as needed
    Does NOT remove non-numeric feature columns yet (that happens after
    preprocess_multirow_header in filter_non_numeric_features).
    """
    cols = [str(c).strip() for c in df.columns]

    # Find and rename 'languages' (case-insensitive)
    lang_idx = None
    for i, c in enumerate(cols):
        if c.lower() == 'languages':
            cols[i] = 'languages'
            lang_idx = i
            break

    # Find and rename 'forms' (case-insensitive)
    forms_idx = None
    for i, c in enumerate(cols):
        if c.lower() == 'forms':
            cols[i] = 'forms'
            forms_idx = i
            break

    df.columns = cols

    if lang_idx is None:
        first_col = df.iloc[:, 0]
        if first_col.dtype == object:
            cols[0] = 'languages'
            df.columns = cols
            lang_idx = 0
        else:
            return df, "Could not find a 'languages' column."

    if forms_idx is None:
        next_idx = lang_idx + 1
        if next_idx < len(df.columns):
            next_col = df.iloc[:, next_idx]
            if pd.api.types.is_numeric_dtype(next_col):
                df.insert(next_idx, 'forms', df['languages'])
            else:
                cols[next_idx] = 'forms'
                df.columns = cols
        else:
            df.insert(lang_idx + 1, 'forms', df['languages'])

    # Ensure 'languages' is column 0 and 'forms' is column 1
    if list(df.columns[:2]) != ['languages', 'forms']:
        other_cols = [c for c in df.columns if c not in ('languages', 'forms')]
        df = df[['languages', 'forms'] + other_cols].copy()

    return df, 'standard'


def filter_non_numeric_features(df):
    """
    Remove any non-numeric columns after 'forms' that are not feature columns
    (e.g., a language family/group column like '语组').
    Should be called AFTER preprocess_multirow_header.
    Keeps the 'type' column if present (for overlay forms).
    """
    keep = ['languages', 'forms']
    if 'type' in df.columns:
        keep.append('type')
    for col in df.columns[2:]:
        if col == 'type':
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            keep.append(col)
    df = df[[c for c in df.columns if c in keep]].copy()
    return df


def extract_form_groups(df):
    """
    Extract grouping information from non-numeric columns after 'forms'.
    Returns a dict: { group_col_name: [ { 'name': group_value, 'form_indices': [0,1,...] }, ... ] }
    If no group columns exist, returns an empty dict.
    """
    groups = {}
    non_numeric_cols = [c for c in df.columns[2:] if not pd.api.types.is_numeric_dtype(df[c]) and c != 'type']
    for col in non_numeric_cols:
        group_values = df[col].ffill().tolist()
        unique_groups = {}
        for idx, val in enumerate(group_values):
            if pd.notna(val):
                val_str = str(val)
                if val_str not in unique_groups:
                    unique_groups[val_str] = []
                unique_groups[val_str].append(idx)
        if unique_groups:
            group_list = [{'name': name, 'form_indices': indices} for name, indices in unique_groups.items()]
            groups[col] = group_list
    return groups


def preprocess_multirow_header(df):
    """
    Detect and merge multi-row headers.
    Some Excel files have category headers in the first data row(s)
    with actual feature names in rows below. This function detects
    such rows, uses the last non-null value as the column name, and
    removes those rows.
    Also handles rows where languages/forms are NaN (merged cells in Excel)
    by forward-filling them.
    """
    # Forward-fill languages and forms columns (Excel merged cells produce NaN)
    if 'languages' in df.columns:
        df['languages'] = df['languages'].ffill()
    if 'forms' in df.columns:
        df['forms'] = df['forms'].ffill()

    # Detect multi-row header: rows where all feature values are non-numeric
    feature_start = 2  # after languages and forms
    rows_to_drop = []
    for idx in range(len(df)):
        row = df.iloc[idx, feature_start:]
        non_null = row.dropna()
        if len(non_null) == 0:
            continue
        numeric_count = sum(pd.api.types.is_number(v) for v in non_null)
        if numeric_count == 0:
            rows_to_drop.append(idx)
        else:
            break

    if rows_to_drop:
        new_columns = list(df.columns)
        for idx in rows_to_drop:
            row = df.iloc[idx, feature_start:]
            for col_idx, val in enumerate(row):
                if pd.notna(val) and not pd.api.types.is_number(val):
                    col_name_idx = col_idx + feature_start
                    new_columns[col_name_idx] = str(val)
        df.columns = new_columns
        df = df.drop(rows_to_drop).reset_index(drop=True)

    # Convert feature columns to numeric dtype (may be object after multi-row header removal)
    # Skip non-numeric text columns (e.g., group/category columns like '语组')
    for col in df.columns[feature_start:]:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        # Check if the column looks like a text/group column (mostly non-numeric)
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        numeric_ratio = sum(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null) / len(non_null)
        if numeric_ratio > 0.5:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # Text column: forward-fill and keep as-is
            df[col] = df[col].ffill()

    return df


def validate_data(df):
    """
    Validate whether the data conforms to the specifications.
    """
    # Rule 1: First column must be 'languages'
    if df.columns[0] != 'languages':
        return "First column must be named 'languages'."

    # Rule 2: Second column must be 'forms'
    if df.columns[1] != 'forms':
        return "Second column must be named 'forms'."

    # Rule 3: At least two feature columns after the first two (excluding 'type')
    feature_count = len([c for c in df.columns[2:] if c != 'type'])
    if feature_count < 2:
        return "There must be at least two more columns (features) after 'languages' and 'forms'."

    # Rule 4: At least one row of data
    if len(df) == 0:
        return "The file must contain at least one row of data (one form)."

    # Rule 5: All columns after the first two (except 'type') must have numeric data
    feature_cols = [c for c in df.columns[2:] if c != 'type']
    if len(feature_cols) < 2:
        return "There must be at least two feature columns after 'languages' and 'forms'."
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return f"Feature '{col}' must contain only numeric (integer or float) values."

    return 'standard'



def process_label(df: pd.DataFrame, expected_columns: list):
    """
    Validate the ground truth data and standardize it.
    """
    if df.empty:
        return None

    # Get the feature names from the ground truth table's columns (skip first col).
    actual_columns = df.columns.tolist()[1:]

    # Validation Rule 1: Check whether the column names match (as sets).
    if set(actual_columns) != set(expected_columns):
        print(actual_columns, expected_columns)
        print("The ground truth data is inconsistent: the features and the co-occurrence table do not match.")
        return None

    # Validation Rule 2: Check whether the rows and column names match (as sets).
    first_column_data = df.iloc[:, 0].tolist()
    if set(first_column_data) != set(actual_columns):
        print(first_column_data, actual_columns)
        print("The ground truth data is problematic: rows and columns must be consistent.")
        return None

    # Reorder the GT matrix to match expected_columns order.
    # Build a mapping from feature name -> row index (using first column).
    row_label_to_idx = {str(label).strip(): i for i, label in enumerate(first_column_data)}
    # Reorder columns and rows to match expected_columns.
    reordered_matrix = np.zeros((len(expected_columns), len(expected_columns)))
    for i, feat_i in enumerate(expected_columns):
        row_idx = row_label_to_idx.get(str(feat_i).strip())
        if row_idx is None:
            continue
        for j, feat_j in enumerate(expected_columns):
            col_idx = list(actual_columns).index(feat_j) + 1  # +1 to skip first col
            val = df.iloc[row_idx, col_idx]
            reordered_matrix[i, j] = float(val) if pd.notna(val) else 0.0

    # Fill in the missing values.
    df.iloc[:, 0] = df.iloc[:, 0].fillna('<unk>')
    df.fillna(0, inplace=True)

    # Return the reordered ground truth adjacency matrix.
    return reordered_matrix



def process_data(df):
    """
    Standardize the tabular data and create a corresponding NumPy matrix for subsequent adjacency matrix generation.
    If a 'type' column exists, rows with type='overlay' are excluded from the co-occurrence matrix
    but their info is preserved separately for projection.
    """
    df.fillna({'languages': '<unk>', 'forms': '<unk>'}, inplace=True)
    if 'type' in df.columns:
        df['type'] = df['type'].fillna('data')
    else:
        df['type'] = 'data'
    df.fillna(0, inplace=True)
    columns = df.columns.tolist()

    # Separate overlay rows from data rows
    is_overlay = df['type'] == 'overlay'
    df_data = df[~is_overlay].reset_index(drop=True)
    overlay_info = None
    if is_overlay.any():
        overlay_df = df[is_overlay].reset_index(drop=True)
        overlay_numeric = overlay_df.iloc[:, 2:]
        overlay_numeric = overlay_numeric[[c for c in overlay_numeric.columns if c != 'type']]
        overlay_info = {
            'languages': overlay_df['languages'].tolist(),
            'forms': overlay_df['forms'].tolist(),
            'matrix': overlay_numeric.to_numpy(dtype=np.float64)
        }

    numeric_cols = df_data.iloc[:, 2:]
    numeric_cols = numeric_cols[[c for c in numeric_cols.columns if c != 'type']]
    forms = df_data[['languages', 'forms']].apply(
        lambda row: {'language': row['languages'], 'form': row['forms']}, axis=1
    ).tolist()
    co_occurrence_matrix = numeric_cols.to_numpy(dtype=np.float64)

    feature_names = [c for c in columns[2:] if c != 'type']
    return feature_names, forms, co_occurrence_matrix, overlay_info



def process_graph(graph):
    """
    Create an adjacency matrix based on the map information.
    """
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    name = graph.get("map_name", "")

    num_nodes = len(nodes)

    if not nodes:
        return None, name
    elif not edges:
        return np.zeros((num_nodes, num_nodes), dtype=float), name

    # Create a mapping from node IDs to indices.
    node_id_to_index = {node["id"]: idx for idx, node in enumerate(nodes)}

    # Initialize the adjacency matrix.
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=float)

    # Populate the adjacency matrix.
    for edge in edges:
        from_id = edge["from"]
        to_id = edge["to"]
        value = edge["value"]  # Use float type as edge weights.

        if from_id in node_id_to_index and to_id in node_id_to_index:
            i = node_id_to_index[from_id]
            j = node_id_to_index[to_id]
            adj_matrix[i][j] = value  # If it's a directed graph, assign the weight in only one direction.
            # adj_matrix[j][i] = value

    return adj_matrix, name



def process_symmetric_matrix(matrix):
    """
    Determine whether the input matrix is symmetric.
    If it is, return its upper triangular matrix with the main diagonal set to 0;
    otherwise, return the original matrix.
    """
    if not isinstance(matrix, np.ndarray):
        raise ValueError("The input must be a NumPy ndarray.")

    if matrix.shape[0] != matrix.shape[1]:
        return matrix

    if np.array_equal(matrix, matrix.T):
        # If it is a symmetric matrix, take the upper triangular part and set the diagonal to 0.
        upper_triangle = np.triu(matrix, k=1)  # k=1 means the main diagonal is excluded.
        return upper_triangle
    else:
        return matrix



def convert_to_graph_data_merged(edges:list):
    """
    Convert the edge list into the format required for frontend visualization.
    """
    graph_edges = list()
    for edge in edges:
        graph_edges.append({
            'id': f'e{uuid.uuid4().hex[:8]}',
            'from': str(edge[0]),
            'to': str(edge[1]),
            'label': str(round(edge[2], 3)),
            'value': float(round(edge[2], 3))
        })
    return graph_edges



def convert_to_graph_data(map_name, adjacency_matrix, node_labels):
    """
    Convert the adjacency matrix into the format required for frontend visualization.
    """
    nodes = []
    edges = []

    # Create nodes.
    for i, label in enumerate(node_labels):
        nodes.append({
            'id': str(i),
            'label': str(label),
            'title': str(label)
        })

    # Create edges.
    for i in range(len(adjacency_matrix)):
        for j in range(len(adjacency_matrix[i])):
            weight = adjacency_matrix[i][j]
            if weight > 0:
                edge_id = f'e{uuid.uuid4().hex[:8]}'
                edges.append({
                    'id': edge_id,
                    'from': str(i),
                    'to': str(j),
                    'label': str(round(weight, 3)),
                    'value': float(round(weight, 3))
                })

    return {'map_name': map_name, 'nodes': nodes, 'edges': edges}



def parse_sheet3_labels(sheet3_data, unique_feat_names):
    """
    Parse Sheet3 (feature name mapping table) into a dict keyed by feature name.
    Sheet3 format (from header:1 mode):
      row 0: ['英文名称', 'defeat', 'lose', ...]   or  ['功能名称', '类同', ...]
      row 1: ['中文名称', '输', '输', ...]          or  ['英文全称', 'Additive Focus', ...]
    Returns: {feature_name: {'chinese': ..., 'english': ...}} covering all unique_feat_names
    (including slash-merged names).
    """
    if not sheet3_data or len(sheet3_data) < 2:
        return {}

    row0 = sheet3_data[0]
    row1 = sheet3_data[1]
    if not row0 or not row1:
        return {}

    # Detect which row is English and which is Chinese by checking the first cell
    header0 = str(row0[0]).strip() if row0[0] is not None else ''
    header1 = str(row1[0]).strip() if row1[0] is not None else ''

    if '英文' in header0 or 'English' in header0.lower():
        en_row, cn_row = row0, row1
    elif '中文' in header0 or 'Chinese' in header0.lower():
        cn_row, en_row = row0, row1
    else:
        # Default: row0=English, row1=Chinese (matching ditransitive format)
        en_row, cn_row = row0, row1

    # Build mapping: key = English name (lowercased for matching), value = {chinese, english}
    raw_map = {}
    for i in range(1, max(len(en_row), len(cn_row))):
        en = str(en_row[i]).strip() if i < len(en_row) and en_row[i] is not None else ''
        cn = str(cn_row[i]).strip() if i < len(cn_row) and cn_row[i] is not None else ''
        if en and en != 'nan':
            en_clean = en.strip()
            cn_clean = cn if cn and cn != 'nan' else en_clean
            raw_map[en_clean.lower()] = {'chinese': cn_clean, 'english': en_clean}
            raw_map[en_clean] = {'chinese': cn_clean, 'english': en_clean}
            if cn and cn != 'nan':
                raw_map[cn_clean] = {'chinese': cn_clean, 'english': en_clean}

    # Build result dict keyed by each unique_feat_name (handle slash-merged names)
    result = {}
    for fname in unique_feat_names:
        parts = str(fname).split('/')
        chinese_parts = []
        english_parts = []
        for part in parts:
            part = part.strip()
            info = raw_map.get(part) or raw_map.get(part.lower())
            if info:
                chinese_parts.append(info['chinese'])
                english_parts.append(info['english'])
            else:
                chinese_parts.append(part)
                english_parts.append(part)
        result[str(fname)] = {
            'chinese': '/'.join(chinese_parts),
            'english': '/'.join(english_parts)
        }

    return result



def get_forms_with_nodes(df, feature_map, overlay_info=None, semantic_maps=None):
    """
    Get all forms with the given feature_map.
    feature_map is the deduplicated form-feature matrix, not the original co-occurrence matrix.
    overlay_info: dict with 'languages', 'forms', 'matrix' for overlay forms to be projected.
    semantic_maps: the SemanticMap object (needed for merge_feat_info to map overlay correctly).
    """
    col1 = df.columns[0]
    col2 = df.columns[1]

    if 'type' in df.columns:
        df_base = df[df['type'] != 'overlay'].reset_index(drop=True)
    else:
        df_base = df

    languages = df_base[col1].tolist()
    forms = df_base[col2].tolist()

    all_forms_with_nodes = list()

    for row in range(feature_map.shape[0]):
        form = forms[row]
        language = languages[row]
        related_cols = list()
        for col in range(feature_map.shape[1]):
            if feature_map[row, col] > 0:
                related_cols.append(str(col))
        all_forms_with_nodes.append({'language': language, 'form': form, 'nodes': related_cols, 'isOverlay': False})

    if overlay_info is not None:
        for i in range(len(overlay_info['forms'])):
            overlay_row = overlay_info['matrix'][i]
            merged_row = overlay_row.copy()
            if semantic_maps is not None and hasattr(semantic_maps, 'merge_feat_info') and semantic_maps.merge_feat_info:
                for target_col, merged_col in semantic_maps.merge_feat_info:
                    if merged_col < len(merged_row) and target_col < len(merged_row):
                        merged_row[target_col] = max(merged_row[target_col], merged_row[merged_col])
                keep_cols = [j for j in range(len(merged_row)) if not any(mc == j for _, mc in semantic_maps.merge_feat_info)]
                merged_row = merged_row[keep_cols]
            related_cols = list()
            for col in range(min(feature_map.shape[1], len(merged_row))):
                if merged_row[col] > 0:
                    related_cols.append(str(col))
            all_forms_with_nodes.append({
                'language': overlay_info['languages'][i],
                'form': overlay_info['forms'][i],
                'nodes': related_cols,
                'isOverlay': True
            })

    return all_forms_with_nodes



@app.route('/api/supplementary-metadata', methods=['GET'])
def supplementary_metadata():
    """
    Return feature label mapping and example sentences for the
    'supplementary adverbs' example.
    """
    try:
        import os
        base = os.path.dirname(os.path.abspath(__file__))

        # --- Feature labels: abbreviation -> {chinese, english, abbrev} ---
        label_path = os.path.join(base, 'data', '副词实例.xlsx')
        feature_labels = {}
        if os.path.exists(label_path):
            df_label = pd.read_excel(label_path, sheet_name=0, header=None)
            chinese_row = df_label.iloc[0].tolist()
            english_row = df_label.iloc[1].tolist()
            abbrev_row = df_label.iloc[2].tolist()
            for i in range(2, len(chinese_row)):
                abbr = str(abbrev_row[i]).strip() if pd.notna(abbrev_row[i]) else None
                cn = str(chinese_row[i]).strip() if pd.notna(chinese_row[i]) else None
                en = str(english_row[i]).strip() if pd.notna(english_row[i]) else None
                if abbr:
                    feature_labels[abbr] = {'chinese': cn, 'english': en, 'abbrev': abbr}
                if cn and cn not in feature_labels:
                    feature_labels[cn] = {'chinese': cn, 'english': en, 'abbrev': abbr}

        # --- Feature examples: "language|form|feature" -> example text ---
        examples_path = os.path.join(base, 'data', 'data_150625_cp.xlsx')
        feature_examples = {}
        if os.path.exists(examples_path):
            df_ex = pd.read_excel(examples_path, sheet_name=0, header=None)
            header = df_ex.iloc[0].tolist()
            feature_names_in_file = [str(h).strip() if pd.notna(h) else '' for h in header[2:]]
            df_data = df_ex.iloc[1:].reset_index(drop=True)
            # ffill language column (merged cells in Excel)
            df_data.iloc[:, 0] = df_data.iloc[:, 0].ffill()
            for row_idx in range(len(df_data)):
                lang_val = df_data.iloc[row_idx, 0]
                form_val = df_data.iloc[row_idx, 1]
                if pd.isna(form_val):
                    continue
                lang_str = str(lang_val).strip() if pd.notna(lang_val) else ''
                form_str = str(form_val).strip()
                for col_idx in range(2, len(header)):
                    cell = df_data.iloc[row_idx, col_idx]
                    if pd.notna(cell) and str(cell).strip():
                        feat_name = feature_names_in_file[col_idx - 2]
                        key = f"{lang_str}|{form_str}|{feat_name}"
                        feature_examples[key] = str(cell).strip()
                        if feat_name in feature_labels:
                            abbr = feature_labels[feat_name]['abbrev']
                            if abbr and abbr != feat_name:
                                key2 = f"{lang_str}|{form_str}|{abbr}"
                                feature_examples[key2] = str(cell).strip()
                        # Also key without language for simple lookup
                        key_no_lang = f"{form_str}|{feat_name}"
                        if key_no_lang not in feature_examples:
                            feature_examples[key_no_lang] = str(cell).strip()
                        if feat_name in feature_labels:
                            abbr = feature_labels[feat_name]['abbrev']
                            if abbr and abbr != feat_name:
                                key_no_lang2 = f"{form_str}|{abbr}"
                                if key_no_lang2 not in feature_examples:
                                    feature_examples[key_no_lang2] = str(cell).strip()

        return jsonify({
            'feature_labels': feature_labels,
            'feature_examples': feature_examples
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/evolution-data', methods=['GET'])
def evolution_data():
    try:
        import os
        base = os.path.dirname(os.path.abspath(__file__))

        ancient_path = os.path.join(base, 'data', 'supplementary adverbs_ancient.xlsx')
        if not os.path.exists(ancient_path):
            return jsonify({'error': 'Ancient adverbs data file not found'}), 404

        df = pd.read_excel(ancient_path, sheet_name=0)

        df['period'] = df['period'].ffill()

        label_path = os.path.join(base, 'data', '副词实例.xlsx')
        feature_labels = {}
        if os.path.exists(label_path):
            df_label = pd.read_excel(label_path, sheet_name=0, header=None)
            chinese_row = df_label.iloc[0].tolist()
            english_row = df_label.iloc[1].tolist()
            abbrev_row = df_label.iloc[2].tolist()
            for i in range(2, len(chinese_row)):
                abbr = str(abbrev_row[i]).strip() if pd.notna(abbrev_row[i]) else None
                cn = str(chinese_row[i]).strip() if pd.notna(chinese_row[i]) else None
                en = str(english_row[i]).strip() if pd.notna(english_row[i]) else None
                if abbr:
                    feature_labels[abbr] = {'chinese': cn, 'english': en, 'abbrev': abbr}
                if cn and cn not in feature_labels:
                    feature_labels[cn] = {'chinese': cn, 'english': en, 'abbrev': abbr}

        verb_labels = {
            '返回_动': {'chinese': '返回', 'english': 'Return', 'type': 'verb'},
            '偿还_动': {'chinese': '偿还', 'english': 'Repay', 'type': 'verb'},
            '改变_动': {'chinese': '改变', 'english': 'Change', 'type': 'verb'},
            '重复_动': {'chinese': '重复', 'english': 'Repeat', 'type': 'verb'},
            '二_数': {'chinese': '二', 'english': 'Two', 'type': 'number'},
            '恢复_动': {'chinese': '恢复', 'english': 'Restore', 'type': 'verb'},
        }

        all_feature_cols = [c for c in df.columns if c not in ('period', 'languages', 'forms', 'type')]
        verb_cols = [c for c in all_feature_cols if c.endswith('_动') or c.endswith('_数')]
        function_cols = [c for c in all_feature_cols if c not in verb_cols]

        period_order = ['上古', '中古', '近代', '现代']

        words = []
        seen_words = set()
        for w in df['forms'].tolist():
            w = str(w)
            if w not in seen_words:
                words.append(w)
                seen_words.add(w)

        df = df.fillna(0)

        # --- Corpus text from the ancient corpus file ---
        corpus_path = os.path.join(base, 'data', 'supplementary adverbs_ancient_corpus.xlsx')
        corpus = {}
        if os.path.exists(corpus_path):
            df_corpus = pd.read_excel(corpus_path, sheet_name=0)
            df_corpus['period'] = df_corpus['period'].ffill()
            # Align function columns by name with the ancient data
            corpus_func_cols = [c for c in df_corpus.columns
                                if c in function_cols]
            corpus_lang_col = 'languages' if 'languages' in df_corpus.columns else df_corpus.columns[1]
            for word in words:
                corpus[word] = {}
                wmask = df_corpus['forms'].astype(str) == word
                for period in period_order:
                    pmask = df_corpus['period'].astype(str) == period
                    rows = df_corpus[wmask & pmask]
                    per_func = {}
                    for col in corpus_func_cols:
                        entries = []
                        for _, row in rows.iterrows():
                            val = row[col]
                            if pd.isna(val):
                                continue
                            sval = str(val).strip()
                            if not sval:
                                continue
                            # Skip pure numeric markers (0/1) and placeholders
                            if sval in ('0', '1', '0.0', '1.0', 'MISSING', 'nan', 'NaN'):
                                continue
                            dynasty = str(row[corpus_lang_col]).strip() if pd.notna(row[corpus_lang_col]) else ''
                            entries.append({'dynasty': dynasty, 'text': sval})
                        if entries:
                            per_func[col] = entries
                    corpus[word][period] = per_func

        data = {}
        for word in words:
            data[word] = {}
            word_mask = df['forms'].astype(str) == word

            for period in period_order:
                period_mask = df['period'].astype(str) == period
                mask = word_mask & period_mask
                period_rows = df[mask]

                functions_union = set()
                verbs_union = set()

                for _, row in period_rows.iterrows():
                    for col in function_cols:
                        val = row[col]
                        if val == 1 or val == 1.0:
                            functions_union.add(col)
                    for col in verb_cols:
                        val = row[col]
                        if val == 1 or val == 1.0:
                            verbs_union.add(col)

                data[word][period] = {
                    'functions': sorted(list(functions_union)),
                    'verbs': sorted(list(verbs_union))
                }

        return jsonify({
            'dynasties': period_order,
            'words': words,
            'function_features': function_cols,
            'verb_features': verb_cols,
            'feature_labels': feature_labels,
            'verb_labels': verb_labels,
            'data': data,
            'corpus': corpus
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/sla-concept-space', methods=['POST'])
def sla_concept_space():
    """
    Generate a concept space for SLA view based on filtered Sheet1 data
    and active features from Sheet5 for the selected proficiency level.
    """
    try:
        sheet1_data = request.json.get('sheet1_data', [])
        sheet5_features = request.json.get('sheet5_features', [])
        form = request.json.get('form', '')
        native_lang = request.json.get('native_lang', '')
        target_lang = request.json.get('target_lang', '')
        level = request.json.get('level', '')
        calc_type = request.json.get('calc_type', 'J')
        blend_weight = request.json.get('blend_weight', 0.1)

        if not sheet1_data:
            return jsonify({'error': 'Missing required data.'}), 400

        # Build a DataFrame from the filtered Sheet1 rows — keep ALL feature columns
        df = pd.DataFrame(sheet1_data)

        # Fill missing values with 0 for feature columns
        feature_cols = [c for c in df.columns if str(c).strip() not in ('languages', 'forms', 'type')]
        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        # Normalize column names
        df, norm_result = normalize_columns(df)
        if norm_result != 'standard':
            return jsonify({'error': norm_result}), 400

        df = preprocess_multirow_header(df)
        df = filter_non_numeric_features(df)

        v_result = validate_data(df)
        if v_result != 'standard':
            return jsonify({'error': v_result}), 400

        features, forms, co_occurrence_matrix, overlay_info = process_data(df)

        # Determine llm_sim source
        llm_sim_path = None
        llm_sim_data = None
        if calc_type == 'L':
            llm_sim_csv = request.json.get('llm_sim_csv', None)
            if llm_sim_csv:
                llm_sim_data = pd.read_csv(io.StringIO(llm_sim_csv), index_col=0)

        semantic_maps = SemanticMap(
            co_occurrence_matrix, features, forms, None, None,
            calc_type=calc_type, blend=(blend_weight, True),
            llm_sim_path=llm_sim_path, llm_sim_data=llm_sim_data, cooc_scale="minmax")

        matrix_data = semantic_maps.get_all_matrix()
        node_labels = semantic_maps.unique_featNames

        graph_data = list()
        trees_matrix_data = matrix_data.get('trees')
        for index, tree_matrix in enumerate(trees_matrix_data):
            tree_matrix = process_symmetric_matrix(tree_matrix)
            tree_graph_data = convert_to_graph_data(
                'SLA Concept Map ' + str(index + 1), tree_matrix, node_labels)
            tree_evaluation_metric = semantic_maps.get_evaluation_metric(
                tree_matrix, range(len(forms)))
            tree_graph_data['evaluation_metric'] = tree_evaluation_metric
            graph_data.append(tree_graph_data)

        forms_with_nodes = get_forms_with_nodes(df, semantic_maps.tfM, overlay_info, semantic_maps)

        return jsonify({
            'graph_data': graph_data,
            'forms_with_nodes': forms_with_nodes,
            'active_features': sheet5_features
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port)