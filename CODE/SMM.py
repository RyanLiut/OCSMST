import numpy as np
import networkx as nx
import copy
import time
import os
import io

import pandas as pd
from collections import Counter
from itertools import combinations


class SemanticMap(object):

    def __init__(self, tfM, featNames, formNames, adjM=None, GT_adj=None, calc_type='G', blend=None, llm_sim_path=None, llm_sim_data=None, cooc_scale=None):
        # form-feature matrix    np.array  (N, D)
        self.tfM = tfM
        # feature map    dict(int: str)
        self.origin_featNames = featNames
        # deduplicated feature list
        self.unique_featNames = []
        # form list
        self.formNames = formNames
        # the ground truth adjacency matrix
        self.GT_adj = GT_adj
        # calculation method
        self.calc_type = calc_type
        # blend parameter: controls both the similarity weight and whether the
        # co-occurrence is normalized (each form's row divided by its edge
        # count - 1, i.e. the former 'N' scheme).
        # Accepted forms:
        #   - number w            -> (w, False): raw co-occurrence, weight w
        #   - (w, normalized)     -> explicit weight + normalization flag
        #   - 'N' / 'normalized'  -> normalized co-occurrence, type-specific default weight
        #   - None                -> type-specific default weight, raw co-occurrence
        # Type-specific default weights: G=zeroOcc, J=0.1, D=1.0, L=0.5
        print("1blend: ", blend)
        self.blend = blend
        # fully connected adjacency matrix
        # can be a numpy array, a path to a CSV file, or None
        self.adjM = adjM
        # path to LLM embedding similarity matrix CSV (for calc_type='L')
        self.llm_sim_path = llm_sim_path
        # LLM embedding similarity matrix as a DataFrame (for calc_type='L')
        self.llm_sim_data = llm_sim_data
        # Optional scaling applied to the co-occurrence term that is blended
        # with Jaccard/Dice/LLM similarity. It only affects the blend weight;
        # Jaccard, Dice, feature frequencies and the graph structure keep the
        # raw counts unless the scaling itself maps a pair to zero.
        #   None (default)  -> raw co-occurrence counts (legacy behaviour)
        #   'minmax'        -> (x - min) / (max - min) over the off-diagonal
        #   'max'           -> x / max over the off-diagonal
        self.cooc_scale = cooc_scale
        # graph corresponding to the fully connected adjacency matrix
        self.G = None
        # candidate tree list
        self.trees = []
        # record deduplication information    list
        self.merge_feat_info = []
        # No feature deduplication — keep all original columns
        self.unique_featNames = list(self.origin_featNames)
        # if adjM is a path, load the similarity matrix from CSV after
        # deduplication so columns can be checked against unique_featNames
        if isinstance(self.adjM, str):
            self.adjM = self._load_adjM_from_csv(self.adjM)
        


    def merge_feat(self):
        """
        merge the feature columns if they are totally the same.
        """
        unique_featNames = []
        unique_columns = []
        for i, col in enumerate(self.tfM.T.tolist()):
            if col not in unique_columns or sum(col) == 0:
                unique_columns.append(col)
                unique_featNames.append(self.origin_featNames[i])
            else:
                ix = unique_columns.index(col)
                unique_featNames[ix] += "/" + self.origin_featNames[i]
                self.merge_feat_info.append((ix, i)) #  the i-th feature is merged into the ix-th feature.

        # update form-feature matrix
        self.tfM = np.stack(unique_columns, axis=1)
        # update deduplicated feature list
        self.unique_featNames = unique_featNames



    def _load_feat_name_mapping(self):
        """
        Build feature name mappings from data/副词实例.xlsx.
        row 0 = Chinese, row 1 = English full name, row 2 = English abbreviation.
        Returns a list of unified mapping dicts, each covering all 18 features:
          cn->en, en->cn, cn->abbr, abbr->cn, en->abbr, abbr->en
        """
        mapping_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'data', '副词实例.xlsx')
        mappings = []
        if os.path.isfile(mapping_file):
            try:
                df = pd.read_excel(mapping_file, header=None)
                cn = [str(df.iloc[0, j]).strip() for j in range(1, df.shape[1])]
                en = [str(df.iloc[1, j]).strip() for j in range(1, df.shape[1])]
                abbr = [str(df.iloc[2, j]).strip() for j in range(1, df.shape[1])]
                for i in range(len(cn)):
                    if cn[i] == 'nan' or en[i] == 'nan' or abbr[i] == 'nan':
                        continue
                pairs = [(cn, en), (en, cn),
                         (cn, abbr), (abbr, cn),
                         (en, abbr), (abbr, en)]
                for src, dst in pairs:
                    mappings.append({src[i]: dst[i] for i in range(len(src))
                                     if src[i] != 'nan' and dst[i] != 'nan'})
            except Exception:
                pass
        return mappings



    @staticmethod
    def _translate_feat_names(names, mapping):
        """
        Translate a list of feature names using the given mapping.
        Handles merged names containing '/' by translating each part.
        """
        result = []
        for name in names:
            if '/' in name:
                parts = [mapping.get(p, p) for p in name.split('/')]
                result.append('/'.join(parts))
            else:
                result.append(mapping.get(name, name))
        return result



    def _load_adjM_from_csv(self, csv_path):
        """
        Load the adjacency/similarity matrix from a CSV file. The CSV's
        columns (and index) must match self.unique_featNames, supporting
        both Chinese and English feature names (auto-translated via
        data/副词实例.xlsx). The returned matrix is reordered to match
        the order of self.unique_featNames.
        """
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"adjM CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path, index_col=0)
        columns = list(df.columns)
        index = list(df.index)
        target = self.unique_featNames

        # try direct match first
        if columns == target and index == target:
            return df.values

        # try translation mappings (cn<->en, cn<->abbr, en<->abbr)
        mappings = self._load_feat_name_mapping()

        for mapping in mappings:
            translated_cols = self._translate_feat_names(columns, mapping)
            translated_index = self._translate_feat_names(index, mapping)
            if translated_cols == target and translated_index == target:
                return df.values

        # if no direct or translated match, raise error
        raise ValueError(
            f"CSV columns/index do not match unique_featNames "
            f"(checked Chinese, English, and abbreviations).\n"
            f"  CSV columns:      {columns}\n"
            f"  unique_featNames: {target}")



    def add_node(self, adj_matrix, original_index, insert_index):
        """
        在邻接矩阵中添加一个新节点，该节点与指定节点具有相同的邻接关系，并与指定节点相连

        参数:
        adj_matrix (np.ndarray): 原始邻接矩阵 (n x n)
        original_index (int): 要复制的原始节点索引 (0-based)
        insert_index (int): 新节点要插入的位置索引 (0-based)

        返回:
        np.ndarray: 新的邻接矩阵 ((n+1) x (n+1))
        """
        n = adj_matrix.shape[0]

        # 创建新的 (n+1) x (n+1) 矩阵
        new_adj_matrix = np.zeros((n + 1, n + 1), dtype=adj_matrix.dtype)

        # 复制原始矩阵数据（跳过插入位置）
        # 上半部分（插入位置之前）
        new_adj_matrix[:insert_index, :insert_index] = adj_matrix[:insert_index, :insert_index]
        new_adj_matrix[:insert_index, insert_index + 1:] = adj_matrix[:insert_index, insert_index:]

        # 下半部分（插入位置之后）
        new_adj_matrix[insert_index + 1:, :insert_index] = adj_matrix[insert_index:, :insert_index]
        new_adj_matrix[insert_index + 1:, insert_index + 1:] = adj_matrix[insert_index:, insert_index:]

        # 确定原始节点在新矩阵中的位置
        pos_old = original_index if original_index < insert_index else original_index + 1

        # 复制原始节点的邻接关系到新节点
        # 处理插入位置之前的列
        if insert_index > 0:
            new_adj_matrix[insert_index, :insert_index] = adj_matrix[original_index, :insert_index]
            new_adj_matrix[:insert_index, insert_index] = adj_matrix[original_index, :insert_index]

        # 处理插入位置之后的列
        if insert_index < n:
            new_adj_matrix[insert_index, insert_index + 1:] = adj_matrix[original_index, insert_index:]
            new_adj_matrix[insert_index + 1:, insert_index] = adj_matrix[original_index, insert_index:]

        # 设置新节点与原始节点之间的邻接关系为1
        new_adj_matrix[insert_index, pos_old] = 1
        new_adj_matrix[pos_old, insert_index] = 1

        return new_adj_matrix



    def update_record(self, record, insert_index):
        """
        update the record of merged feature columns
        """
        for original_index, new_original_index in record.items():
            if new_original_index >= insert_index:
                record[original_index] = new_original_index + 1
        return record



    def get_unmerged_matrix(self, adj_matrix):
        """
        restore the merged feature columns to maintain consistency with the features in the form-feature matrix
        """
        record = dict()
        for i in range(adj_matrix.shape[0]):
            record[i] = i
        for original_index, insert_index in self.merge_feat_info:
            new_original_index = record[original_index]
            adj_matrix = self.add_node(adj_matrix, new_original_index, insert_index)
            record = self.update_record(record, insert_index)
        return adj_matrix



    def _load_llm_sim_matrix(self, n_features):
        """
        Load the LLM embedding similarity matrix from self.llm_sim_path
        or self.llm_sim_data.
        Columns/index must match self.unique_featNames (supports Chinese,
        English, and abbreviations). Returns an (n_features, n_features)
        numpy array.
        """
        if self.llm_sim_data is not None:
            df = self.llm_sim_data
        elif self.llm_sim_path:
            if not os.path.isfile(self.llm_sim_path):
                raise FileNotFoundError(
                    f"LLM similarity matrix not found: {self.llm_sim_path}")
            df = pd.read_csv(self.llm_sim_path, index_col=0)
        else:
            raise ValueError(
                "calc_type='L' requires llm_sim_path or llm_sim_data to be provided")
        columns = list(df.columns)
        index = list(df.index)
        target = self.unique_featNames

        if columns == target and index == target:
            return df.values.astype(float)

        mappings = self._load_feat_name_mapping()
        for mapping in mappings:
            translated_cols = self._translate_feat_names(columns, mapping)
            translated_index = self._translate_feat_names(index, mapping)
            if translated_cols == target and translated_index == target:
                return df.values.astype(float)

        raise ValueError(
            f"LLM similarity matrix columns/index do not match "
            f"unique_featNames (checked Chinese, English, and abbreviations).\n"
            f"  CSV columns:      {columns}\n"
            f"  unique_featNames: {target}")



    @staticmethod
    def scale_cooccurrence(cooccurrence, scale=None):
        """Scale the co-occurrence counts used in the J/D/L blend term.

        Returns a new symmetric matrix in [0, 1] (or the untouched input when
        scale is None). Jaccard/Dice/frequencies keep the raw counts.
        """
        if not scale:
            return cooccurrence
        values = np.array(cooccurrence, dtype=float)
        rows, cols = values.shape
        upper = np.triu_indices(rows, 1)
        off = values[upper]
        if scale == 'minmax':
            lo, hi = float(off.min()), float(off.max())
            if hi - lo <= 0:
                scaled = np.zeros_like(values)
            else:
                scaled = (values - lo) / (hi - lo)
        elif scale == 'max':
            hi = float(np.abs(off).max())
            scaled = values / hi if hi > 0 else np.zeros_like(values)
        else:
            raise ValueError(
                "cooc_scale must be None, 'minmax' or 'max', got %r" % (scale,))
        scaled = np.triu(scaled, 1)
        scaled = scaled + scaled.T
        np.fill_diagonal(scaled, 0)
        return scaled

    def calculate_semantic_relations(self, matrix: np.ndarray,
                                     calc_type: str = 'G') -> np.ndarray:
        """
        compute the weights in the adjacency matrix

        self.blend controls both the similarity weight and whether the
        co-occurrence count is normalized (the former 'N' scheme, where each
        form's row is divided by its number of edges - 1):
          - number w            -> raw co-occurrence, weight w
          - (w, normalized)      -> weight w + normalization flag
          - 'N' / 'normalized'   -> normalized co-occurrence, default weight
          - None                 -> default weight, raw co-occurrence
        """
        # input validation
        if not isinstance(matrix, np.ndarray):
            raise ValueError("the input must be a NumPy array")

        if matrix.ndim != 2:
            raise ValueError("the input must be a 2D array")

        if calc_type not in ['G', 'J', 'D', 'L']:
            raise ValueError("the computation type must be 'G', 'J', 'D', or 'L'")

        n_forms, n_features = matrix.shape

        # parse self.blend into (weight, normalized)
        print("blend: ", self.blend)
        if self.blend is None:
            weight, normalized = None, False
        elif isinstance(self.blend, tuple):
            weight, normalized = self.blend
        elif isinstance(self.blend, str) and self.blend in ('N', 'normalized'):
            weight, normalized = None, True
        elif isinstance(self.blend, (int, float, np.floating, np.integer)):
            weight, normalized = float(self.blend), False
        else:
            raise ValueError(
                "blend must be a number, a (weight, normalized) tuple, "
                "or 'N'/'normalized'")

        # Type-specific default weights: G=zeroOcc, J=0.1, D=1.0, L=0.5
        if calc_type == 'G':
            weight = weight if weight is not None else 0.001
        elif calc_type == 'J':
            weight = weight if weight is not None else 0.1
        elif calc_type == 'D':
            weight = weight if weight is not None else 0.2
        elif calc_type == 'L':
            weight = weight if weight is not None else 0.4

        # compute the (possibly normalized) co-occurrence matrix and feature
        # frequencies. Normalization divides each form's row by (|V_m| - 1),
        # so both co-occurrence and feature frequencies stay consistent.
        print("normalized: ", normalized)

        # -1 marks a feature that is explicitly absent for a form. It only
        # takes effect for Non-coocur (calc_type='G' with weight > 0), which
        # uses a binary (0/1) matrix for co-occurrence and a signed
        # (-1/0/1) matrix for the supplementary signal. Every other option -
        # NoBlend, Jaccard, Dice and EmbSim - treats -1 exactly like 0.
        non_coocur = calc_type == 'G' and weight > 0
        has_neg_one = bool(np.any(matrix == -1))
        if has_neg_one and not non_coocur:
            matrix = np.where(matrix == -1, 0.0, matrix)
            has_neg_one = False

        if calc_type == 'G':
            if weight > 0 and not has_neg_one:
                raise ValueError(
                    "Non-coocur (G) blend requires -1 values in the data "
                    "to indicate explicitly absent features. This dataset "
                    "does not contain -1 values, so Non-coocur is not available. "
                    "Please use Jaccard, Dice, or EmbSim instead.")

            # NoBlend (weight 0) and datasets without -1: plain co-occurrence
            # only, without entering any blend branch.
            if not has_neg_one:
                if normalized:
                    row_sums = matrix.sum(axis=1)
                    denom = row_sums - 1
                    denom = np.where(denom <= 0, 1, denom)
                    norm_matrix = matrix / denom[:, np.newaxis]
                    norm_matrix[row_sums <= 1, :] = 0
                    adj_matrix = norm_matrix.T @ matrix
                    adj_matrix = np.round(adj_matrix, decimals=3)
                else:
                    adj_matrix = matrix.T @ matrix
                np.fill_diagonal(adj_matrix, 0)
                return adj_matrix

            binary = np.where(matrix > 0, 1, 0).astype(float)
            signed = matrix.astype(float)

            if normalized:
                row_sums = binary.sum(axis=1)
                denom = row_sums - 1
                denom = np.where(denom <= 0, 1, denom)
                norm_binary = binary / denom[:, np.newaxis]
                norm_binary[row_sums <= 1, :] = 0
                norm_signed = signed / denom[:, np.newaxis]
                norm_signed[row_sums <= 1, :] = 0
                cooccurrence = norm_binary.T @ binary
                cooccurrence = np.round(cooccurrence, decimals=3)
                signed_cross = norm_signed.T @ signed
                supplementary = np.round(signed_cross - cooccurrence, decimals=3)
                feature_freq = norm_binary.sum(axis=0)
            else:
                cooccurrence = binary.T @ binary
                supplementary = signed.T @ signed - cooccurrence
                feature_freq = np.sum(binary, axis=0)

            print("weight: ", weight, "cooccurrence: ", cooccurrence, "supplementary: ", supplementary)
            adj_matrix = cooccurrence * (1 - weight) + supplementary * weight
            np.fill_diagonal(adj_matrix, 0)
            return adj_matrix

        elif normalized:
            row_sums = matrix.sum(axis=1)  # shape: (M,)
            denom = row_sums - 1
            denom = np.where(denom <= 0, 1, denom)  # shape: (M,)
            normalized_matrix = matrix / denom[:, np.newaxis]  # shape: (M, N)
            # forms with only 1 node contribute nothing
            normalized_matrix[row_sums <= 1, :] = 0
            cooccurrence = normalized_matrix.T @ matrix
            cooccurrence = np.round(cooccurrence, decimals=3)
            feature_freq = normalized_matrix.sum(axis=0)
        else:
            cooccurrence = matrix.T @ matrix
            feature_freq = np.sum(matrix, axis=0)  # shape: (n_features,)

        # weight 0 switches the blend term off: return the plain co-occurrence
        # so every blend type at weight 0 gives exactly the NoBlend result.
        if weight == 0:
            adj_matrix = np.array(cooccurrence, copy=True)
            np.fill_diagonal(adj_matrix, 0)
            return adj_matrix

        # optional scaling of the co-occurrence term used in the blend (J/D/L)
        blend_cooc = self.scale_cooccurrence(cooccurrence, self.cooc_scale)

        # initialize the adjacency matrix
        adj_matrix = np.zeros((n_features, n_features))

        if calc_type == 'J':
            for i in range(n_features):
                for j in range(n_features):
                    if i == j:
                        adj_matrix[i, j] = 0.0
                    else:
                        union = feature_freq[i] + feature_freq[j] - cooccurrence[i, j]
                        if union > 0:
                            jaccard_val = cooccurrence[i, j] / union
                            adj_matrix[i, j] = weight * jaccard_val + (1 - weight) * blend_cooc[i, j]
                            # print("jaccard_val: ", jaccard_val, "cooccurrence: ", cooccurrence[i, j], "weight: ", weight, "union: ", union)
                        else:
                            adj_matrix[i, j] = 0.0

        elif calc_type == 'D':
            for i in range(n_features):
                for j in range(n_features):
                    if i == j:
                        adj_matrix[i, j] = 0.0
                    else:
                        total = feature_freq[i] + feature_freq[j]
                        if total > 0:
                            dice_val = 2 * cooccurrence[i, j] / total
                            adj_matrix[i, j] = weight * dice_val + (1 - weight) * blend_cooc[i, j]
                        else:
                            adj_matrix[i, j] = 0.0

        elif calc_type == 'L':
            llm_sim = self._load_llm_sim_matrix(n_features)
            adj_matrix = weight * llm_sim + (1 - weight) * blend_cooc
            print("llm_sim: ", llm_sim, "cooccurrence: ", cooccurrence, "weight: ", weight)
            np.fill_diagonal(adj_matrix, 0)

        return adj_matrix



    def connected_graph(self):
        """
        To construct an undirected acyclic (fully connected) graph according to a term-feature matrix
        """

        if self.adjM is None:
            adj_matrix = self.calculate_semantic_relations(self.tfM, self.calc_type)
            self.adjM = np.triu(adj_matrix) # get the upper triangular matrix

        # set the diagonal of the matrix to zero
        np.fill_diagonal(self.adjM, val=0)
        self.G = nx.from_numpy_array(self.adjM)



    def get_optimal_SpanningTrees(self, top_n=5, max_tied_trees=6000):
        """
        Select candidate maps using the application's original policy.
        Defaults retain the original five-candidate behavior: enumerate maximum
        trees (at most 6000), sort ties by degree std, sample across that sorted
        pool, and top up if needed. top_n generalizes the SAME selection policy;
        different requested counts need not be nested prefixes after sampling.
        """

        if not isinstance(top_n, (int, np.integer)) or top_n < 1:
            raise ValueError("top_n must be a positive integer")
        if not isinstance(max_tied_trees, (int, np.integer)) or max_tied_trees < 0:
            raise ValueError("max_tied_trees must be a non-negative integer (0 = unlimited)")
        self.trees = []

        self.connected_graph()
        selection_start = time.perf_counter()
        degree_time = 0.0
        sorting_time = 0.0
        sampling_time = 0.0
        enumerated = 0
        truncated = False
        source_ranks = {}

        def signature(tree):
            return frozenset(frozenset((u, v)) for u, v in tree.edges())

        start_trees = time.time()
        trees = nx.algorithms.tree.SpanningTreeIterator(self.G, minimum=False)
        end_trees = time.time()
        print("trees time: ", (end_trees - start_trees)*1000)
        try:
            start_num = time.time()
            number_trees = nx.algorithms.tree.number_of_spanning_trees(self.G)
            end_num = time.time()
            print("Spanning Trees: ", number_trees, (end_num-start_num)*1000)
        except AttributeError:
            print("Spanning Trees: (number_of_spanning_trees not available in this networkx version)")

        # Phase 1: collect all maximum-weight trees (with std), and if unique max,
        # keep collecting by weight until we have 5 or run out.
        max_weight = 0
        start_std = time.time()
        collect_top5 = False
        for index, tree in enumerate(trees):
            enumerated += 1
            total_weight = sum(d['weight'] for u, v, d in tree.edges(data=True))
            if index == 0:
                max_weight = total_weight

            if not collect_top5 and total_weight < max_weight:
                if len(self.trees) == 1:
                    collect_top5 = True
                else:
                    print(f"Number of trees with most weight (if less than 6000): {index}")
                    break

            if not collect_top5 and max_tied_trees and index >= max_tied_trees:
                truncated = True
                print(f"Number of trees with most weight (if less than 6000): {index}")
                break

            if collect_top5 and len(self.trees) >= top_n:
                break
            if not collect_top5:
                degree_start = time.perf_counter()
                deg = [d for _, d in tree.degree()]
                deg_std = np.std(deg)
                degree_time += time.perf_counter() - degree_start

            self.trees.append((tree, deg_std))
            source_ranks[signature(tree)] = index + 1
        end_std = time.time()
        print("std time: ", (end_std - start_std)*1000)

        pool_count = len(self.trees)
        if collect_top5:
            print(f"Unique maximum spanning tree. Collected top {len(self.trees)} trees by weight.")
            self.trees = [item[0] for item in self.trees]
        else:
            print(f"{len(self.trees)} trees tie for maximum weight. Sorting by std.")
            start_sort = time.time()
            sorting_start = time.perf_counter()
            self.trees = sorted(self.trees, key=lambda x: x[1], reverse=False)
            sorting_time += time.perf_counter() - sorting_start
            end_sort = time.time()
            print("sort time: ", (end_sort - start_sort)*1000)

            start_sample = time.time()
            sampling_start = time.perf_counter()
            if len(self.trees) > top_n:
                new_trees= []
                new_trees.append(self.trees[0])
                if top_n > 2:
                    all_selected_index = np.linspace(1, len(self.trees)-2, top_n - 2, dtype=int)
                    for selected_index in all_selected_index:
                        new_trees.append(self.trees[selected_index])
                if top_n > 1:
                    new_trees.append(self.trees[-1])
                self.trees = new_trees

            self.trees = [item[0] for item in self.trees]
            sampling_time += time.perf_counter() - sampling_start
            end_sample = time.time()
            print("sample time: ", (end_sample-start_sample)*1000)

        # Phase 2: if fewer than 5 trees, top up from the next-best trees
        # by continuing the iterator (only works if iterator wasn't exhausted)
        if len(self.trees) < top_n:
            print(f"Only {len(self.trees)} trees collected, topping up from next-best...")
            # Re-create the iterator to get remaining trees
            trees2 = nx.algorithms.tree.SpanningTreeIterator(self.G, minimum=False)
            seen_edges = set()
            # Build a set of edge signatures already collected
            for t in self.trees:
                sig = frozenset(frozenset((u, v)) for u, v in t.edges())
                seen_edges.add(sig)
            for source_rank, tree in enumerate(trees2, 1):
                enumerated += 1
                sig = frozenset(frozenset((u, v)) for u, v in tree.edges())
                if sig in seen_edges:
                    continue
                seen_edges.add(sig)
                self.trees.append(tree)
                source_ranks[signature(tree)] = source_rank
                print(f"  Topped up: tree {len(self.trees)}, weight={sum(d['weight'] for _,_,d in tree.edges(data=True)):.3f}")
                if len(self.trees) >= top_n:
                    break
            print(f"Final tree count: {len(self.trees)}")

        selection_time = time.perf_counter() - selection_start
        self.tree_selection_stats = {
            'protocol': 'smm_native_v1', 'top_n': int(top_n),
            'max_tied_trees': int(max_tied_trees), 'enumerated_trees': enumerated,
            'candidate_pool_count': pool_count, 'truncated': truncated,
            'unique_maximum': collect_top5,
            'selected_iterator_ranks': [source_ranks[signature(tree)] for tree in self.trees],
            'degree_std_time_s': degree_time, 'sorting_time_s': sorting_time + sampling_time,
            'sampling_time_s': sampling_time,
            'generation_time_s': max(0., selection_time - degree_time - sorting_time - sampling_time),
            'selection_time_s': selection_time,
        }



    def get_poss_subgraph(self, graph, max_size=5):
        """
        Find all possible connected subgraphs in the current graph.
        :param graph: current graph
        :return: list of connected subgraphs
        """
        nodes = list(graph.nodes)
        poss_subG_list = []

        for r in range(2, max_size):
            for node_combination in combinations(nodes, r):
                sub_graph = graph.subgraph(node_combination)
                if nx.is_connected(sub_graph):
                    poss_subG_list.append(sub_graph)
        return poss_subG_list



    def count_connected_subgraphs(self, graph, max_size):
        """
        Count all connected subgraphs of size 2 to max_size-1 using BFS
        growth from the minimum node to avoid duplicates. Only counts (no
        subgraph objects created), much faster than get_poss_subgraph.
        """
        n = graph.number_of_nodes()
        if n < 2 or max_size <= 2:
            return 0

        adj = [set(graph.neighbors(i)) for i in range(n)]
        total = 0
        upper = max_size - 1  # original loop: range(2, max_size)

        for start in range(n):
            visited = set()
            init_candidates = {v for v in adj[start] if v > start}
            stack = [(frozenset([start]), init_candidates)]
            while stack:
                current, candidates = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                if len(current) >= 2:
                    total += 1
                if len(current) >= upper:
                    continue
                for v in candidates:
                    new_set = current | {v}
                    if new_set not in visited:
                        new_candidates = (candidates - {v}) | {
                            w for w in adj[v] if w > start and w not in new_set
                        }
                        stack.append((new_set, new_candidates))
        return total



    def norm_matrix(self, matrix):
        # 确保输入是方阵
        assert matrix.shape[0] == matrix.shape[1], "Input matrix must be square"
        # 将下三角的数据累加到上三角
        result = np.triu(matrix, 1) + np.tril(matrix, -1).T
        result = result + result.T
        return result



    def evaluate_against_gt(self, pre_matrix):
        """
        Evaluate accuracy, precision, recall, and F1 score against the ground truth
        :param pre_matrix: predicted adjacency matrix
        :return: accuracy, precision, recall, and F1 score
        """
        try:
            if self.GT_adj is None:
                print("Ground truth adjacency matrix is not provided.")
                return None

            pre_matrix = self.get_unmerged_matrix(pre_matrix)
            pre_matrix = self.norm_matrix(pre_matrix)
            gt_matrix = self.norm_matrix(self.GT_adj)
            # gt_matrix = self.norm_matrix(gt_matrix)
            acc = np.sum((pre_matrix!=0) == (gt_matrix!=0)) / (gt_matrix.shape[0] ** 2)

            true_edges = (gt_matrix != 0)
            pred_edges = (pre_matrix != 0)

            # tp: predicted edge exists and ground truth edge exists
            tp = np.sum(true_edges & pred_edges)
            # fp: predicted edge exists but ground truth edge does not exist
            fp = np.sum(~true_edges & pred_edges)
            # fn: ground truth edge exists but predicted edge does not exist
            fn = np.sum(true_edges & ~pred_edges)
            # tn: ground truth edge does not exist and predicted edge does not exist
            # tn = np.sum(~true_edges & ~pred_edges)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (fn + tp) if (fn + tp) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            return {"acc": acc, "precision": precision, "recall": recall, "f1": f1}
        except Exception as e:
            print(e)
            return {}



    def find_cycles_networkx(self, nx_graph):
        """
        使用NetworkX快速找到环的个数和节点信息

        参数:
        nx_graph: NetworkX图对象

        返回:
        tuple: (环的总数, 环信息列表)
        """
        # 获取所有环的基础集合
        cycles = nx.cycle_basis(nx_graph)

        cycle_mean_length = 0
        for i, cycle in enumerate(cycles):
            cycle_mean_length += 1/len(cycle)

        return len(cycles), cycle_mean_length/len(cycles) if cycle_mean_length > 0 else 0



    def get_gt_rank(self, time_limit=10):
        """
        Find the rank of the best-GT-matching spanning tree among all spanning trees
        sorted by weight (descending). Returns (rank, f1) where rank is 1-indexed.
        If GT is a tree, looks for exact edge match. If GT has extra edges, looks for
        a tree that is a subset of GT. Otherwise, finds the tree with highest F1.
        Returns None if GT is not available.
        Time-limited: if not found within time_limit seconds, returns (>count, best_f1).
        """
        if self.GT_adj is None:
            return None

        n_dedup = len(self.unique_featNames)
        gt_norm = self.norm_matrix(self.GT_adj)

        # Map GT from original space to dedup space
        merged_set = set()
        target_map = {}
        for kept_idx, merged_idx in self.merge_feat_info:
            merged_set.add(merged_idx)
            target_map.setdefault(kept_idx, []).append(merged_idx)
        origin_to_dedup = {}
        dedup_idx = 0
        for oi in range(gt_norm.shape[0]):
            if oi in merged_set:
                continue
            origin_to_dedup[oi] = dedup_idx
            for mi in target_map.get(oi, []):
                origin_to_dedup[mi] = dedup_idx
            dedup_idx += 1

        # Build GT edge set in dedup space
        gt_edges = set()
        for oi in range(gt_norm.shape[0]):
            for oj in range(oi + 1, gt_norm.shape[1]):
                if gt_norm[oi, oj] > 0:
                    di = origin_to_dedup.get(oi, oi)
                    dj = origin_to_dedup.get(oj, oj)
                    if di != dj:
                        gt_edges.add(frozenset({di, dj}))

        gt_is_tree = (len(gt_edges) == n_dedup - 1)

        trees = nx.algorithms.tree.SpanningTreeIterator(self.G, minimum=False)
        best_rank = None
        best_f1 = -1
        start_time = time.time()

        for rank, tree in enumerate(trees, 1):
            tree_edges = set(frozenset({u, v}) for u, v in tree.edges())

            if gt_is_tree:
                if tree_edges == gt_edges:
                    return (rank, 1.0)
            else:
                if tree_edges.issubset(gt_edges):
                    tp = len(tree_edges)
                    fn = len(gt_edges) - tp
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    return (rank, recall)

            tp = len(tree_edges & gt_edges)
            fp = len(tree_edges - gt_edges)
            fn = len(gt_edges - tree_edges)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_rank = rank

            if time.time() - start_time > time_limit:
                print(f"[get_gt_rank] Time limit {time_limit}s reached at rank {rank}, best_f1={best_f1:.3f}")
                return (f">{rank}", best_f1)

        return (best_rank, best_f1)


    def get_evaluation_metric(self, adj_matrix, selected_ins):
        """
        Determine the connectivity of the subgraph in the semantic map based on the selected form.
        :param adj_matrix: adjacency matrix
        :param selected_ins: form index list
        :return:
        """
        try:
            t_total_start = time.time()
            nx_graph = nx.from_numpy_array(adj_matrix)

            # 环情况
            t0 = time.time()
            circle_num, circle_mean_length = self.find_cycles_networkx(nx_graph)
            print("环总数：", circle_num)
            print(f"环平均长度：{circle_mean_length:.3f}")
            print(f"[eval-timing] find_cycles: {(time.time()-t0)*1000:.2f} ms")

            # connectivity status of the subgraph corresponding to the selected forms
            t0 = time.time()
            connected_flag_list = []
            # names of the forms corresponding to all disconnected subgraphs
            unconnected_forms = []
            # iterate over each form and examine the connectivity of the subgraph
            for form_idx, form in enumerate(self.tfM[selected_ins, :]):
                # construct subgraph based on non-zero features
                nodes = {ix for ix, semantics in enumerate(form) if semantics > 0}
                if len(nodes) == 0:
                    connected_flag_list.append(False)
                    unconnected_forms.append(self.formNames[form_idx])
                    continue
                sub_nx_graph = nx_graph.subgraph(nodes)
                # check the connectivity of the subgraph
                connected_flag = nx.is_connected(sub_nx_graph) if len(nodes) > 1 else True
                connected_flag_list.append(connected_flag)
                if not connected_flag:
                    unconnected_forms.append(self.formNames[form_idx])
            print(f"[eval-timing] connectivity_check: {(time.time()-t0)*1000:.2f} ms")

            # coverage, productivity
            t0 = time.time()
            coverage = sum(connected_flag_list) / len(selected_ins)
            productivity = None
            if len(self.unique_featNames) < 20:
                n_connected = self.count_connected_subgraphs(nx_graph, len(self.unique_featNames))
                productivity = sum(connected_flag_list) / n_connected if n_connected > 0 else 0
            print(f"[eval-timing] productivity: {(time.time()-t0)*1000:.2f} ms")

            # degree
            t0 = time.time()
            deg = [d for _, d in nx_graph.degree()]
            deg_mean = np.mean(deg)
            deg_std = np.std(deg)
            print(f"[eval-timing] degree: {(time.time()-t0)*1000:.2f} ms")

            # acc，p, r, f1
            t0 = time.time()
            evaluation = self.evaluate_against_gt(adj_matrix)
            if evaluation is None:
                evaluation = {}
            print(f"[eval-timing] evaluate_against_gt: {(time.time()-t0)*1000:.2f} ms")

            # weight sum
            t0 = time.time()
            weight_sum = sum(data.get('weight', 1) for u, v, data in nx_graph.edges(data=True))
            print(f"[eval-timing] weight_sum: {(time.time()-t0)*1000:.2f} ms")

            # number of edges
            t0 = time.time()
            num_edges = nx_graph.number_of_edges()
            print(f"[eval-timing] num_edges: {(time.time()-t0)*1000:.2f} ms")

            print(f"[eval-timing] TOTAL: {(time.time()-t_total_start)*1000:.2f} ms")

            return {
                "acc": evaluation.get("acc", None),
                "prec": evaluation.get("precision", None),
                "recall": evaluation.get("recall", None),
                "F1": evaluation.get("f1", None),
                "weight_sum": weight_sum,
                "deg_mean": deg_mean,
                "deg_std": deg_std,
                "productivity": productivity,
                "coverage": coverage,
                "unconnected_forms": unconnected_forms,
                "num_edges": num_edges # 前端是否需要？
            }
        except Exception as e:
            print(e)
            return {}



    def get_all_matrix(self, top_n=5, max_tied_trees=6000):
        """
        return the fully connected adjacency matrix and the adjacency matrices of all candidate trees.
        """
        # generate candidate trees
        start = time.time()
        self.get_optimal_SpanningTrees(top_n=top_n, max_tied_trees=max_tied_trees)
        end = time.time()
        print(f"get_all_matrix time: {end-start}")
        # fully connected adjacency matrix and the adjacency matrices corresponding to candidate trees
        return {
            'origin_matrix': self.adjM,
            'trees': [nx.to_numpy_array(tree) for tree in self.trees],
            'selection_stats': dict(self.tree_selection_stats)
        }



    def check_subgraph_connectivity(self, nx_graph, selected_ins) -> list:
        """
        Check the connectivity of the subgraph in the semantic map based on the selected forms.
        :param nx_graph: The graph to be checked
        :param selected_ins: The index list corresponding to the form that requires connectivity check
        :return: unconnected_forms: Unconnected form list
        """
        # 子图连通标志
        # connected_flag_list = []
        unconnected_forms = []

        # 遍历每个form 查看每个form对应的所有语义节点构成的子图的连接情况
        for form_index in selected_ins:
            form = self.tfM[form_index, :]
            nodes = {ix for ix, semantics in enumerate(form) if semantics > 0}
            if len(nodes) == 0:
                unconnected_forms.append(form_index)
                continue
            sub_nx_graph = nx_graph.subgraph(nodes)
            connected_flag = nx.is_connected(sub_nx_graph) if len(nodes) > 1 else True
            if not connected_flag:
                unconnected_forms.append(form_index)
        return unconnected_forms



    def get_candidate_edges(self, unconnect_form_index_list) -> list:
        """
        Retrieve candidate edges for merging based on all disconnected forms.
        :param unconnect_form_index_list: unconnected form index
        :return: Sorted candidate edge list
        """
        possible_edges = []

        # 逐 form 遍历
        for form_index in unconnect_form_index_list:
            # 不连通的 form
            feat_list = [feat_index for feat_index in range(len(self.unique_featNames)) if
                         self.tfM[form_index, feat_index] > 0]
            # print(feat_list)
            # print(nx.to_numpy_array(self.G))
            if len(feat_list) < 2:
                continue
            # 获取边的权重
            for feat_index in range(len(feat_list)):
                for feat_index_after in range(feat_index + 1, len(feat_list)):
                    node1 = feat_list[feat_index]
                    node2 = feat_list[feat_index_after]
                    edge = self.G.get_edge_data(node1, node2)
                    weight = 0.00001
                    if edge:
                        weight = self.G.get_edge_data(node1, node2)['weight']
                    possible_edges.append((node1, node2, {'weight': weight}))

        # 统计边的次数（只考虑节点对，忽略权重）
        edges_without_attributes = [(u, v) for u, v, _ in possible_edges]
        edge_counts = Counter(edges_without_attributes)

        # 构建一个包含次数和最大权重的结构
        edge_info = {}
        for u, v, attr in possible_edges:
            edge = (u, v)
            weight = attr['weight']
            if edge not in edge_info:
                edge_info[edge] = {"count": edge_counts[edge], "max_weight": weight} # 优先级考虑，先count再weight
            else:
                edge_info[edge]["max_weight"] = max(edge_info[edge]["max_weight"], weight)

        # 排序：按次数和最大权重的累加降序进行排序
        # sorted_edges_by_number = sorted(
        #     edge_info.items(),
        #     key=lambda x: x[1]["count"] + x[1]["max_weight"],
        #     reverse=True,
        # )
        # 排序：优先按照count再根据weight进行降序排序
        sorted_edges_by_number = sorted(
            edge_info.items(),
            key=lambda x: (-x[1]["count"], -x[1]["max_weight"])
        )
        sorted_edges_by_number = [(edge[0], edge[1], {'weight': info['max_weight'], 'count': info['count']}) for
                                  edge, info in sorted_edges_by_number]

        return sorted_edges_by_number



    def merge_edge(self, adj_matrix):

        start = time.time()
        self.connected_graph()

        # 选择所有form
        selected_ins = range(len(self.formNames))

        nx_graph = nx.from_numpy_array(adj_matrix)

        # 原始图的子图连接情况
        unconnect_form_index_list = self.check_subgraph_connectivity(nx_graph, selected_ins)
        sorted_edges_by_number = self.get_candidate_edges(unconnect_form_index_list)

        # 'CO-CD'
        # 添加的边需要高亮出来
        highlight_edges = list()

        # 全连通后跳出循环停止merge
        while unconnect_form_index_list and sorted_edges_by_number:
            # last_unconnect_form_index_list = unconnect_form_index_list

            # 首先是最大count的edge
            confirm_edge = sorted_edges_by_number[0]
            # 可能引起的未连通变化
            max_unconnected_form_change = 0
            new_unconnect_form_index_list = unconnect_form_index_list
            # 第一个未出现在图中的候选边
            is_first_edge = True

            for index, e in enumerate(sorted_edges_by_number):
                # 构建临时图 深拷贝
                temporary_graph = copy.deepcopy(nx_graph)

                # 在图中 跳过
                if (e[0], e[1]) in temporary_graph.edges or (e[1], e[0]) in temporary_graph.edges:
                    continue
                # 不在图中
                else:
                    if is_first_edge:
                        confirm_edge = e
                        is_first_edge = False
                    # 给临时graph添加候选边
                    temporary_graph.add_edge(e[0], e[1], **e[2])
                    # 查看添加候选边后未连通子图的变化情况
                    temporary_unconnect_form_index_list = self.check_subgraph_connectivity(temporary_graph, unconnect_form_index_list)
                    # 变化的数量
                    unconnected_form_change = len(unconnect_form_index_list) - len(temporary_unconnect_form_index_list)
                    # 更新本轮的最大变化 以及 带来变化的候选边 以及变化后的未连通子图情况
                    # 必须是大于，不能是大于等于
                    if unconnected_form_change > max_unconnected_form_change:
                        max_unconnected_form_change = unconnected_form_change
                        confirm_edge = e
                        new_unconnect_form_index_list = temporary_unconnect_form_index_list
                    # temporary_graph.remove_edge(e[0], e[1])

            # 添加边
            nx_graph.add_edge(confirm_edge[0], confirm_edge[1], **confirm_edge[2])
            highlight_edges.append((confirm_edge[0], confirm_edge[1], confirm_edge[2].get('weight')))
            # 更新未连通情况：merge当前边后的未连通情况
            unconnect_form_index_list = new_unconnect_form_index_list
            # 更新候选边：从候选列表中剔除已经被merge的边
            sorted_edges_by_number = self.get_candidate_edges(unconnect_form_index_list)

            ## 有可能添加了某个边后子图连通性未发生任何变化！！！！！！


        nx_graph_merged_matrix = nx.to_numpy_array(nx_graph)
        print(highlight_edges)

        end = time.time()
        print(f"merge_edge time: {end-start}")

        return nx_graph_merged_matrix, highlight_edges







