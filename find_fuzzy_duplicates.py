import polars as pl
from thefuzz import fuzz
from typing import List, Optional
import networkx as nx
from tqdm.auto import tqdm
import time # Used for a more responsive print

def find_fuzzy_duplicates(
    df: pl.DataFrame,
    columns: List[str],
    similarity_threshold: int = 90,
    columns_to_view: Optional[List[str]] = None,
    blocking_column: Optional[str] = None,
    show_progress: bool = True
) -> pl.DataFrame:
    """
    Finds and groups fuzzy duplicates using a reliable `thefuzz` implementation
    combined with a "blocking" strategy for performance.

    This function is designed for correctness and will not raise an AttributeError.

    Args:
        df: The input Polars DataFrame.
        columns: A list of column names to use for fuzzy matching.
        similarity_threshold: The average score (0-100) required to consider
                              two records a match.
        columns_to_view: Optional list of additional columns to include in the
                         output. If None, all original columns are returned.
        blocking_column: A column for an initial exact match to reduce
                         comparisons (e.g., "POSTNR"). If None, a full
                         cross-join is used (slower).
        show_progress: If True, displays progress indicators for long-running steps.

    Returns:
        A new Polars DataFrame containing only the duplicate rows, sorted by
        group, with a new 'fuzzy_group' integer column at the start.
    """
    # --- Step 0: Schema setup for returning empty DataFrames ---
    def get_empty_df_with_schema() -> pl.DataFrame:
        base_df = df.clear().with_columns(pl.lit(None, dtype=pl.Int64).alias("fuzzy_group"))
        if columns_to_view is None:
            output_cols = ["fuzzy_group"] + df.columns
        else:
            output_cols = ["fuzzy_group"] + columns + [c for c in columns_to_view if c not in columns]
        
        valid_cols = [c for c in output_cols if c in base_df.columns or c == "fuzzy_group"]
        return base_df.select(valid_cols)

    if df.height < 2:
        return get_empty_df_with_schema()

    df_with_id = df.with_row_count("unique_id")

    # --- Step 1: The "Blocking" Join (Performance Tuning) ---
    if show_progress:
        join_type = "inner join on blocking column" if blocking_column else "cross join"
        print(f"Step 1/3: Creating candidate pairs via {join_type}... (this can be very slow)")
        # Flush the output to make sure the message appears immediately
        time.sleep(0.1) 

    if blocking_column and blocking_column in df.columns:
        pairs_df = df_with_id.join(
            df_with_id, on=blocking_column, how="inner", suffix="_right"
        )
    else:
        pairs_df = df_with_id.join(df_with_id, how="cross", suffix="_right")

    pairs_df = pairs_df.filter(pl.col("unique_id") < pl.col("unique_id_right"))
    
    if show_progress:
        print(f" -> Found {pairs_df.height:,} pairs to compare.")

    if pairs_df.height == 0:
        return get_empty_df_with_schema()

    # --- Step 2: Reliable Fuzzy Scoring with Progress Bar ---
    comparison_cols = columns + [f"{c}_right" for c in columns]
    rows_to_compare = pairs_df.select(comparison_cols).to_dicts()
    
    scores = []
    for row in tqdm(rows_to_compare, desc="Step 2/3: Calculating Similarities", disable=not show_progress, leave=False):
        row_scores = []
        for col in columns:
            str1 = row.get(col)
            str2 = row.get(f"{col}_right")
            if str1 is None or str2 is None:
                row_scores.append(0)
                continue
            row_scores.append(fuzz.token_set_ratio(str1, str2))
        
        average_score = sum(row_scores) / len(row_scores) if row_scores else 0.0
        scores.append(average_score)

    similarity_series = pl.Series("similarity", scores, dtype=pl.Float64)
    duplicate_pairs = pairs_df.with_columns(similarity_series).filter(
        pl.col("similarity") >= similarity_threshold
    )

    # --- Step 3: Graph Grouping & Final Assignment ---
    if duplicate_pairs.height == 0:
        return get_empty_df_with_schema()

    if show_progress:
        print(f"Step 3/3: Found {duplicate_pairs.height:,} duplicate pairs. Building graph and assigning group IDs...")

    edges = duplicate_pairs.select(["unique_id", "unique_id_right"]).rows()
    G = nx.Graph(edges)
    G.add_nodes_from(df_with_id["unique_id"])
    components = nx.connected_components(G)
    
    component_list = list(components)
    
    group_mapping = []
    group_id_counter = 0
    for group in tqdm(component_list, desc=" -> Processing groups", disable=not show_progress):

        if len(group) > 1:
            for node in group:
                group_mapping.append({"unique_id": node, "group_id": group_id_counter})
            group_id_counter += 1

    if not group_mapping:
        return get_empty_df_with_schema()

    group_mapping_df = pl.DataFrame(group_mapping)
    df_with_groups = df_with_id.join(group_mapping_df, on="unique_id", how="inner")
    sorted_df = df_with_groups.sort(["group_id", "unique_id"])

    # --- Final Output Selection ---
    final_df_with_group = sorted_df.with_columns(pl.col("group_id").alias("fuzzy_group"))

    if columns_to_view is None:
        output_columns = ["fuzzy_group"] + df.columns
    else:
        output_columns = ["fuzzy_group"] + columns
        for col in columns_to_view:
            if col not in output_columns:
                output_columns.append(col)

    valid_output_columns = [c for c in output_columns if c in final_df_with_group.columns]
    return final_df_with_group.select(valid_output_columns)
