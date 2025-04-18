"""
Reusable Plotly functions for visualizing embeddings and clusters.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Plotly Availability Check ---
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
    logger.info("Plotly library loaded successfully.")
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly library not found. Visualization functions will not be available.")
    px = None
    go = None

def plot_embeddings_interactive(
    embeddings_reduced: np.ndarray,
    metadata_df: pd.DataFrame,
    plot_title: str,
    color_by_key: Optional[str] = None,
    symbol_by_key: Optional[str] = None,
    hover_data_keys: Optional[List[str]] = None,
    dimensions: int = 2,
    filter_outliers: bool = False,
    outlier_quantile: float = 0.01,
    show_figure: bool = True,
    save_path: Optional[str] = None
) -> Optional[go.Figure]:
    """
    Creates an interactive scatter plot (2D or 3D) of reduced embeddings using Plotly.

    Args:
        embeddings_reduced: Numpy array of reduced embeddings (shape: [n_samples, dimensions]).
        metadata_df: DataFrame containing metadata aligned with embeddings_reduced.
                     Must have the same number of rows.
        plot_title: The title for the plot.
        color_by_key: Column name in metadata_df to use for coloring points.
        symbol_by_key: Column name in metadata_df to use for different symbols.
        hover_data_keys: List of column names from metadata_df to show on hover.
                         If None, uses a default set or all available.
        dimensions: The number of dimensions to plot (2 or 3).
        filter_outliers: Whether to filter outliers based on quantiles before plotting.
        outlier_quantile: The quantile to use for outlier filtering (e.g., 0.01 means filter bottom/top 1%).
        show_figure: Whether to display the figure using fig.show().
        save_path: Optional path to save the plot as an HTML file.

    Returns:
        The Plotly Figure object, or None if Plotly is not available or an error occurs.
    """
    if not PLOTLY_AVAILABLE:
        logger.error("Plotly is not installed. Cannot create visualization.")
        return None

    if embeddings_reduced is None or metadata_df is None:
        logger.error("Reduced embeddings or metadata DataFrame is missing.")
        return None

    if len(embeddings_reduced) != len(metadata_df):
        logger.error("Mismatch between number of embeddings and metadata rows.")
        return None

    if dimensions not in [2, 3]:
        logger.error("Plot dimensions must be 2 or 3.")
        return None

    if embeddings_reduced.shape[1] < dimensions:
         logger.error(f"Reduced embeddings have only {embeddings_reduced.shape[1]} dimensions, cannot plot in {dimensions}D.")
         return None

    logger.info(f"Creating {dimensions}D Plotly visualization: '{plot_title}'")

    # --- Prepare Data for Plotting ---
    plot_df = metadata_df.copy()

    # Add embedding dimensions to the DataFrame
    axis_labels = ['X', 'Y', 'Z']
    for i in range(dimensions):
        plot_df[axis_labels[i]] = embeddings_reduced[:, i]

    # --- Filtering Outliers (Optional) ---
    if filter_outliers:
        logger.info(f"Filtering outliers using quantile: {outlier_quantile}")
        original_count = len(plot_df)
        for dim in axis_labels[:dimensions]:
            lower_bound = plot_df[dim].quantile(outlier_quantile)
            upper_bound = plot_df[dim].quantile(1 - outlier_quantile)
            plot_df = plot_df[(plot_df[dim] >= lower_bound) & (plot_df[dim] <= upper_bound)]
        filtered_count = len(plot_df)
        logger.info(f"Filtered {original_count - filtered_count} outliers ({ (original_count - filtered_count) / original_count:.1%}). {filtered_count} points remaining.")
        if filtered_count == 0:
            logger.warning("All points were filtered as outliers. Cannot create plot.")
            return None

    # --- Configure Plot Arguments ---
    plot_args = {
        'data_frame': plot_df,
        'title': plot_title,
        'color': color_by_key if color_by_key in plot_df.columns else None,
        'symbol': symbol_by_key if symbol_by_key in plot_df.columns else None,
    }

    # Prepare hover data with truncated text
    if hover_data_keys:
        valid_hover_keys = [k for k in hover_data_keys if k in plot_df.columns]
        if len(valid_hover_keys) != len(hover_data_keys):
            logger.warning(f"Some requested hover data keys not found in metadata: {set(hover_data_keys) - set(valid_hover_keys)}")
        
        # Create truncated columns for hover data
        for key in valid_hover_keys:
            if pd.api.types.is_string_dtype(plot_df[key]):
                hover_key = f"{key}_hover"
                plot_df[hover_key] = plot_df[key].astype(str).apply(lambda x: f"{x[:97]}..." if len(x) > 100 else x)
                valid_hover_keys[valid_hover_keys.index(key)] = hover_key
        
        plot_args['hover_data'] = valid_hover_keys
    else:
        # Default hover data with truncation
        default_hover = [col for col in plot_df.columns if col not in axis_labels[:dimensions]]
        # Create truncated columns for hover data
        for key in default_hover:
            if pd.api.types.is_string_dtype(plot_df[key]):
                hover_key = f"{key}_hover"
                plot_df[hover_key] = plot_df[key].astype(str).apply(lambda x: f"{x[:97]}..." if len(x) > 100 else x)
                default_hover[default_hover.index(key)] = hover_key
        
        plot_args['hover_data'] = default_hover
        logger.info(f"Using default hover data (truncated): {default_hover}")

    # Handle potential categorical coloring issues (convert to string)
    if plot_args['color'] and not pd.api.types.is_numeric_dtype(plot_df[plot_args['color']]):
         plot_df[plot_args['color']] = plot_df[plot_args['color']].astype(str)
         logger.debug(f"Converted color column '{plot_args['color']}' to string for discrete coloring.")

    # Handle potential categorical symbol issues
    if plot_args['symbol'] and not pd.api.types.is_numeric_dtype(plot_df[plot_args['symbol']]):
         plot_df[plot_args['symbol']] = plot_df[plot_args['symbol']].astype(str)
         logger.debug(f"Converted symbol column '{plot_args['symbol']}' to string.")


    # --- Create Plot (2D or 3D) ---
    fig = None
    try:
        if dimensions == 2:
            plot_args['x'] = 'X'
            plot_args['y'] = 'Y'
            fig = px.scatter(**plot_args)
            fig.update_layout(xaxis_title="UMAP Dimension 1", yaxis_title="UMAP Dimension 2")
        elif dimensions == 3:
            plot_args['x'] = 'X'
            plot_args['y'] = 'Y'
            plot_args['z'] = 'Z'
            fig = px.scatter_3d(**plot_args)
            fig.update_layout(scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3"))

        # --- Customize Layout ---
        if fig:
            fig.update_layout(
                title_x=0.5, # Center title
                legend_title_text=color_by_key if color_by_key else 'Color',
                margin=dict(l=0, r=0, t=40, b=0)
            )
            fig.update_traces(marker=dict(size=3, opacity=0.7)) # Adjust marker style

    except Exception as e:
        logger.error(f"Error creating Plotly figure: {e}", exc_info=True)
        return None

    # --- Show or Save Figure ---
    if fig:
        if save_path:
            try:
                logger.info(f"Saving plot to: {save_path}")
                fig.write_html(save_path)
            except Exception as e:
                logger.error(f"Failed to save plot to {save_path}: {e}", exc_info=True)

        if show_figure:
            logger.info("Displaying plot...")
            fig.show()

    return fig


# Example Usage (for testing)
# if __name__ == "__main__":
#     print("--- Testing Visualization --- ")
#     if not PLOTLY_AVAILABLE:
#         print("Plotly not available, skipping test.")
#     else:
#         # Create dummy reduced embeddings and metadata
#         np.random.seed(42)
#         n_points = 200
#         # 2D example
#         dummy_embeddings_2d = np.random.rand(n_points, 2) * 10
#         # 3D example
#         dummy_embeddings_3d = np.random.rand(n_points, 3) * 10

#         # Dummy Metadata
#         dummy_metadata = pd.DataFrame({
#             'id': [f'ID_{i:03d}' for i in range(n_points)],
#             'cluster_label': np.random.choice(['A', 'B', 'C', -1], n_points, p=[0.3, 0.3, 0.3, 0.1]).astype(str),
#             'document_length': np.random.randint(50, 500, n_points),
#             'category': np.random.choice(['Support', 'Sales', 'Billing'], n_points),
#             'sentiment_score': np.random.rand(n_points)
#         })

#         hover_cols = ['id', 'document_length', 'category']

#         print("\nTesting 2D Plot...")
#         fig2d = plot_embeddings_interactive(
#             embeddings_reduced=dummy_embeddings_2d,
#             metadata_df=dummy_metadata,
#             plot_title="Dummy 2D Embeddings by Cluster",
#             color_by_key='cluster_label',
#             symbol_by_key='category',
#             hover_data_keys=hover_cols,
#             dimensions=2,
#             show_figure=True,
#             save_path="dummy_plot_2d.html"
#         )
#         if fig2d:
#             print("2D Plot generated.")
#         else:
#              print("2D Plot FAILED.")

#         print("\nTesting 3D Plot...")
#         fig3d = plot_embeddings_interactive(
#             embeddings_reduced=dummy_embeddings_3d,
#             metadata_df=dummy_metadata,
#             plot_title="Dummy 3D Embeddings by Length",
#             color_by_key='document_length',
#             hover_data_keys=hover_cols,
#             dimensions=3,
#             show_figure=True,
#             save_path="dummy_plot_3d.html"
#         )
#         if fig3d:
#             print("3D Plot generated.")
#         else:
#             print("3D Plot FAILED.") 