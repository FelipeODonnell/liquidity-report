"""
Utility functions for enhancing chart display in the Liquidity Report.
"""

import pandas as pd
import logging
import plotly.graph_objects as go
from components.charts import create_pie_chart

# Configure logging
logger = logging.getLogger(__name__)


def create_treemap_chart(df, values_col, names_col, parent_col=None, title=None, height=500, width=None, color_map=None):
    """
    Create a treemap chart as an alternative to pie charts for hierarchical data visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    values_col : str
        Column containing values
    names_col : str
        Column containing names/labels
    parent_col : str, optional
        Column containing parent values for hierarchical treemap
    title : str, optional
        Chart title
    height : int, optional
        Chart height in pixels
    width : int, optional
        Chart width in pixels
    color_map : dict, optional
        Mapping of names to colors
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The created treemap figure
    """
    try:
        # Make a copy to avoid modifying the original dataframe
        plot_df = df.copy()
        
        # Ensure the values column is numeric
        plot_df[values_col] = pd.to_numeric(plot_df[values_col], errors='coerce')
        
        # Remove rows with zero or NaN values
        plot_df = plot_df[plot_df[values_col] > 0].dropna(subset=[values_col])
        
        if plot_df.empty:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No data available after filtering",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            layout_params = {'height': height}
            if title:
                layout_params['title'] = title
            if width:
                layout_params['width'] = width
            fig.update_layout(**layout_params)
            return fig
        
        # Create labels, parents, and values lists
        labels = plot_df[names_col].tolist()
        values = plot_df[values_col].tolist()
        
        # Set parents based on parent_col if provided, otherwise use empty string for all
        parents = plot_df[parent_col].tolist() if parent_col in plot_df.columns else [''] * len(labels)
        
        # Create treemap figure
        fig = go.Figure(go.Treemap(
            labels=labels,
            values=values,
            parents=parents,
            texttemplate="%{label}<br>%{percentRoot:.1f}%",
            hovertemplate='<b>%{label}</b><br>Value: %{value:,.0f}<br>Percentage: %{percentRoot:.1f}%<extra></extra>',
            marker=dict(
                # Set colors if color map is provided
                colors=[color_map.get(name, None) for name in labels] if color_map else None,
                line=dict(width=1, color='white')
            ),
            textfont=dict(size=12),
            root_color="white",  # Set root node color to white
            branchvalues="total"  # Show values as percentage of total
        ))
        
        # Set layout
        layout_params = {
            'margin': dict(t=50, l=0, r=0, b=0),
            'height': height,
        }
        
        # Add title if provided
        if title:
            layout_params['title'] = title
            
        # Add width if provided
        if width:
            layout_params['width'] = width
            
        fig.update_layout(**layout_params)
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating treemap chart: {e}")
        # Return basic empty figure as fallback
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating treemap: {str(e)}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig

def create_enhanced_pie_chart(df, values_col, names_col, title, color_map=None, **kwargs):
    """
    Create a properly formatted bar chart (replacing pie chart) with correct data handling and no text overlaps.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    values_col : str
        Column containing values
    names_col : str
        Column containing names/labels
    title : str
        Chart title
    color_map : dict, optional
        Mapping of names to colors
    **kwargs : additional arguments
        Additional arguments to pass to create_pie_chart
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The created figure
    """
    try:
        # Make a defensive copy of the data
        working_df = df.copy()
        
        # Ensure the values column is numeric
        working_df[values_col] = pd.to_numeric(working_df[values_col], errors='coerce')
        
        # Drop rows with zero or NaN values
        working_df = working_df.dropna(subset=[values_col])
        working_df = working_df[working_df[values_col] > 0]
        
        # Remove any "All" items which shouldn't be in the chart
        working_df = working_df[working_df[names_col] != "All"]
        
        # Additional exclude names from kwargs if provided
        exclude_names = kwargs.pop('exclude_names', ["All"])
        if isinstance(exclude_names, list):
            for name in exclude_names:
                working_df = working_df[working_df[names_col] != name]
        
        # Determine if we need outside labels based on number of data points
        # This helps prevent text overlapping for charts with many slices
        num_items = len(working_df)
        if 'show_outside_labels' not in kwargs:
            # Automatically use outside labels if there are more than 5 slices
            kwargs['show_outside_labels'] = num_items > 5
        
        # Display legends for pies with many slices or if explicitly requested
        if 'show_legend' not in kwargs:
            kwargs['show_legend'] = num_items > 7 or kwargs.get('show_outside_labels', False)
            
        # For smaller pie charts, show labels inline
        if 'show_outside_labels' in kwargs and not kwargs['show_outside_labels']:
            if num_items <= 5 and 'display_percentages' not in kwargs:
                kwargs['display_percentages'] = True
                
        # Use short labels for charts with many items to prevent overlap
        if 'use_short_labels' not in kwargs:
            kwargs['use_short_labels'] = num_items > 7
        
        # Set additional parameters with sensible defaults if not provided
        params = {
            'show_top_n': 8,  # Show top 8 items by default
            'min_percent': 2.0,  # Group items with less than 2% share by default
            'height': 400,
            'pull_out_top': False,  # Do not pull out largest slice
            'display_percentages': True,
            'display_labels': True,
            'show_outside_labels': num_items > 5,  # Use outside labels for more than 5 items
            'use_short_labels': num_items > 7  # Use short labels for more than 7 items
        }
        
        # Update params with any provided kwargs
        params.update(kwargs)
        
        # Check if too many small items and adjust min_percent if needed
        if 'min_percent' in params and params['min_percent'] < 1.0:
            small_items_count = len(working_df[working_df[values_col] / working_df[values_col].sum() < 0.01])
            if small_items_count > 5:  # Too many tiny slices
                params['min_percent'] = max(params['min_percent'], 1.0)
        
        # Create the pie chart with our enhanced function and proper error handling
        try:
            fig = create_pie_chart(
                df=working_df,
                values_col=values_col,
                names_col=names_col,
                title=title,
                color_map=color_map,
                exclude_names=exclude_names,
                **params
            )
            return fig
        except Exception as e:
            logger.error(f"Error in create_pie_chart: {e}")
            # Create a simpler fallback chart without problematic options
            return create_simple_fallback_pie(
                df=working_df,
                values_col=values_col,
                names_col=names_col,
                title=title,
                color_map=color_map
            )
    
    except Exception as e:
        logger.error(f"Error creating enhanced pie chart: {e}")
        # Return a basic pie chart as fallback with outside labels to prevent overlapping
        try:
            return create_simple_fallback_pie(
                df=df,
                values_col=values_col,
                names_col=names_col,
                title=title,
                color_map=color_map
            )
        except Exception as e2:
            logger.error(f"Fallback pie chart also failed: {e2}")
            # Return an empty figure as last resort
            fig = go.Figure()
            fig.add_annotation(
                text="Error creating chart: Unable to process data",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="red")
            )
            fig.update_layout(title=title)
            return fig


def create_simple_fallback_pie(df, values_col, names_col, title, color_map=None, height=400):
    """
    Create a very simple bar chart (replacing pie chart) as a fallback.
    
    This bare-bones implementation provides a simple horizontal bar visualization
    that displays the same data that would have been in a pie chart.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    values_col : str
        Column containing values
    names_col : str
        Column containing names/labels
    title : str
        Chart title
    color_map : dict, optional
        Mapping of names to colors
    height : int
        Chart height in pixels
        
    Returns:
    --------
    plotly.graph_objects.Figure
        A simple bar chart or error figure
    """
    # Create a copy of the dataframe to work with
    df_copy = df.copy()
    
    # Clean data - ensure numeric values and filter nulls/zeros
    df_copy[values_col] = pd.to_numeric(df_copy[values_col], errors='coerce')
    df_copy = df_copy[df_copy[values_col] > 0].dropna(subset=[values_col])
    
    # Remove 'All' category if present
    df_copy = df_copy[~df_copy[names_col].isin(['All', 'all', 'ALL'])]
    
    # If no valid data left, return empty chart
    if df_copy.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid data for chart",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False
        )
        fig.update_layout(title=title, height=height)
        return fig
    
    # Calculate percentages
    total = df_copy[values_col].sum()
    df_copy['percent'] = df_copy[values_col] / total * 100
    df_copy['text'] = df_copy['percent'].apply(lambda x: f'{x:.1f}%')
    
    # Sort by value for better visualization (ascending for horizontal bars - largest at top)
    df_copy = df_copy.sort_values(by=values_col, ascending=True)
    
    # Create a horizontal bar chart as replacement for pie chart
    fig = go.Figure(data=[go.Bar(
        y=df_copy[names_col],
        x=df_copy[values_col],
        orientation='h',
        text=df_copy['text'],
        textposition='auto',
        textfont=dict(size=12, color='white'),
        hovertemplate='%{y}<br>Value: %{x:,.0f}<br>Percentage: %{text}<extra></extra>',
        marker=dict(
            line=dict(color='white', width=1),
            color=[color_map.get(name, None) for name in df_copy[names_col]] if color_map else None
        )
    )])
    
    # Improved layout for bar chart
    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,  # Position title higher
            'font': {'color': 'white'}
        },
        height=height,
        margin=dict(t=70, b=50, l=120, r=40),  # Increase left margin for category labels
        yaxis=dict(
            title=None,
            automargin=True  # Auto-adjust margin to fit labels
        ),
        xaxis=dict(
            title='Value',
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)'
        )
    )
    
    return fig