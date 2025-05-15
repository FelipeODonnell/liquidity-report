"""
Chart creation utilities for the Izun Crypto Liquidity Report application.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import ASSET_COLORS, CHART_COLORS, EXCHANGE_COLORS

def apply_chart_theme(fig, template="plotly_dark"):
    """
    Apply a consistent theme to a Plotly figure.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The figure to apply the theme to
    template : str
        The Plotly template to use
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The themed figure
    """
    fig.update_layout(
        template=template,
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes
    fig.update_xaxes(
        gridcolor="rgba(128, 128, 128, 0.1)",
        zeroline=False
    )
    fig.update_yaxes(
        gridcolor="rgba(128, 128, 128, 0.1)",
        zeroline=False
    )
    
    return fig

def create_time_series(df, x_col, y_col, title, color_col=None, color_discrete_map=None, height=400, show_legend=True):
    """
    Create a time series line chart.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    x_col : str
        Column to use for the x-axis (usually datetime)
    y_col : str or list
        Column(s) to use for the y-axis
    title : str
        Chart title
    color_col : str, optional
        Column to use for color differentiation
    color_discrete_map : dict, optional
        Mapping of color values to colors
    height : int
        Chart height in pixels
    show_legend : bool
        Whether to show the legend
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The created figure
    """
    if df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title=title, height=height)
        return apply_chart_theme(fig)
    
    # Handle multiple y columns
    if isinstance(y_col, list):
        # Create figure with multiple traces
        fig = go.Figure()
        
        for col in y_col:
            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[col],
                    mode="lines",
                    name=col
                )
            )
    else:
        # Create figure with plotly express
        if color_col:
            fig = px.line(
                df,
                x=x_col,
                y=y_col,
                color=color_col,
                color_discrete_map=color_discrete_map or EXCHANGE_COLORS,
                title=title,
                height=height
            )
        else:
            fig = px.line(
                df,
                x=x_col,
                y=y_col,
                title=title,
                height=height
            )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title=y_col if isinstance(y_col, str) else None,
        hovermode="x unified",
        showlegend=show_legend
    )
    
    return apply_chart_theme(fig)

def create_ohlc_chart(df, datetime_col, open_col, high_col, low_col, close_col, title, volume_col=None, height=500):
    """
    Create an OHLC candlestick chart with optional volume.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    datetime_col : str
        Column containing datetime values
    open_col, high_col, low_col, close_col : str
        Columns containing OHLC data
    title : str
        Chart title
    volume_col : str, optional
        Column containing volume data
    height : int
        Chart height in pixels
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The created figure
    """
    if df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title=title, height=height)
        return apply_chart_theme(fig)
    
    # Create figure with or without volume
    if volume_col and volume_col in df.columns:
        # Create figure with secondary y-axis for volume
        fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03, 
            row_heights=[0.7, 0.3]
        )
        
        # Add candlestick trace
        fig.add_trace(
            go.Candlestick(
                x=df[datetime_col],
                open=df[open_col],
                high=df[high_col],
                low=df[low_col],
                close=df[close_col],
                name="OHLC"
            ),
            row=1, 
            col=1
        )
        
        # Add volume trace
        fig.add_trace(
            go.Bar(
                x=df[datetime_col],
                y=df[volume_col],
                name="Volume",
                marker_color="rgba(128, 128, 128, 0.5)"
            ),
            row=2, 
            col=1
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=False
        )
    else:
        # Create simple candlestick chart
        fig = go.Figure(data=[
            go.Candlestick(
                x=df[datetime_col],
                open=df[open_col],
                high=df[high_col],
                low=df[low_col],
                close=df[close_col],
                name="OHLC"
            )
        ])
        
        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            xaxis_rangeslider_visible=False
        )
    
    return apply_chart_theme(fig)

def create_bar_chart(df, x_col, y_col, title, color_col=None, color_discrete_map=None, height=400, orientation='v', barmode='group'):
    """
    Create a bar chart.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    x_col : str
        Column to use for the x-axis
    y_col : str or list
        Column(s) to use for the y-axis
    title : str
        Chart title
    color_col : str, optional
        Column to use for color differentiation
    color_discrete_map : dict, optional
        Mapping of color values to colors
    height : int
        Chart height in pixels
    orientation : str
        Chart orientation ('v' for vertical, 'h' for horizontal)
    barmode : str
        Bar mode ('group', 'stack', 'relative', 'overlay')
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The created figure
    """
    if df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title=title, height=height)
        return apply_chart_theme(fig)
    
    # Handle multiple y columns
    if isinstance(y_col, list):
        # Create figure with multiple traces
        fig = go.Figure()
        
        for col in y_col:
            if orientation == 'v':
                fig.add_trace(
                    go.Bar(
                        x=df[x_col],
                        y=df[col],
                        name=col
                    )
                )
            else:
                fig.add_trace(
                    go.Bar(
                        y=df[x_col],
                        x=df[col],
                        name=col,
                        orientation='h'
                    )
                )
        
        # Update layout for barmode
        fig.update_layout(barmode=barmode)
    else:
        # Create figure with plotly express
        if orientation == 'v':
            fig = px.bar(
                df,
                x=x_col,
                y=y_col,
                color=color_col,
                color_discrete_map=color_discrete_map or EXCHANGE_COLORS,
                title=title,
                height=height,
                barmode=barmode
            )
        else:
            fig = px.bar(
                df,
                y=x_col,
                x=y_col,
                color=color_col,
                color_discrete_map=color_discrete_map or EXCHANGE_COLORS,
                title=title,
                height=height,
                orientation='h',
                barmode=barmode
            )
    
    # Update layout
    if orientation == 'v':
        fig.update_layout(
            title=title,
            xaxis_title=None,
            yaxis_title=y_col if isinstance(y_col, str) else None
        )
    else:
        fig.update_layout(
            title=title,
            yaxis_title=None,
            xaxis_title=y_col if isinstance(y_col, str) else None
        )
    
    return apply_chart_theme(fig)

def create_time_series_with_bar(df, x_col, line_y_col, bar_y_col, title, height=500):
    """
    Create a combination chart with line and bar.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    x_col : str
        Column to use for the x-axis
    line_y_col : str
        Column to use for the line chart
    bar_y_col : str
        Column to use for the bar chart
    title : str
        Chart title
    height : int
        Chart height in pixels
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The created figure
    """
    if df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title=title, height=height)
        return apply_chart_theme(fig)
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar chart
    fig.add_trace(
        go.Bar(
            x=df[x_col], 
            y=df[bar_y_col], 
            name=bar_y_col if isinstance(bar_y_col, str) else "Bar",
            marker_color=CHART_COLORS['primary']
        ),
        secondary_y=False,
    )
    
    # Add line chart
    fig.add_trace(
        go.Scatter(
            x=df[x_col], 
            y=df[line_y_col], 
            name=line_y_col if isinstance(line_y_col, str) else "Line", 
            mode="lines",
            line=dict(color=CHART_COLORS['secondary'])
        ),
        secondary_y=True,
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        hovermode="x unified",
        height=height
    )
    
    # Update axes titles
    fig.update_yaxes(title_text=bar_y_col if isinstance(bar_y_col, str) else "Bar", secondary_y=False)
    fig.update_yaxes(title_text=line_y_col if isinstance(line_y_col, str) else "Line", secondary_y=True)
    
    return apply_chart_theme(fig)

def create_pie_chart(df, values_col, names_col, title, height=400, color_map=None, exclude_names=None, 
                 show_top_n=None, min_percent=1.0, pull_out_top=True, 
                 hover_info='label+percent+value', show_legend=True, show_outside_labels=False,
                 display_percentages=True, display_labels=True, use_short_labels=False, width=None):
    """
    Create a pie chart with improved formatting and filtering options.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    values_col : str
        Column containing values
    names_col : str
        Column containing names
    title : str
        Chart title
    height : int
        Chart height in pixels
    color_map : dict, optional
        Mapping of names to colors
    exclude_names : list, optional
        List of names to exclude from the chart (e.g., "All")
    show_top_n : int, optional
        Only show the top N items by value and group the rest as "Others"
    min_percent : float, optional
        Minimum percentage to include a slice (items below will be grouped as "Others")
    pull_out_top : bool, optional
        Whether to pull out the largest slice for emphasis
    hover_info : str
        Information to display in hover tooltip
    show_legend : bool
        Whether to show the legend
    show_outside_labels : bool
        Whether to place labels outside the pie (prevents overlapping)
    display_percentages : bool
        Whether to display percentage values on labels
    display_labels : bool
        Whether to display text labels for slices
    use_short_labels : bool
        Whether to truncate long labels
    width : int, optional
        Chart width in pixels
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The created figure
    """
    if df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title=title, height=height)
        return apply_chart_theme(fig)
    
    # Make a copy to avoid modifying the original dataframe
    plot_df = df.copy()
    
    # Exclude specific names (like "All")
    if exclude_names:
        plot_df = plot_df[~plot_df[names_col].isin(exclude_names)]
    
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
        fig.update_layout(title=title, height=height)
        return apply_chart_theme(fig)
    
    # Calculate percentages for grouping small slices
    total_value = plot_df[values_col].sum()
    plot_df['percent'] = (plot_df[values_col] / total_value) * 100
    
    # Group small slices as "Others" based on minimum percentage
    if min_percent > 0:
        small_items = plot_df[plot_df['percent'] < min_percent]
        if not small_items.empty:
            # Create an "Others" row
            others_row = pd.DataFrame({
                names_col: ['Others'],
                values_col: [small_items[values_col].sum()],
                'percent': [small_items['percent'].sum()]
            })
            # Remove small items and add Others
            plot_df = plot_df[plot_df['percent'] >= min_percent]
            plot_df = pd.concat([plot_df, others_row], ignore_index=True)
    
    # Show only top N items if specified
    if show_top_n and len(plot_df) > show_top_n:
        # Sort by value
        plot_df = plot_df.sort_values(values_col, ascending=False)
        # Take top N-1 items and group the rest as "Others"
        top_items = plot_df.iloc[:show_top_n-1]
        bottom_items = plot_df.iloc[show_top_n-1:]
        
        # Check if bottom items exist before creating "Others" row
        if not bottom_items.empty:
            # Create an "Others" row
            others_row = pd.DataFrame({
                names_col: ['Others'],
                values_col: [bottom_items[values_col].sum()],
                'percent': [bottom_items['percent'].sum()]
            })
            
            # Combine top items with Others
            plot_df = pd.concat([top_items, others_row], ignore_index=True)
        else:
            plot_df = top_items
    
    # Sort by value for consistent ordering
    plot_df = plot_df.sort_values(values_col, ascending=False)
    
    # Create a list of pulls for the largest slice if requested
    pulls = None
    if pull_out_top and len(plot_df) > 1:
        pulls = [0.1] + [0] * (len(plot_df) - 1)
    
    # Format labels, possibly truncating long ones
    labels = plot_df[names_col].tolist()
    if use_short_labels:
        labels = [label[:15] + '...' if isinstance(label, str) and len(label) > 15 else label for label in labels]
    
    # Determine textinfo based on parameters
    if display_percentages and display_labels:
        text_info = 'percent+label'
    elif display_percentages:
        text_info = 'percent'
    elif display_labels:
        text_info = 'label'
    else:
        text_info = 'none'
    
    # Determine text position based on whether to show outside labels
    text_position = 'outside' if show_outside_labels else 'inside'
    
    # Use auto for inside text orientation when using outside labels
    inside_text_orientation = 'auto' if show_outside_labels else 'radial'
    
    # Create custom text template based on display preferences
    if display_percentages and display_labels:
        if show_outside_labels:
            # For outside labels, more compact format
            text_template = '%{label}: %{percent:.1f}%'
        else:
            # For inside labels, potentially multi-line
            text_template = '%{label}<br>%{percent:.1f}%'
    elif display_percentages:
        text_template = '%{percent:.1f}%'
    elif display_labels:
        text_template = '%{label}'
    else:
        text_template = ''
    
    # Create the pie chart with improved layout
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=plot_df[values_col],
        textinfo=text_info,
        hoverinfo=hover_info,
        hovertemplate='%{label}<br>%{percent:.1f}%<br>Value: %{value:,.0f}<extra></extra>',
        texttemplate=text_template,
        textposition=text_position,
        insidetextorientation=inside_text_orientation,
        pull=pulls,
        rotation=90,  # Start at top for better readability
        direction='clockwise',
        showlegend=show_legend,
        # Increase font size for better readability
        textfont=dict(size=12),
        # Define consistent marker parameters
        marker=dict(
            line=dict(color='white', width=1),  # Add thin white borders between slices
            # Set colors if color map is provided
            colors=[color_map.get(name, None) for name in plot_df[names_col]] if color_map else None
        )
    )])
    
    # Determine legend position based on whether outside labels are used
    legend_y = -0.2 if show_legend and not show_outside_labels else -0.5 if show_legend else -0.1
    
    # Layout improvements for better readability
    layout_updates = {
        'title': title,
        'height': height,
        'margin': dict(t=50, b=50 if show_legend else 30, l=30, r=30),
        'showlegend': show_legend,
        'uniformtext': dict(
            minsize=10,  # Minimum text size
            mode='hide'  # Hide text that can't fit at the minimum size
        )
    }
    
    # Add width if provided
    if width:
        layout_updates['width'] = width
    
    # Add legend configuration if showing legend
    if show_legend:
        layout_updates['legend'] = dict(
            orientation="h",
            yanchor="bottom",
            y=legend_y,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
            # Adjust legend item size and spacing
            itemsizing='constant',
            itemwidth=30
        )
    
    # Update layout
    fig.update_layout(**layout_updates)
    
    # Update traces with additional styling if using outside labels
    if show_outside_labels:
        fig.update_traces(
            # Outside label line properties
            outsidetextfont=dict(size=12, color='black'),
            # Define the line connecting to outside labels
            marker_line_width=1
        )
    
    return apply_chart_theme(fig)

def create_area_chart(df, x_col, y_col, title, color_col=None, color_discrete_map=None, height=400, stacked=False):
    """
    Create an area chart.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    x_col : str
        Column to use for the x-axis
    y_col : str or list
        Column(s) to use for the y-axis
    title : str
        Chart title
    color_col : str, optional
        Column to use for color differentiation
    color_discrete_map : dict, optional
        Mapping of color values to colors
    height : int
        Chart height in pixels
    stacked : bool
        Whether to create a stacked area chart
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The created figure
    """
    if df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title=title, height=height)
        return apply_chart_theme(fig)
    
    # Handle multiple y columns
    if isinstance(y_col, list):
        # Create figure with multiple traces
        fig = go.Figure()
        
        for col in y_col:
            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[col],
                    mode="lines",
                    name=col,
                    fill="tonexty" if stacked else "tozeroy"
                )
            )
    else:
        # Create figure with plotly express
        fig = px.area(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            color_discrete_map=color_discrete_map or EXCHANGE_COLORS,
            title=title,
            height=height
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title=y_col if isinstance(y_col, str) else None,
        hovermode="x unified"
    )
    
    return apply_chart_theme(fig)

def create_scatter_chart(df, x_col, y_col, title, color_col=None, size_col=None, color_discrete_map=None, height=400):
    """
    Create a scatter chart.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    x_col : str
        Column to use for the x-axis
    y_col : str
        Column to use for the y-axis
    title : str
        Chart title
    color_col : str, optional
        Column to use for color differentiation
    size_col : str, optional
        Column to use for point size
    color_discrete_map : dict, optional
        Mapping of color values to colors
    height : int
        Chart height in pixels
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The created figure
    """
    if df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title=title, height=height)
        return apply_chart_theme(fig)
    
    # Create scatter chart
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        color_discrete_map=color_discrete_map or EXCHANGE_COLORS,
        title=title,
        height=height
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        hovermode="closest"
    )
    
    return apply_chart_theme(fig)

def create_heatmap(df, x_col, y_col, z_col, title, height=500, colorscale="Viridis"):
    """
    Create a heatmap.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    x_col : str
        Column to use for the x-axis
    y_col : str
        Column to use for the y-axis
    z_col : str
        Column to use for the z-axis (color)
    title : str
        Chart title
    height : int
        Chart height in pixels
    colorscale : str
        Colorscale to use for the heatmap
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The created figure
    """
    if df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title=title, height=height)
        return apply_chart_theme(fig)
    
    # Check if data needs to be pivoted
    if df[x_col].nunique() * df[y_col].nunique() == len(df):
        # Data needs to be pivoted
        pivot_df = df.pivot(index=y_col, columns=x_col, values=z_col)
        z_values = pivot_df.values
        x_values = pivot_df.columns
        y_values = pivot_df.index
    else:
        # Data is already in the right format or cannot be pivoted
        z_values = df[z_col]
        x_values = df[x_col]
        y_values = df[y_col]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=x_values,
        y=y_values,
        colorscale=colorscale
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        height=height,
        xaxis_title=x_col,
        yaxis_title=y_col
    )
    
    return apply_chart_theme(fig)

def create_correlation_matrix(df, title="Correlation Matrix", height=500, excluded_cols=None, colorscale="RdBu_r"):
    """
    Create a correlation matrix heatmap.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    title : str
        Chart title
    height : int
        Chart height in pixels
    excluded_cols : list, optional
        Columns to exclude from the correlation matrix
    colorscale : str
        Colorscale to use for the heatmap
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The created figure
    """
    if df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title=title, height=height)
        return apply_chart_theme(fig)
    
    # Filter out excluded columns and non-numeric columns
    if excluded_cols:
        df = df.drop(columns=[col for col in excluded_cols if col in df.columns])
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No numeric columns available for correlation",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title=title, height=height)
        return apply_chart_theme(fig)
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=colorscale,
        zmin=-1,
        zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        hovertemplate="Correlation between %{x} and %{y}<br>r = %{z:.2f}<extra></extra>"
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        height=height,
        xaxis=dict(
            tickangle=-45
        )
    )
    
    return apply_chart_theme(fig)

def create_funnel_chart(df, values_col, stages_col, title, height=400):
    """
    Create a funnel chart.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    values_col : str
        Column containing values
    stages_col : str
        Column containing stages
    title : str
        Chart title
    height : int
        Chart height in pixels
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The created figure
    """
    if df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title=title, height=height)
        return apply_chart_theme(fig)
    
    # Create funnel chart
    fig = px.funnel(
        df,
        x=values_col,
        y=stages_col,
        title=title
    )
    
    # Update layout
    fig.update_layout(
        height=height
    )
    
    return apply_chart_theme(fig)

def create_stacked_bar_chart(df, x_col, y_cols, title, colors=None, height=400):
    """
    Create a stacked bar chart.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    x_col : str
        Column to use for the x-axis
    y_cols : list
        List of columns to stack
    title : str
        Chart title
    colors : list, optional
        List of colors for the bars
    height : int
        Chart height in pixels
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The created figure
    """
    if df.empty or not y_cols:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title=title, height=height)
        return apply_chart_theme(fig)
    
    # Create figure
    fig = go.Figure()
    
    # Default colors if not provided
    if not colors:
        colors = px.colors.qualitative.Plotly
    
    # Ensure we have enough colors
    if len(colors) < len(y_cols):
        colors = colors * (len(y_cols) // len(colors) + 1)
    
    # Add traces for each column
    for i, col in enumerate(y_cols):
        fig.add_trace(go.Bar(
            x=df[x_col],
            y=df[col],
            name=col,
            marker_color=colors[i % len(colors)]
        ))
    
    # Update layout for stacked bars
    fig.update_layout(
        barmode='stack',
        title=title,
        height=height,
        xaxis_title=None,
        yaxis_title=None,
        hovermode="x unified"
    )
    
    return apply_chart_theme(fig)

def create_radar_chart(df, categories_col, values_col, title, height=500):
    """
    Create a radar chart.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    categories_col : str
        Column containing categories
    values_col : str
        Column containing values
    title : str
        Chart title
    height : int
        Chart height in pixels
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The created figure
    """
    if df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title=title, height=height)
        return apply_chart_theme(fig)
    
    # Create radar chart
    fig = go.Figure()
    
    # Add trace
    fig.add_trace(go.Scatterpolar(
        r=df[values_col],
        theta=df[categories_col],
        fill='toself',
        name=values_col
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        height=height,
        polar=dict(
            radialaxis=dict(
                visible=True
            )
        )
    )
    
    return apply_chart_theme(fig)

def display_chart(fig, use_container_width=True):
    """
    Display a chart in Streamlit.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The figure to display
    use_container_width : bool
        Whether to use the full container width
    """
    try:
        st.plotly_chart(fig, use_container_width=use_container_width)
    except Exception as e:
        st.error(f"Error displaying chart: {e}")
        logger.error(f"Error displaying chart: {e}")

def display_filterable_chart(
    fig, 
    filter_options=None,
    chart_id="chart",
    chart_title=None,
    asset="generic",
    use_container_width=True,
    custom_filters_function=None
):
    """
    Display a Plotly chart with beautiful, compact filters in a card-like layout.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The Plotly figure to display
    filter_options : dict, optional
        Dictionary containing filter options with keys:
        - 'exchanges': list of available exchanges
        - 'selected_exchanges': list of default selected exchanges, or a single default exchange
        - 'assets': list of available assets
        - 'selected_assets': list of default selected assets, or a single default asset
    chart_id : str
        Unique identifier for this chart (used in filter keys)
    chart_title : str, optional
        Title to display above the filter group. If None, will use the chart's title
    asset : str
        Current asset name (used in filter keys)
    use_container_width : bool, default=True
        Whether to use the full container width
    custom_filters_function : callable, optional
        Function that returns additional custom filters
    
    Returns:
    --------
    dict
        Dictionary containing the selected filter values:
        - 'exchange': Selected exchange (string)
        - 'asset': Selected asset (string)
        - 'custom': Any custom filter values returned by custom_filters_function
    """
    # Import filters module
    from components.filters import chart_filter_group
    
    selected = {}
    
    try:
        # If chart_title is not provided, try to extract it from the figure
        if not chart_title and hasattr(fig, 'layout') and hasattr(fig.layout, 'title') and hasattr(fig.layout.title, 'text'):
            chart_title = fig.layout.title.text
        # If still no title, use a default based on chart_id
        if not chart_title:
            chart_title = chart_id.replace('_', ' ').title()
        
        # Create a filter group for the chart if filter options are provided
        if filter_options:
            # Extract filter options
            available_exchanges = filter_options.get('exchanges')
            available_assets = filter_options.get('assets')
            default_exchange = filter_options.get('selected_exchanges')
            if isinstance(default_exchange, list) and len(default_exchange) > 0:
                default_exchange = default_exchange[0]  # Take the first if it's a list
            
            default_asset = filter_options.get('selected_assets')
            if isinstance(default_asset, list) and len(default_asset) > 0:
                default_asset = default_asset[0]  # Take the first if it's a list
            
            # Create the filter group with all options
            filter_results = chart_filter_group(
                chart_title=chart_title,
                available_exchanges=available_exchanges,
                available_assets=available_assets,
                default_exchange=default_exchange,
                default_asset=default_asset,
                key_prefix=chart_id,
                show_exchanges=bool(available_exchanges),
                show_assets=bool(available_assets),
                additional_filters=custom_filters_function
            )
            
            # Store the results
            selected = filter_results
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=use_container_width)
        else:
            # Just display the chart without filters
            st.plotly_chart(fig, use_container_width=use_container_width)
    except Exception as e:
        st.error(f"Error displaying filterable chart: {e}")
        logger.error(f"Error displaying filterable chart: {e}")
        # Still try to display the chart without filters
        try:
            st.plotly_chart(fig, use_container_width=use_container_width)
        except Exception as e2:
            logger.error(f"Failed to display chart as fallback: {e2}")
            st.error("Failed to display chart")
    
    return selected