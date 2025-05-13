# Streamlit Application Design Structure

This document outlines the architecture and design patterns used in the Izun Liquidity Report Streamlit application. Use this as a reference when creating similar multi-page data dashboards with Streamlit.

## Application Architecture

The application follows a modular design with clear separation of concerns:

```
app/
├── main.py                     # Main entry point and home page
├── pages/                      # Additional pages (auto-discovered by Streamlit)
│   ├── spot_and_perp_metrics.py# Combined market metrics
│   ├── spot_metrics.py         # Spot market metrics
│   ├── ai_questions.py         # AI-powered analysis
│   ├── historical_reports.py   # Historical data analysis
│   └── download_center.py      # Data export functionality
├── components/                 # Reusable UI components
│   ├── data_display.py         # Data visualization components
│   ├── filters.py              # Filter UI components
│   └── sidebar.py              # Sidebar components and layout
├── utils/                      # Utility functions
│   ├── api.py                  # API interactions
│   ├── data_processing.py      # Data manipulation
│   ├── visualization.py        # Chart creation
│   └── ...                     # Other utilities
```

## Key Design Patterns

### 1. Multi-page Structure

The application uses Streamlit's native multi-page app structure where:
- `main.py` serves as the entry point and home page
- Additional pages in the `pages/` folder are automatically discovered by Streamlit
- Each page is independent but shares session state and utilities

### 2. Shared Session State

Global state is managed through Streamlit's session state:

```python
# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = load_latest_data()
    
if 'selected_filters' not in st.session_state:
    st.session_state.selected_filters = {
        'market_type': 'all',
        'exchanges': [],
        'cryptocurrencies': []
    }
```

### 3. Reusable Components

UI components are modularized for reuse across pages:

```python
# components/sidebar.py
def render_sidebar():
    """Renders the sidebar with consistent filters and navigation"""
    st.sidebar.title("Filters")
    
    # Market type selector
    market_type = st.sidebar.radio(
        "Market Type",
        ["All Markets", "Spot Markets", "Perpetual Markets"],
        index=0
    )
    
    # Exchange selector
    exchanges = st.sidebar.multiselect(
        "Exchanges",
        get_available_exchanges(),
        default=[]
    )
    
    # Update session state
    st.session_state.selected_filters.update({
        'market_type': market_type.lower().split()[0],
        'exchanges': exchanges
    })
    
    # Add additional sidebar elements
    render_sidebar_info()
    render_data_refresh_button()
```

### 4. Data Flow Pattern

The data flow follows a consistent pattern:

1. Data Loading (from cached files)
2. Data Filtering (based on user selections)
3. Data Processing (compute metrics, aggregations)
4. Data Visualization (render charts and tables)

```python
# Typical page structure
def main():
    # 1. Load data
    data = st.session_state.data
    
    # 2. Apply filters from session state
    filtered_data = apply_filters(data, st.session_state.selected_filters)
    
    # 3. Process data for visualization
    summary_metrics = calculate_summary_metrics(filtered_data)
    
    # 4. Visualize data
    st.header("Market Overview")
    render_metrics_cards(summary_metrics)
    render_price_chart(filtered_data)
    render_volume_chart(filtered_data)
```

### 5. Consistent Layout Structure

Each page follows a consistent layout pattern:

```python
def page_layout():
    # 1. Sidebar (shared across all pages)
    components.sidebar.render_sidebar()
    
    # 2. Page header
    st.title("Page Title")
    st.write("Brief description of this page's purpose")
    
    # 3. Filter indicators
    render_active_filters(st.session_state.selected_filters)
    
    # 4. Main content area with tabs or sections
    tab1, tab2 = st.tabs(["Section 1", "Section 2"])
    
    with tab1:
        # Tab 1 content
        st.subheader("Section 1")
        # ...visualizations
    
    with tab2:
        # Tab 2 content
        st.subheader("Section 2")
        # ...visualizations
        
    # 5. Additional information or footnotes
    st.caption("Data source: API Name | Last updated: timestamp")
```

### 6. Responsive Visualization Components

The visualization components are designed to be responsive:

```python
# components/data_display.py
def render_metrics_cards(metrics_dict):
    """Display metrics in responsive card layout"""
    cols = st.columns(len(metrics_dict))
    for i, (metric_name, value) in enumerate(metrics_dict.items()):
        with cols[i]:
            st.metric(
                label=metric_name,
                value=format_value(value),
                delta=calculate_delta(value)
            )

def render_chart(data, x_col, y_col, title, chart_type='line'):
    """Responsive chart component with consistent styling"""
    fig = create_figure(data, x_col, y_col, chart_type)
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
```

## UI/UX Design Guidelines

### 1. Color System

The application uses a consistent color system:

```python
# utils/style_guide.py
COLORS = {
    'primary': '#3366CC',      # Primary brand color
    'secondary': '#FF9900',    # Secondary brand color
    'spot': '#4CAF50',         # Green for spot markets
    'perpetual': '#FF9800',    # Orange for perpetual markets
    'positive': '#4CAF50',     # Green for positive changes
    'negative': '#F44336',     # Red for negative changes
    'neutral': '#9E9E9E',      # Gray for neutral/unchanged values
    'background': '#1E1E1E',   # Dark background
    'text': '#FFFFFF',         # Light text for dark theme
    'grid': '#333333',         # Grid lines
}

def apply_theme(fig):
    """Apply consistent theme to Plotly figures"""
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font={'color': COLORS['text']},
        legend={'font': {'color': COLORS['text']}},
        xaxis={'gridcolor': COLORS['grid']},
        yaxis={'gridcolor': COLORS['grid']}
    )
```

### 2. Data Tables

Consistent table styling with conditional formatting:

```python
def display_data_table(df, formatting_rules=None):
    """Display a styled data table with conditional formatting"""
    # Apply default styling
    styled_df = df.style.format(precision=2)
    
    # Apply conditional formatting
    if formatting_rules:
        for column, rule in formatting_rules.items():
            if rule['type'] == 'color_scale':
                styled_df = styled_df.background_gradient(
                    cmap=rule['cmap'],
                    subset=[column]
                )
            elif rule['type'] == 'bar':
                styled_df = styled_df.bar(
                    subset=[column],
                    color=rule['color']
                )
    
    # Display the table
    st.dataframe(styled_df, use_container_width=True)
```

### 3. Interactive Filters

Consistent use of interactive filters:

```python
def create_date_filter(default_days=7):
    """Create a standardized date range filter"""
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=default_days)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now()
        )
    return start_date, end_date

def create_numeric_range_filter(label, min_val, max_val, default_range=None):
    """Create a standardized numeric range filter"""
    if default_range is None:
        default_range = (min_val, max_val)
    return st.slider(
        label,
        min_value=min_val,
        max_value=max_val,
        value=default_range
    )
```

## Application State Management

### 1. Data Loading and Caching

Efficient data loading with Streamlit's caching:

```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_latest_data():
    """Load the most recent data file with caching"""
    data_files = sorted(glob.glob('data/*_data.csv'))
    if not data_files:
        return pd.DataFrame()
    
    latest_file = data_files[-1]
    return pd.read_csv(latest_file)

@st.cache_data
def calculate_summary_metrics(data):
    """Calculate summary metrics with caching"""
    # Expensive calculation that's cached
    return {
        'Total Volume': data['Volume_24h_Quote'].sum(),
        'Average Price': data['Price'].mean(),
        # ... other metrics
    }
```

### 2. User Preferences Storage

Store user preferences in session state:

```python
# Initialize user preferences
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        'default_view': 'spot',
        'default_timeframe': '7d',
        'default_exchanges': ['binance', 'coinbase'],
        'theme': 'dark',
        'chart_type': 'line'
    }

# Allow users to update preferences
def update_preferences():
    with st.expander("Display Settings"):
        st.session_state.user_preferences['default_view'] = st.selectbox(
            "Default View",
            options=['spot', 'perpetual', 'all'],
            index=['spot', 'perpetual', 'all'].index(
                st.session_state.user_preferences['default_view']
            )
        )
        
        st.session_state.user_preferences['chart_type'] = st.selectbox(
            "Chart Type",
            options=['line', 'bar', 'area'],
            index=['line', 'bar', 'area'].index(
                st.session_state.user_preferences['chart_type']
            )
        )
```

## Advanced Features Implementation

### 1. Dynamic Chart Creation

Function to create dynamic charts based on user selections:

```python
def create_dynamic_chart(data, chart_type, x_axis, y_axes, color_by=None):
    """Create a dynamic chart based on user selections"""
    if chart_type == 'line':
        fig = px.line(
            data, 
            x=x_axis, 
            y=y_axes,
            color=color_by if color_by else None,
            color_discrete_map={
                'spot': COLORS['spot'],
                'perpetual': COLORS['perpetual']
            }
        )
    elif chart_type == 'bar':
        fig = px.bar(
            data, 
            x=x_axis, 
            y=y_axes,
            color=color_by if color_by else None
        )
    elif chart_type == 'scatter':
        fig = px.scatter(
            data, 
            x=x_axis, 
            y=y_axes,
            color=color_by if color_by else None,
            size_max=15
        )
    
    # Apply consistent styling
    apply_theme(fig)
    return fig
```

### 2. Data Export Capabilities

Functions to enable data export:

```python
def create_download_button(data, filename, button_text="Download Data"):
    """Create a download button for the given data"""
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

def export_visualization(fig, filename, format='png'):
    """Export a visualization as image"""
    img_bytes = fig.to_image(format=format, scale=2)
    b64 = base64.b64encode(img_bytes).decode()
    href = f'<a href="data:image/{format};base64,{b64}" download="{filename}.{format}">Download {format.upper()}</a>'
    st.markdown(href, unsafe_allow_html=True)
```

## Streamlit Optimization Techniques

### 1. Performance Optimization

Techniques to optimize Streamlit performance:

```python
# 1. Use st.cache_data for expensive operations
@st.cache_data
def expensive_calculation(data):
    # Expensive operation
    return result

# 2. Avoid unnecessary re-renders by using keys
st.text_input("Input", key="unique_key")

# 3. Use containers to better control UI updates
container = st.container()
with container:
    # Content that may be updated together
    st.metric("Value", 100)
    st.write("Description")

# 4. Use st.empty() for content that needs frequent updates
placeholder = st.empty()
for i in range(100):
    # This replaces the content without creating a new element
    placeholder.metric("Value", i)
    time.sleep(0.1)
```

### 2. Layout Techniques

Advanced layout techniques:

```python
# 1. Create responsive grid layouts
def create_metrics_grid(metrics, num_columns=3):
    """Create a responsive grid of metrics"""
    # Calculate number of rows needed
    num_rows = (len(metrics) + num_columns - 1) // num_columns
    
    for row in range(num_rows):
        cols = st.columns(num_columns)
        for col in range(num_columns):
            idx = row * num_columns + col
            if idx < len(metrics):
                with cols[col]:
                    name, value = metrics[idx]
                    st.metric(name, value)

# 2. Dynamic sidebar width
def set_sidebar_width(width=350):
    """Set the sidebar width using custom CSS"""
    st.markdown(
        f"""
        <style>
            section[data-testid="stSidebar"] > div {{
                width: {width}px !important;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
```

## Data Visualization Best Practices

### 1. Consistent Chart Creation

```python
def create_standard_chart(data, x, y, title, type='line', color_column=None):
    """Create a standardized chart with consistent styling"""
    # Choose chart type
    if type == 'line':
        fig = px.line(data, x=x, y=y, color=color_column)
    elif type == 'bar':
        fig = px.bar(data, x=x, y=y, color=color_column)
    elif type == 'scatter':
        fig = px.scatter(data, x=x, y=y, color=color_column)
    
    # Apply consistent styling
    fig.update_layout(
        title=title,
        template='plotly_dark',
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation='h', y=-0.2)
    )
    
    # Apply axis formatting
    fig.update_xaxes(title=x.replace('_', ' ').title())
    fig.update_yaxes(title=y.replace('_', ' ').title())
    
    return fig
```

### 2. Chart Type Selection Guide

Guidelines for when to use specific chart types:

```python
def suggest_chart_type(data_type, num_categories, time_series=False):
    """Suggest appropriate chart type based on data characteristics"""
    if time_series:
        if data_type == 'categorical':
            return 'grouped_bar'
        else:
            return 'line'
    
    if data_type == 'categorical':
        if num_categories <= 5:
            return 'pie'
        else:
            return 'bar'
    
    if data_type == 'continuous':
        if 'comparison' in analysis_type:
            return 'bar'
        elif 'distribution' in analysis_type:
            return 'histogram'
        elif 'correlation' in analysis_type:
            return 'scatter'
    
    # Default fallback
    return 'bar'
```

## Implementing This Design in Your Project

1. **Start with folder structure**: Create the basic directory layout following the architecture shown
2. **Implement core components**: Build the reusable components (sidebar, filters, data display)
3. **Create utility functions**: Implement the styling, data processing, and visualization utilities
4. **Build main pages**: Develop the main app and individual pages using the consistent patterns
5. **Add interactivity**: Implement filters, dynamic charts, and session state management
6. **Optimize performance**: Apply caching and performance optimization techniques
7. **Enhance user experience**: Add download capabilities, responsive layouts, and error handling

This modular approach allows for a maintainable, scalable Streamlit application that provides a consistent user experience.