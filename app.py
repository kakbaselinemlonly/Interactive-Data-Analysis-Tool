import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt


# ============================================================
# Backend functions
# ============================================================

from analysis_functions import (
    create_dataset_overview,
    descriptive_stats,
    create_categorical_summary,
    create_group_sum,
    create_correlation_matrix,
    create_independent_ttest,
    create_anova1,
    create_pearson_corr,
    create_ols_regression,
    create_histogram,
    create_bar_chart,
    create_box_plot,
    create_scatter_plot,
    create_heatmap,
    sql_query
)


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Interactive Data Analysis Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS design
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .main-subtitle {
        font-size: 1.05rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }

    .section-card {
        background-color: white;
        padding: 1.25rem;
        border-radius: 14px;
        border: 1px solid #E5E7EB;
        margin-bottom: 1rem;
        min-height: 140px;
    }

    .small-note {
        color: #6B7280;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Reusable helper function
# ============================================================

def display_result(result):
    """
    Display common result types returned by backend functions.
    """

    if result is None:
        st.warning(
            "No valid result is available for the selected analysis."
        )

    elif isinstance(result, str):
        st.warning(result)

    elif isinstance(result, pd.DataFrame):
        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.write(result)


# ============================================================
# Home page
# ============================================================

def show_home_page(data, uploaded_file):

    st.markdown(
        '<div class="main-title">Interactive Data Analysis Tool</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="main-subtitle">
            Explore data quality, run statistical analyses,
            create visualizations, and execute SQLite queries.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Feature cards
    # --------------------------------------------------------

    feature_col1, feature_col2, feature_col3 = st.columns(3)

    with feature_col1:
        st.markdown(
            """
            <div class="section-card">
                <h4>Explore</h4>
                <p class="small-note">
                    Review structure, missing values,
                    descriptive summaries, and correlations.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with feature_col2:
        st.markdown(
            """
            <div class="section-card">
                <h4>Analyze</h4>
                <p class="small-note">
                    Run statistical tests and investigate
                    relationships between variables.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with feature_col3:
        st.markdown(
            """
            <div class="section-card">
                <h4>Visualize</h4>
                <p class="small-note">
                    Build histograms, bar charts, box plots,
                    scatter plots, and correlation heatmaps.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # --------------------------------------------------------
    # Dataset metrics
    # --------------------------------------------------------

    row_count = data.shape[0]
    column_count = data.shape[1]
    missing_count = int(data.isna().sum().sum())
    duplicate_count = int(data.duplicated().sum())

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric(
            label="Rows",
            value=f"{row_count:,}"
        )

    with metric_col2:
        st.metric(
            label="Columns",
            value=f"{column_count:,}"
        )

    with metric_col3:
        st.metric(
            label="Missing Values",
            value=f"{missing_count:,}"
        )

    with metric_col4:
        st.metric(
            label="Duplicate Rows",
            value=f"{duplicate_count:,}"
        )

    # --------------------------------------------------------
    # Dataset details
    # --------------------------------------------------------

    with st.expander("Dataset Details"):
        st.write(f"**File name:** {uploaded_file.name}")
        st.write(f"**Rows:** {row_count:,}")
        st.write(f"**Columns:** {column_count:,}")
        st.write(
            f"**File size:** "
            f"{uploaded_file.size / 1024 / 1024:.2f} MB"
        )

    # --------------------------------------------------------
    # Data preview
    # --------------------------------------------------------

    st.subheader("Data Preview")

    if row_count == 0:
        st.warning(
            "The uploaded CSV contains column names "
            "but no data rows."
        )
        return

    max_preview_rows = min(100, row_count)

    preview_rows = st.slider(
        "Number of rows to preview",
        min_value=1,
        max_value=max_preview_rows,
        value=min(10, max_preview_rows),
        step=1,
        key="home_preview_rows"
    )

    st.dataframe(
        data.head(preview_rows),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# EDA page
# ============================================================

def show_eda_page(
    data,
    numeric_columns,
    categorical_columns
):

    st.header("Exploratory Data Analysis")

    st.write(
        "Choose an exploratory analysis method."
    )

    eda_method = st.selectbox(
        "EDA method",
        [
            "Dataset Overview",
            "Descriptive Statistics",
            "Categorical Summary",
            "Group Summary",
            "Correlation Matrix"
        ],
        key="eda_method"
    )

    # --------------------------------------------------------
    # Dataset Overview
    # --------------------------------------------------------

    if eda_method == "Dataset Overview":

        st.write(
            "Review the structure and basic data-quality metrics "
            "of the uploaded dataset."
        )

        result = create_dataset_overview(data)

        display_result(result)

    # --------------------------------------------------------
    # Descriptive Statistics
    # --------------------------------------------------------

    elif eda_method == "Descriptive Statistics":

        if len(numeric_columns) == 0:
            st.warning(
                "No numerical columns are available."
            )
            return

        selected_columns = st.multiselect(
            "Select numerical columns",
            numeric_columns,
            default=numeric_columns,
            key="eda_descriptive_columns"
        )

        if st.button(
            "Generate Descriptive Statistics",
            use_container_width=True,
            key="eda_descriptive_button"
        ):
            if len(selected_columns) == 0:
                st.warning(
                    "Please select at least one numerical column."
                )

            else:

                selected_data = data[selected_columns]

                result = descriptive_stats(
                    selected_data
                )

                display_result(result)

    # --------------------------------------------------------
    # Categorical Summary
    # --------------------------------------------------------

    elif eda_method == "Categorical Summary":

        if len(categorical_columns) == 0:
            st.warning(
                "No categorical columns are available."
            )
            return

        selected_column = st.selectbox(
            "Select a categorical column",
            categorical_columns,
            key="eda_categorical_column"
        )

        if st.button(
            "Generate Categorical Summary",
            use_container_width=True,
            key="eda_categorical_button"
        ):

            selected_data = data[
                [selected_column]
            ]

            result = create_categorical_summary(
                selected_data
            )

            if result is None:
                st.warning(
                    "No categorical summary is available."
                )

            else:
                summary_table = result[
                    selected_column
                ].reset_index()


                summary_table.columns = [
                    selected_column,
                    "count",
                    "percentage"
                ]

                st.dataframe(
                    summary_table,
                    use_container_width=True,
                    hide_index=True
                )

    # --------------------------------------------------------
    # Group Summary
    # --------------------------------------------------------

    elif eda_method == "Group Summary":

        if (
            len(categorical_columns) == 0
            or len(numeric_columns) == 0
        ):
            st.warning(
                "Group Summary requires at least one categorical "
                "and one numerical column."
            )
            return

        group_column = st.selectbox(
            "Select a grouping column",
            categorical_columns,
            key="eda_group_column"
        )


        value_columns = st.multiselect(
            "Select one or more numerical columns",
            numeric_columns,
            default=[numeric_columns[0]],
            key="eda_group_value_columns"
        )

        if st.button(
            "Generate Group Summary",
            use_container_width=True,
            key="eda_group_button"
        ):
            if len(value_columns) == 0:
                st.warning(
                    "Please select at least one numerical column."
                )

            else:

                # Call the real backend Group Summary function.
                result = create_group_sum(
                    data,
                    group_column,
                    value_columns
                )

                display_result(result)

    # --------------------------------------------------------
    # Correlation Matrix
    # --------------------------------------------------------

    elif eda_method == "Correlation Matrix":

        if len(numeric_columns) < 2:
            st.warning(
                "A correlation matrix requires at least "
                "two numerical columns."
            )
            return

        selected_columns = st.multiselect(
            "Select numerical columns",
            numeric_columns,
            default=numeric_columns,
            key="eda_correlation_columns"
        )

        if st.button(
            "Generate Correlation Matrix",
            use_container_width=True,
            key="eda_correlation_button"
        ):
            if len(selected_columns) < 2:
                st.warning(
                    "Please select at least two numerical columns."
                )

            else:
  
                # Call the backend correlation function.
                selected_data = data[
                    selected_columns
                ]

                result = create_correlation_matrix(
                    selected_data
                )

                display_result(result)


# ============================================================
# Statistics page
# ============================================================

def show_statistics_page(
    data,
    numeric_columns,
    categorical_columns
):

    st.header("Statistical Analysis")

    st.write(
        "Choose a statistical method and select the required variables."
    )

    statistical_method = st.selectbox(
        "Statistical method",
        [
            "Independent T-test",
            "One-way ANOVA",
            "Pearson Correlation",
            "OLS Regression"
        ],
        key="statistical_method"
    )

    # --------------------------------------------------------
    # Independent T-test
    # --------------------------------------------------------

    if statistical_method == "Independent T-test":

        st.info(
            "An independent t-test requires one numerical outcome "
            "and one categorical variable containing exactly two groups."
        )

        if (
            len(numeric_columns) == 0
            or len(categorical_columns) == 0
        ):
            st.warning(
                "This test requires at least one numerical "
                "and one categorical column."
            )
            return

        numerical_column = st.selectbox(
            "Select a numerical outcome",
            numeric_columns,
            key="ttest_numeric_column"
        )

        group_column = st.selectbox(
            "Select a two-group categorical column",
            categorical_columns,
            key="ttest_group_column"
        )


        # Add a button and call the real backend t-test function.
        if st.button(
            "Run Independent T-test",
            use_container_width=True,
            key="ttest_button"
        ):
            result = create_independent_ttest(
                data,
                group_column,
                numerical_column
            )

            display_result(result)

    # --------------------------------------------------------
    # One-way ANOVA
    # --------------------------------------------------------

    elif statistical_method == "One-way ANOVA":

        st.info(
            "One-way ANOVA compares the mean of one numerical "
            "outcome across three or more groups."
        )

        if (
            len(numeric_columns) == 0
            or len(categorical_columns) == 0
        ):
            st.warning(
                "ANOVA requires at least one numerical "
                "and one categorical column."
            )
            return

        numerical_column = st.selectbox(
            "Select a numerical outcome",
            numeric_columns,
            key="anova_numeric_column"
        )

        group_column = st.selectbox(
            "Select a grouping column",
            categorical_columns,
            key="anova_group_column"
        )


        # Add a button and call create_anova1().
        if st.button(
            "Run One-way ANOVA",
            use_container_width=True,
            key="anova_button"
        ):
            result = create_anova1(
                data,
                group_column,
                numerical_column
            )

            display_result(result)

    # --------------------------------------------------------
    # Pearson Correlation
    # --------------------------------------------------------

    elif statistical_method == "Pearson Correlation":

        if len(numeric_columns) < 2:
            st.warning(
                "Pearson correlation requires at least "
                "two numerical columns."
            )
            return

        first_column = st.selectbox(
            "Select the first numerical column",
            numeric_columns,
            key="pearson_first_column"
        )

        second_options = [
            column
            for column in numeric_columns
            if column != first_column
        ]

        second_column = st.selectbox(
            "Select the second numerical column",
            second_options,
            key="pearson_second_column"
        )


        # Add the missing analysis button and backend call.
        if st.button(
            "Run Pearson Correlation",
            use_container_width=True,
            key="pearson_button"
        ):
            result = create_pearson_corr(
                data,
                first_column,
                second_column
            )

            display_result(result)

    # --------------------------------------------------------
    # OLS Regression
    # --------------------------------------------------------

    elif statistical_method == "OLS Regression":

        if len(numeric_columns) < 2:
            st.warning(
                "OLS regression requires at least "
                "two numerical columns."
            )
            return

        dependent_column = st.selectbox(
            "Select the dependent variable",
            numeric_columns,
            key="ols_dependent_column"
        )

        independent_options = [
            column
            for column in numeric_columns
            if column != dependent_column
        ]

        independent_column = st.selectbox(
            "Select the independent variable",
            independent_options,
            key="ols_independent_column"
        )

        st.caption(
            "The model estimates the dependent variable "
            "from one selected independent variable."
        )

        # Add the missing OLS button and backend call.
        # Backend order:
        # num_col1 = independent variable
        # num_col2 = dependent variable
        if st.button(
            "Run OLS Regression",
            use_container_width=True,
            key="ols_button"
        ):
            result = create_ols_regression(
                data,
                independent_column,
                dependent_column
            )

            display_result(result)


# ============================================================
# Visualization page
# ============================================================

def show_visualization_page(
    data,
    numeric_columns,
    categorical_columns
):
    st.header("Data Visualization")

    st.write(
        "Select a chart type and choose the columns to visualize."
    )

    chart_type = st.selectbox(
        "Choose a chart type",
        [
            "Histogram",
            "Bar Chart",
            "Box Plot",
            "Scatter Plot",
            "Correlation Heatmap"
        ],
        key="chart_type"
    )

    # --------------------------------------------------------
    # Histogram
    # --------------------------------------------------------

    if chart_type == "Histogram":

        if len(numeric_columns) == 0:
            st.warning(
                "No numerical columns are available."
            )
            return

        selected_column = st.selectbox(
            "Select a numerical column",
            numeric_columns,
            key="histogram_column"
        )

        if st.button(
            "Create Histogram",
            use_container_width=True,
            key="histogram_button"
        ):
            result = create_histogram(
                data,
                selected_column
            )

            if isinstance(result, str):
                st.warning(result)

            else:
                st.pyplot(result)
                plt.close(result)

    # --------------------------------------------------------
    # Bar Chart
    # --------------------------------------------------------

    elif chart_type == "Bar Chart":

        if len(categorical_columns) == 0:
            st.warning(
                "No categorical columns are available."
            )
            return

        selected_column = st.selectbox(
            "Select a categorical column",
            categorical_columns,
            key="bar_chart_column"
        )

        if st.button(
            "Create Bar Chart",
            use_container_width=True,
            key="bar_chart_button"
        ):
            result = create_bar_chart(
                data,
                selected_column
            )

            if isinstance(result, str):
                st.warning(result)

            else:
                st.pyplot(result)
                plt.close(result)

    # --------------------------------------------------------
    # Box Plot
    # --------------------------------------------------------

    elif chart_type == "Box Plot":

        if (
            len(numeric_columns) == 0
            or len(categorical_columns) == 0
        ):
            st.warning(
                "A box plot requires at least one numerical "
                "and one categorical column."
            )
            return

        cat_col = st.selectbox(
            "Select a categorical column",
            categorical_columns,
            key="box_plot_categorical_column"
        )

        num_col = st.selectbox(
            "Select a numerical column",
            numeric_columns,
            key="box_plot_numerical_column"
        )

        if st.button(
            "Create Box Plot",
            use_container_width=True,
            key="box_plot_button"
        ):
            result = create_box_plot(
                data,
                cat_col,
                num_col
            )

            if isinstance(result, str):
                st.warning(result)

            else:
                st.pyplot(result)
                plt.close(result)

    # --------------------------------------------------------
    # Scatter Plot
    # --------------------------------------------------------

    elif chart_type == "Scatter Plot":

        if len(numeric_columns) < 2:
            st.warning(
                "A scatter plot requires at least "
                "two numerical columns."
            )
            return

        x_column = st.selectbox(
            "Select the X-axis column",
            numeric_columns,
            key="scatter_x_column"
        )

        y_options = [
            column
            for column in numeric_columns
            if column != x_column
        ]

        y_column = st.selectbox(
            "Select the Y-axis column",
            y_options,
            key="scatter_y_column"
        )

        if st.button(
            "Create Scatter Plot",
            use_container_width=True,
            key="scatter_plot_button"
        ):
            result = create_scatter_plot(
                data,
                x_column,
                y_column
            )

            if isinstance(result, str):
                st.warning(result)

            else:
                st.pyplot(result)
                plt.close(result)

    # --------------------------------------------------------
    # Correlation Heatmap
    # --------------------------------------------------------

    elif chart_type == "Correlation Heatmap":

        if len(numeric_columns) < 2:
            st.warning(
                "A heatmap requires at least "
                "two numerical columns."
            )
            return

        st.info(
            "The heatmap uses all numerical columns in the dataset."
        )

        if st.button(
            "Create Correlation Heatmap",
            use_container_width=True,
            key="heatmap_button"
        ):
            result = create_heatmap(data)

            if isinstance(result, str):
                st.warning(result)

            else:
                st.pyplot(result)
                plt.close(result)


# ============================================================
# SQL page
# ============================================================

def show_sql_page(data):
    st.header("SQL Query")

    st.info(
        "The uploaded dataset is available as a SQLite "
        "table named `data`."
    )

    st.caption(
        "Only SELECT and WITH queries are supported."
    )

    query = st.text_area(
        "Enter a SQLite query",
        value="""SELECT *
FROM data
LIMIT 10;""",
        height=180,
        key="sql_query_input"
    )

    if st.button(
        "Run SQL Query",
        use_container_width=True,
        key="run_sql_button"
    ):
        result = sql_query(
            data,
            query
        )

        if isinstance(result, str):
            st.error(result)

        else:
            st.success(
                f"Query returned {len(result):,} rows."
            )

            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.title("Data Explorer")

    st.caption(
        "Explore, analyze, visualize, and query CSV data."
    )

    st.divider()

    selected_page = st.radio(
        "Navigation",
        [
            "Home",
            "EDA",
            "Statistics",
            "Visualization",
            "SQL"
        ],
        key="selected_page"
    )

    st.divider()

    st.caption(
        "SQL queries use SQLite syntax."
    )


# ============================================================
# Global upload section
# ============================================================

st.subheader("Upload Dataset")

st.write(
    "Upload a CSV file to begin. "
    "The configured maximum upload size is 1000 MB."
)

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"],
    help="Only CSV files are supported.",
    key="main_csv_uploader"
)


# ============================================================
# Stop until a file is uploaded
# ============================================================

if uploaded_file is None:
    st.info(
        "Please upload a CSV file to unlock the analysis tools."
    )

    st.stop()


# ============================================================
# Read CSV
# ============================================================

try:
    df = pd.read_csv(
        uploaded_file,
        encoding="utf-8"
    )

except UnicodeDecodeError:
    uploaded_file.seek(0)

    try:
        df = pd.read_csv(
            uploaded_file,
            encoding="latin-1"
        )

    except Exception as error:
        st.error(
            f"Unable to read the CSV file: {error}"
        )
        st.stop()

except Exception as error:
    st.error(
        f"Unable to read the CSV file: {error}"
    )
    st.stop()


st.success(
    f"{uploaded_file.name} uploaded successfully."
)


# ============================================================
# Identify column types
# ============================================================

numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()

categorical_columns = df.select_dtypes(
    exclude="number"
).columns.tolist()


# ============================================================
# Page routing
# ============================================================

if selected_page == "Home":
    show_home_page(
        df,
        uploaded_file
    )

elif selected_page == "EDA":
    show_eda_page(
        df,
        numeric_columns,
        categorical_columns
    )

elif selected_page == "Statistics":
    show_statistics_page(
        df,
        numeric_columns,
        categorical_columns
    )

elif selected_page == "Visualization":
    show_visualization_page(
        df,
        numeric_columns,
        categorical_columns
    )

elif selected_page == "SQL":
    show_sql_page(df)