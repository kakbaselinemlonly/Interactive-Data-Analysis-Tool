import pandas as pd
import numpy as np
import sqlite3

from scipy import stats

import statsmodels.api as sm
from statsmodels.formula.api import ols

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Common missing-value markers
# ============================================================

MISSING_MARKERS = [
    ".",
    "N/A",
    "n/a",
    "NA",
    "",
    "Unknown",
    "unknown"
]

# ============================================================
# EDA — Dataset Overview
# ============================================================

def create_dataset_overview(data):
    overview = pd.DataFrame({
        "metric": [
            "Number of rows",
            "Number of columns",
            "Numerical columns",
            "Categorical columns",
            "Missing values",
            "Duplicate rows"
        ],
        "value": [
            data.shape[0],
            data.shape[1],
            len(
                data.select_dtypes(
                    include=np.number
                ).columns
            ),
            len(
                data.select_dtypes(
                    include=["object", "category"]
                ).columns
            ),
            data.isna().sum().sum(),
            data.duplicated().sum()
        ]
    })

    return overview


# ============================================================
# EDA — Descriptive Statistics
# ============================================================

def descriptive_stats(data):
    # Find all numerical columns.
    num_cols = (
        data.select_dtypes(
            include=np.number
        )
        .columns
        .to_list()
    )

    if len(num_cols) == 0:
        return None

    descriptive_statistics = (
        data[num_cols]
        .describe()
        .transpose()
        .round(2)
    )

    return descriptive_statistics


# ============================================================
# EDA — Categorical Summary
# ============================================================

def create_categorical_summary(data):
    categorical_columns = (
        data.select_dtypes(
            include=["object", "category"]
        )
        .columns
        .to_list()
    )

    if len(categorical_columns) == 0:
        return None

    categorical_results = {}

    for column in categorical_columns:
        count_result = data[column].value_counts(
            dropna=False
        )

        percentage_result = (
            data[column]
            .value_counts(
                dropna=False,
                normalize=True
            )
            * 100
        )

        summary_table = pd.DataFrame({
            "count": count_result,
            "percentage": percentage_result.round(2)
        })

        categorical_results[column] = summary_table

    return categorical_results


# ============================================================
# EDA — Group Summary
# ============================================================

def create_group_sum(data, cat_col, num_cols):
    if cat_col is None:
        return None

    if num_cols is None or len(num_cols) == 0:
        return None

    group_by_sum = (
        data
        .groupby(
            cat_col,
            dropna=False
        )[num_cols]
        .agg([
            "count",
            "mean",
            "median",
            "min",
            "max"
        ])
        .round(2)
    )


    group_by_sum.columns = [
        f"{column}: {statistic}"
        for column, statistic in group_by_sum.columns
    ]

    group_by_sum = group_by_sum.reset_index()

    return group_by_sum


# ============================================================
# EDA — Correlation Matrix
# ============================================================

def create_correlation_matrix(data):
    num_cols = (
        data.select_dtypes(
            include=np.number
        )
        .columns
        .to_list()
    )

    if len(num_cols) < 2:
        return None

    correlation_matrix = (
        data[num_cols]
        .corr()
        .round(2)
    )

    return correlation_matrix


# ============================================================
# SQL
# ============================================================

def sql_query(data, query):
    if query is None or query.strip() == "":
        return "A valid SQL query is needed"

    first_word = query.strip().split()[0].upper()

    if first_word not in ["SELECT", "WITH"]:
        return "Only SELECT / WITH are allowed to be the first word"


    connection = sqlite3.connect(":memory:")

    try:
        data.to_sql(
            "data",
            connection,
            if_exists="replace",
            index=False
        )

        result = pd.read_sql_query(
            query,
            connection
        )

        return result

    except Exception as error:
        return f"SQL Error: {error}"

    finally:
        connection.close()


# ============================================================
# Statistics — Independent T-test
# ============================================================

def create_independent_ttest(data, cat_col, num_col):
    clean_data = data.replace(
        MISSING_MARKERS,
        np.nan
    ).copy()


    clean_data[num_col] = pd.to_numeric(
        clean_data[num_col],
        errors="coerce"
    )

    diff_groups = (
        clean_data[cat_col]
        .dropna()
        .unique()
    )

    if len(diff_groups) != 2:
        return "Need exactly 2 groups for independent t-test"

    group1 = diff_groups[0]
    group2 = diff_groups[1]

    cat_data1 = clean_data.loc[
        clean_data[cat_col] == group1,
        num_col
    ].dropna()

    cat_data2 = clean_data.loc[
        clean_data[cat_col] == group2,
        num_col
    ].dropna()

    if len(cat_data1) < 2 or len(cat_data2) < 2:
        return "Each group needs at least 2 valid numerical values"

    independent_ttest = stats.ttest_ind(
        cat_data1,
        cat_data2,
        equal_var=False
    )

    summary = pd.DataFrame({
        "metric": [
            "Group 1",
            "Group 2",
            "Group 1 sample size",
            "Group 2 sample size",
            "T-statistic",
            "P-value",
            "Degrees of freedom"
        ],
        "value": [
            group1,
            group2,
            len(cat_data1),
            len(cat_data2),
            independent_ttest.statistic,
            independent_ttest.pvalue,
            independent_ttest.df
        ]
    })

    return summary


# ============================================================
# Statistics — One-way ANOVA
# ============================================================

def create_anova1(data, cat_col, num_col):
    clean_data = data.replace(
        MISSING_MARKERS,
        np.nan
    ).copy()


    clean_data[num_col] = pd.to_numeric(
        clean_data[num_col],
        errors="coerce"
    )

    analysis_data = clean_data[
        [cat_col, num_col]
    ].dropna()

    diff_groups = analysis_data[
        cat_col
    ].unique()

    if len(diff_groups) < 3:
        return "Need at least 3 groups of categories for ANOVA"


    if len(analysis_data) < 6:
        return "Need at least 6 valid observations for ANOVA"

    formula = f"{num_col} ~ C({cat_col})"

    try:
        model = ols(
            formula,
            data=analysis_data
        ).fit()

        anova_table = sm.stats.anova_lm(
            model,
            typ=2
        )

        return anova_table.round(4)


    except Exception as error:
        return f"ANOVA Error: {error}"


# ============================================================
# Statistics — Pearson Correlation
# ============================================================

def create_pearson_corr(
    data,
    num_col1,
    num_col2
):
    if num_col1 == num_col2:
        return "Please select two different numerical columns"

    clean_data = data.replace(
        MISSING_MARKERS,
        np.nan
    ).copy()


    clean_data[num_col1] = pd.to_numeric(
        clean_data[num_col1],
        errors="coerce"
    )

    clean_data[num_col2] = pd.to_numeric(
        clean_data[num_col2],
        errors="coerce"
    )

    paired_data = clean_data[
        [num_col1, num_col2]
    ].dropna()

    if len(paired_data) < 3:
        return "Need at least 3 valid paired observations"


    if paired_data[num_col1].nunique() < 2:
        return f"{num_col1} needs at least 2 different values"

    if paired_data[num_col2].nunique() < 2:
        return f"{num_col2} needs at least 2 different values"

    pearson_corr = stats.pearsonr(
        paired_data[num_col1],
        paired_data[num_col2]
    )

    if pearson_corr.pvalue < 0.001:
        p_value = "< 0.001"

    elif pearson_corr.pvalue < 0.05:
        p_value = "< 0.05"

    else:
        p_value = round(
            pearson_corr.pvalue,
            4
        )

    summary = pd.DataFrame({
        "metric": [
            "Pearson correlation",
            "P-value",
            "Number of paired observations",
            "Note"
        ],
        "value": [
            round(pearson_corr.statistic, 4),
            p_value,
            len(paired_data),
            "Remember, correlation is not equal to causation."
        ]
    })

    return summary


# ============================================================
# Statistics — OLS Regression
# ============================================================

def create_ols_regression(
    data,
    num_col1,
    num_col2
):
    if num_col1 == num_col2:
        return "Please select two different numerical columns"

    clean_data = data.replace(
        MISSING_MARKERS,
        np.nan
    ).copy()

    clean_data[num_col1] = pd.to_numeric(
        clean_data[num_col1],
        errors="coerce"
    )

    clean_data[num_col2] = pd.to_numeric(
        clean_data[num_col2],
        errors="coerce"
    )

    selected_data = clean_data[
        [num_col1, num_col2]
    ].dropna()


    if len(selected_data) < 3:
        return "Need at least 3 valid observations for OLS regression"

    if selected_data[num_col1].nunique() < 2:
        return f"{num_col1} needs at least 2 different values"

    formula = f"{num_col2} ~ {num_col1}"

    try:
        model = ols(
            formula,
            data=selected_data
        ).fit()

        coefficient = model.params[num_col1]
        intercept = model.params["Intercept"]
        r_squared = model.rsquared
        p_value = model.pvalues[num_col1]
        sample_size = len(selected_data)

        summary = pd.DataFrame({
            "metric": [
                "Dependent variable",
                "Independent variable",
                "Coefficient",
                "Intercept",
                "R-squared",
                "P-value",
                "Number of observations"
            ],
            "value": [
                num_col2,
                num_col1,
                round(coefficient, 4),
                round(intercept, 4),
                round(r_squared, 4),
                round(p_value, 4),
                sample_size
            ]
        })

        return summary


    except Exception as error:
        return f"OLS Regression Error: {error}"


# ============================================================
# Visualization — Histogram
# ============================================================

def create_histogram(data, num_col):
    clean_data = data.replace(
        MISSING_MARKERS,
        np.nan
    ).copy()


    plot_data = pd.to_numeric(
        clean_data[num_col],
        errors="coerce"
    ).dropna()

    if len(plot_data) == 0:
        return "No available data for plotting histogram"

    fig, ax = plt.subplots()

    ax.hist(
        plot_data,
        bins=20
    )

    ax.set_title(
        f"Distribution of {num_col}"
    )

    ax.set_xlabel(num_col)
    ax.set_ylabel("Frequency")

    fig.tight_layout()

    return fig


# ============================================================
# Visualization — Bar Chart
# ============================================================

def create_bar_chart(data, cat_col):
    clean_data = data.replace(
        MISSING_MARKERS,
        np.nan
    )

    plot_data = clean_data[
        cat_col
    ].dropna()

    if len(plot_data) == 0:

        return "No available data for plotting barchart"

    category_count = plot_data.value_counts()

    fig, ax = plt.subplots()

    ax.bar(
        category_count.index.astype(str),
        category_count.values
    )

    ax.set_title(
        f"Count of {cat_col}"
    )

    ax.set_xlabel(cat_col)
    ax.set_ylabel("Count")

    ax.tick_params(
        axis="x",
        rotation=45
    )

    fig.tight_layout()

    return fig


# ============================================================
# Visualization — Box Plot
# ============================================================

def create_box_plot(
    data,
    cat_col,
    num_col
):
    clean_data = data.replace(
        MISSING_MARKERS,
        np.nan
    ).copy()


    clean_data[num_col] = pd.to_numeric(
        clean_data[num_col],
        errors="coerce"
    )

    plot_data = clean_data[
        [cat_col, num_col]
    ].dropna()

    if len(plot_data) == 0:
        return "No available data for plotting box plot"

    fig, ax = plt.subplots()

    sns.boxplot(
        data=plot_data,
        x=cat_col,
        y=num_col,
        ax=ax
    )

    ax.set_title(
        f"Distribution of {num_col} by {cat_col}"
    )

    ax.set_xlabel(cat_col)
    ax.set_ylabel(num_col)

    ax.tick_params(
        axis="x",
        rotation=45
    )

    fig.tight_layout()

    return fig


# ============================================================
# Visualization — Scatter Plot
# ============================================================

def create_scatter_plot(
    data,
    num_col1,
    num_col2
):
    if num_col1 == num_col2:
        return "Please select two different numerical columns"

    clean_data = data.replace(
        MISSING_MARKERS,
        np.nan
    ).copy()


    clean_data[num_col1] = pd.to_numeric(
        clean_data[num_col1],
        errors="coerce"
    )

    clean_data[num_col2] = pd.to_numeric(
        clean_data[num_col2],
        errors="coerce"
    )

    plot_data = clean_data[
        [num_col1, num_col2]
    ].dropna()

    if len(plot_data) == 0:
        return "No available data for plotting scatter plot"

    fig, ax = plt.subplots()

    sns.scatterplot(
        data=plot_data,
        x=num_col1,
        y=num_col2,
        ax=ax
    )

    ax.set_title(
        f"{num_col1} vs {num_col2}"
    )

    ax.set_xlabel(num_col1)
    ax.set_ylabel(num_col2)

    fig.tight_layout()

    return fig


# ============================================================
# Visualization — Correlation Heatmap
# ============================================================

def create_heatmap(data):
    clean_data = data.replace(
        MISSING_MARKERS,
        np.nan
    )

    numeric_data = clean_data.select_dtypes(
        include="number"
    )

    if numeric_data.shape[1] < 2:
        return (
            "Need at least 2 numerical columns "
            "for correlation heatmap"
        )

    correlation_matrix = numeric_data.corr()


    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    sns.heatmap(
        data=correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        ax=ax
    )

    ax.set_title("Correlation Matrix")

    ax.tick_params(
        axis="x",
        rotation=45
    )

    ax.tick_params(
        axis="y",
        rotation=0
    )

    fig.tight_layout()

    return fig