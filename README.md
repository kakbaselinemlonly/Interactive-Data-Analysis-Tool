# Interactive Data Analysis Tool

An interactive Streamlit application for exploring, analyzing, visualizing, and querying CSV datasets.

interactive_data_analysis_tool/
├── app.py

├── analysis_functions.py

├── requirements.txt

├── README.md

└── .streamlit/

    └── config.toml

## Features

### Exploratory Data Analysis
- Dataset overview
- Descriptive statistics
- Categorical summaries
- Group summaries
- Correlation matrix

### Statistical Analysis
- Independent Welch's t-test
- One-way ANOVA
- Pearson correlation
- Simple OLS regression

### Data Visualization
- Histogram
- Bar chart
- Box plot
- Scatter plot
- Correlation heatmap

### SQL Query
- Execute read-only SQLite `SELECT` and `WITH` queries
- Uploaded data is available as a table named `data`

## Technologies

- Python
- Pandas
- NumPy
- SciPy
- Statsmodels
- Matplotlib
- Seaborn
- SQLite
- Streamlit

## Notes
The application accepts CSV files.
SQL queries use SQLite syntax.
Only SELECT and WITH SQL queries are supported.
Large datasets may require additional memory and processing time.

## Run Locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```markdown


## Live Application
