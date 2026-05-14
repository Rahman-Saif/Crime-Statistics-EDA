import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import zscore

# Load data (adjust path if needed)
data = pd.read_csv("Crime Statistics Of Bangladesh 2010-2019.csv")

# Define crime columns (from your EDA)
crime_columns = ['Dacoity', 'Robbery', 'Murder', 'Speedy Trial', 'Riot',
                 'Woman & Child Repression', 'Kidnapping', 'Police Assault',
                 'Burglary', 'Theft', 'Other Cases', 'Arms Act', 'Explosive',
                 'Narcotics', 'Smuggling']

# App title
st.title("Crime Statistics Dashboard: Bangladesh (2010-2019)")
st.markdown("Interactive EDA for crime trends, geography, and correlations.")


# Sidebar filters
st.sidebar.header("Filters")
selected_years = st.sidebar.multiselect("Select Years", options=data['Year'].unique(), default=data['Year'].unique())
selected_units = st.sidebar.multiselect("Select Units", options=data['Unit Name'].unique(), default=data['Unit Name'].unique())
selected_crime = st.sidebar.selectbox("Select Crime Type for Focus", options=crime_columns + ['Total Cases'])

# Filter data
filtered_data = data[(data['Year'].isin(selected_years)) & (data['Unit Name'].isin(selected_units))]

# Tabs for main sections
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Univariate", "Time-Series", "Geography", "Correlations", "Outliers"])

with tab1:
    st.header("Data Overview")
    st.write(f"Rows: {filtered_data.shape[0]}, Columns: {filtered_data.shape[1]}")
    st.dataframe(filtered_data.head(10))
    st.subheader("Descriptive Stats")
    st.dataframe(filtered_data[crime_columns + ['Total Cases']].describe())
    st.metric("Total Cases (Filtered)", filtered_data['Total Cases'].sum())

with tab2:
    st.header("Univariate Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(filtered_data, x=selected_crime, title=f"Distribution of {selected_crime}")
        st.plotly_chart(fig)
    with col2:
        fig = px.box(filtered_data, y=selected_crime, title=f"Box Plot of {selected_crime}")
        st.plotly_chart(fig)
    st.subheader("Categorical Counts")
    fig = px.bar(filtered_data['Year'].value_counts().reset_index(), x='Year', y='count', title="Crime Reports by Year")
    st.plotly_chart(fig)
    fig = px.bar(filtered_data['Unit Name'].value_counts().reset_index(), x='count', y='Unit Name', orientation='h', title="Crime Reports by Unit")
    st.plotly_chart(fig)

with tab3:
    st.header("Time-Series Trends")
    yearly_data = filtered_data.groupby('Year')[crime_columns + ['Total Cases']].sum().reset_index()
    fig = px.line(yearly_data, x='Year', y='Total Cases', title="Total Cases Over Years")
    st.plotly_chart(fig)
    fig = px.line(yearly_data, x='Year', y=selected_crime, title=f"{selected_crime} Trend")
    st.plotly_chart(fig)

with tab4:
    st.header("Geographical Analysis")
    unit_data = filtered_data.groupby('Unit Name')[crime_columns + ['Total Cases']].sum().reset_index()
    fig = px.bar(unit_data, x='Total Cases', y='Unit Name', orientation='h', title="Total Cases by Unit")
    st.plotly_chart(fig)
    fig = px.bar(unit_data.sort_values(selected_crime, ascending=False), x=selected_crime, y='Unit Name', orientation='h', title=f"{selected_crime} by Unit")
    st.plotly_chart(fig)

with tab5:
    st.header("Correlations")
    corr = filtered_data[crime_columns].corr()
    fig = px.imshow(corr, text_auto=True, title="Crime Type Correlations")
    st.plotly_chart(fig)

with tab6:
    st.header("Outlier Detection")
    filtered_data[f'{selected_crime}_zscore'] = np.abs(zscore(filtered_data[selected_crime]))
    outliers = filtered_data[filtered_data[f'{selected_crime}_zscore'] > 3]
    st.write(f"Outliers in {selected_crime}: {len(outliers)} found")
    st.dataframe(outliers[['Unit Name', 'Year', selected_crime]])
    fig = px.box(filtered_data, y=selected_crime, title=f"Box Plot with Outliers for {selected_crime}")
    st.plotly_chart(fig)

