import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="YojanaSetu",
    page_icon="🏛️",
    layout="wide"
)

# ---------------- DATABASE CONNECTION ---------------- #

conn = sqlite3.connect("schemes.db")

df = pd.read_sql("SELECT * FROM schemes", conn)

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("🔍 Filters")

category = st.sidebar.selectbox(
    "Select Category",
    ["All"] + sorted(df["category"].dropna().unique().tolist())
)

scheme_type = st.sidebar.selectbox(
    "State or Central",
    ["All"] + sorted(df["state_or_central"].dropna().unique().tolist())
)

search = st.sidebar.text_input(
    "Search Scheme",
    placeholder="e.g. PM Kisan, Scholarship..."
)

# ---------------- FILTERING ---------------- #

filtered_df = df.copy()

if category != "All":
    filtered_df = filtered_df[
        filtered_df["category"] == category
    ]


if scheme_type != "All":
    filtered_df = filtered_df[
        filtered_df["state_or_central"] == scheme_type
    ]

if search:
    filtered_df = filtered_df[
        filtered_df["scheme_name"].str.contains(
            search,
            case=False,
            na=False
        )
    ]

# ---------------- HEADER ---------------- #

st.title("🏛️ YojanaSetu")

st.caption("Government Scheme Discovery & Analytics Platform")

st.markdown("""
Discover Indian government schemes, scholarships, and welfare programs through
interactive analytics and intelligent filtering.
""")

# ---------------- METRICS ---------------- #

col1, col2, col3 = st.columns(3)

col1.metric("Total Schemes", len(df))

col2.metric(
    "Categories",
    df["category"].nunique()
)

col3.metric(
    "Filtered Schemes",
    len(filtered_df)
)

st.divider()

# ---------------- SEARCH RESULTS ---------------- #

st.subheader("📋 Filtered Schemes")

if len(filtered_df) > 0:

    for index, row in filtered_df.head(20).iterrows():

        with st.container():

            st.markdown(f"## {row['scheme_name']}")

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Category:** {row['category']}")
                st.write(f"**Type:** {row['state_or_central']}")
                st.write(f"**Launch Year:** {row['launched_year']}")

            with col2:
                st.write(f"**Type:** {row['state_or_central']}")
                st.write(f"**Launch Year:** {row['launched_year']}")

            st.write(f"**Description:** {row['answer_english']}")

            st.markdown(
                f"[🌐 Official Website]({row['official_website']})"
            )

            st.divider()

else:
    st.warning("No schemes found.")

# ---------------- ANALYTICS SECTION ---------------- #

st.header("📊 Analytics Dashboard")

# ---------------- CATEGORY CHART ---------------- #

category_counts = (
    df["category"]
    .value_counts()
    .head(10)
)

fig1 = px.bar(
    x=category_counts.index,
    y=category_counts.values,
    title="Top Scheme Categories",
    labels={
        "x": "Category",
        "y": "Number of Schemes"
    }
)

st.plotly_chart(fig1, use_container_width=True)

# ---------------- STATE VS CENTRAL ---------------- #

scheme_counts = df["state_or_central"].value_counts()

fig2 = px.pie(
    values=scheme_counts.values,
    names=scheme_counts.index,
    title="State vs Central Schemes"
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------- BENEFICIARY ANALYSIS ---------------- #

category_state = (
    df.groupby(["category", "state_or_central"])
    .size()
    .reset_index(name="count")
)

fig3 = px.bar(
    category_state,
    x="category",
    y="count",
    color="state_or_central",
    title="Category-wise State vs Central Schemes"
)

st.plotly_chart(fig3, use_container_width=True)

# ---------------- YEARLY TRENDS ---------------- #

year_counts = (
    df["launched_year"]
    .value_counts()
    .sort_index()
)

fig4 = px.line(
    x=year_counts.index,
    y=year_counts.values,
    title="Scheme Launch Trends Over Years",
    labels={
        "x": "Year",
        "y": "Number of Schemes"
    }
)

st.plotly_chart(fig4, use_container_width=True)

# ---------------- FOOTER ---------------- #

st.divider()

st.caption(
    "Built using Streamlit, SQLite, Python, and Government Scheme Datasets"
)