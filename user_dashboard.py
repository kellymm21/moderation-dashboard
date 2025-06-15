
import streamlit as st
import pandas as pd

st.set_page_config(page_title="User Moderation Dashboard", layout="wide")

df = pd.read_csv("user_latest_actions.csv", parse_dates=["signup_ts", "action_ts"])

st.title("🛡️ User Moderation Dashboard")
st.markdown("Filter and review user profiles alongside their most recent moderation actions.")

st.sidebar.header("Filter Options")
country_filter = st.sidebar.multiselect("Country", sorted(df["country"].dropna().unique()))
region_filter = st.sidebar.multiselect("Region", sorted(df["region"].dropna().unique()))
action_filter = st.sidebar.multiselect("Action Type", sorted(df["action_type"].dropna().unique()))
reason_filter = st.sidebar.multiselect("Reason", sorted(df["reason"].dropna().unique()))

filtered = df.copy()
if country_filter:
    filtered = filtered[filtered["country"].isin(country_filter)]
if region_filter:
    filtered = filtered[filtered["region"].isin(region_filter)]
if action_filter:
    filtered = filtered[filtered["action_type"].isin(action_filter)]
if reason_filter:
    filtered = filtered[filtered["reason"].isin(reason_filter)]

st.write(f"### Results: {len(filtered)} users found")

for _, row in filtered.iterrows():
    with st.container():
        cols = st.columns([1, 4])
        with cols[0]:
            st.image(row["profile_photo"], width=100)
        with cols[1]:
            st.markdown(f"**User ID:** `{row['user_id']}`")
            st.markdown(f"**Handle:** `{row['handle']}`")
            st.markdown(f"**Country / Region:** {row['country']} / {row['region']}")
            st.markdown(f"**Action:** {row['action_type']} - {row['reason']}  
**Timestamp:** {row['action_ts']}")
            st.markdown(f"[🔗 View Profile]({row['profile_url']})", unsafe_allow_html=True)

st.download_button(
    "⬇️ Download Filtered CSV",
    filtered.to_csv(index=False).encode("utf-8"),
    "filtered_user_latest_actions.csv",
    "text/csv"
)
