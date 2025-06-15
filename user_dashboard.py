import streamlit as st
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------- CONFIG ----------
st.set_page_config(page_title="User Moderation Dashboard", layout="wide")

# ---------- LOAD ----------
@st.cache_data
def load_user_data():
    return pd.read_csv("user_latest_actions.csv", parse_dates=["signup_ts", "action_ts"])

@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    return model

@st.cache_resource
def compute_embeddings(df, _model, _transform):
    embeddings = []
    for url in df["profile_photo"]:
        try:
            response = requests.get(url, timeout=5)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img_tensor = _transform(img).unsqueeze(0)
            with torch.no_grad():
                emb = _model(img_tensor).squeeze().numpy()
            embeddings.append(emb)
        except:
            embeddings.append(np.zeros(512))
    return np.stack(embeddings)

# ---------- MAIN ----------
df = load_user_data()

tab1, tab2 = st.tabs(["📊 Filter Users", "🧠 Image Similarity Search"])

with tab1:
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
                st.markdown(
                    f"**Action:** {row['action_type']} - {row['reason']}  \n"
                    f"**Timestamp:** {row['action_ts']}"
                )
                st.markdown(f"[🔗 View Profile]({row['profile_url']})", unsafe_allow_html=True)

    st.download_button(
        "⬇️ Download Filtered CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        "filtered_user_latest_actions.csv",
        "text/csv"
    )

with tab2:
    st.title("🔍 Find Similar Profiles by Image")
    model = load_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    uploaded_file = st.file_uploader("Upload a profile image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=False)

        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            query_embedding = model(img_tensor).squeeze().numpy()

        st.info("Calculating image similarity...")
        embeddings = compute_embeddings(df, model, transform)
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        df["similarity"] = similarities
        top_matches = df.sort_values("similarity", ascending=False).head(5)

        st.subheader("Top 5 Most Similar Profiles")
        for _, row in top_matches.iterrows():
            with st.container():
                cols = st.columns([1, 4])
                with cols[0]:
                    st.image(row["profile_photo"], width=100)
                with cols[1]:
                    st.markdown(f"**User ID:** `{row['user_id']}`")
                    st.markdown(f"**Handle:** `{row['handle']}`")
                    st.markdown(f"**Similarity Score:** `{row['similarity']:.4f}`")
                    st.markdown(f"[🔗 View Profile]({row['profile_url']})", unsafe_allow_html=True)