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

@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove final layer
    model.eval()
    return model

@st.cache_data
def load_user_data():
    return pd.read_csv("user_latest_actions.csv")

@st.cache_data
def compute_embeddings(df, model, transform):
    embeddings = []
    for url in df["profile_photo"]:
        try:
            response = requests.get(url, timeout=5)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                emb = model(img_tensor).squeeze().numpy()
            embeddings.append(emb)
        except:
            embeddings.append(np.zeros(512))
    return np.stack(embeddings)

def main():
    st.title("🔍 Image Similarity Search")
    model = load_model()
    df = load_user_data()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    st.write("Upload a profile image to find visually similar users:")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=False)

        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            query_embedding = model(img_tensor).squeeze().numpy()

        st.info("Calculating similarities...")

        embeddings = compute_embeddings(df, model, transform)
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        df["similarity"] = similarities
        top_matches = df.sort_values("similarity", ascending=False).head(5)

        st.subheader("Top 5 Most Similar Profiles")
        for _, row in top_matches.iterrows():
            st.image(row["profile_photo"], width=100)
            st.markdown(f"**User ID:** `{row['user_id']}`")
            st.markdown(f"**Handle:** `{row['handle']}`")
            st.markdown(f"**Similarity Score:** {row['similarity']:.4f}")
            st.markdown(f"[🔗 View Profile]({row['profile_url']})", unsafe_allow_html=True)
            st.markdown("---")

if __name__ == "__main__":
    main()