import streamlit as st
import requests

st.set_page_config(page_title="Shoplifting Detection", layout="centered")

st.title("Shoplifting Detection System")
st.write("Upload a video to detect suspicious activity")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_file:
    st.video(uploaded_file)

    if st.button("Predict"):
        with st.spinner("Analyzing..."):

            files = {"file": uploaded_file}

            response = requests.post(
                "http://127.0.0.1:8000/api/predict/",
                files=files
            )

            if response.status_code == 200:
                data = response.json()

                if data["label"] == "Shoplifting":
                    st.error(f" {data['label']} ({data['prediction']:.2f})")
                else:
                    st.success(f"{data['label']} ({data['prediction']:.2f})")

            else:
                st.error("Error connecting to backend")