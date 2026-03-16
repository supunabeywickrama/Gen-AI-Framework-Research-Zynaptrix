import os
import sys
import streamlit as st

# Add the project root to sys.path so we can import from the copilot module
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from copilot.industrial_copilot import ask_copilot

# Configure Streamlit page layout and title
st.set_page_config(
    page_title="Industrial Manual Copilot",
    page_icon="🏭",
    layout="wide"
)

# Sidebar styling and info
st.sidebar.title("🏭 Manual Copilot")
st.sidebar.write("Industrial Knowledge Assistant")
st.sidebar.info(
    "This system features:\n"
    "✅ Layout detection (YOLO)\n"
    "✅ OCR extraction\n"
    "✅ Sliding-window chunking\n"
    "✅ OpenAI Embeddings\n"
    "✅ pgvector Similarity Search\n"
    "✅ RAG NLP via GPT-4o-mini"
)

st.sidebar.markdown("---")

# File uploader placeholder for future extensibility
st.sidebar.subheader("Document Control")
uploaded_file = st.sidebar.file_uploader("Upload Manual (Future Feature)", type=["pdf"])
if uploaded_file:
    st.sidebar.warning("Future Feature: Uploaded file will trigger the entire parsing pipeline.")

# Main content
st.title("Industrial Manual Copilot")
st.markdown("Ask natural language questions about your machine manuals.")

st.markdown("---")

question = st.text_input("Ask a question about your manual:", placeholder="e.g. How to replace the cooling fan?")

if question:
    with st.spinner("Copilot is thinking... retrieving context..."):
        try:
            answer, images = ask_copilot(question)
            
            st.subheader("AI Answer")
            st.info(answer)

            if images:
                st.subheader("Related Figures & Tables")
                
                # Display multiple diagrams neatly
                cols = st.columns(len(images))
                for idx, img_path in enumerate(images):
                    if os.path.exists(img_path):
                        with cols[idx]:
                            # Prettify the visual caption (remove confusing .png and underscores)
                            nice_name = os.path.basename(img_path).replace(".png", "").replace("_", " ").title()
                            st.image(img_path, caption=f"Matched Region: {nice_name}", use_container_width=True)
                    else:
                        st.error(f"Image not found at path: {img_path}")
                        
        except Exception as e:
            st.error(f"Error executing copilot query: {e}")
