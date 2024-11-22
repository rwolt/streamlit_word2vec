import streamlit as st
from gensim.models import Word2Vec
import os
import pandas as pd

@st.cache_resource
def load_model(model_path):
    """Load the trained Word2Vec model from the file."""
    return Word2Vec.load(model_path)

def get_pooled_embedding(model, phrases):
    """Generate pooled embedding from a list of phrases."""
    embeddings = []
    for phrase in phrases:
        words = phrase.split()
        phrase_embedding = []
        for word in words:
            if word in model.wv:
                phrase_embedding.append(model.wv[word])
        if phrase_embedding:
            embeddings.append(sum(phrase_embedding) / len(phrase_embedding))

    if embeddings:
        pooled_embedding = sum(embeddings) / len(embeddings)
        return pooled_embedding
    else:
        return None

def find_similar_dissimilar_words(model, embedding, topn=30):
    """Find the most conceptually similar and opposite words."""
    try:
        similar_words = model.wv.similar_by_vector(embedding, topn=topn)
        opposite_words = model.wv.similar_by_vector(-embedding, topn=topn)
        opposite_words = sorted(opposite_words, key=lambda x: x[1], reverse=True)

        # Generate prompts
        style_prompt = ", ".join([sim_word for sim_word, _ in similar_words])
        negative_prompt = ", ".join([opp_word for opp_word, _ in opposite_words])

        return style_prompt, negative_prompt, similar_words, opposite_words

    except KeyError:
        return None, None, [], []

def main():
    st.title("Music Style Prompt Generator for Text-to-Audio")
    st.write("Enter a style prompt to analyze related terms.")

    model_path = "./models/word2vec_popular_music_genres.model"
    if not os.path.exists(model_path):
        st.error("Model file not found. Please train the model first.")
        return

    model = load_model(model_path)

    input_text = st.text_input(
        "Enter a music style prompt (comma-separated for multiple terms):",
        placeholder="e.g., jazz, smooth, mellow"
    )

    if st.button("Analyze"):
        if input_text:
            phrases = input_text.split(", ")
            embedding = get_pooled_embedding(model, phrases)
            if embedding is not None:
                style_prompt, negative_prompt, similar_words, opposite_words = find_similar_dissimilar_words(model, embedding)
                
                if style_prompt and negative_prompt:
                    st.subheader("Style Prompt")
                    st.write(style_prompt)
                    
                    st.subheader("Negative Prompt")
                    st.write(negative_prompt)

                    # Display similar and dissimilar words in tables
                    st.subheader("Most Conceptually Similar and Opposite Words")
                    
                    similar_df = pd.DataFrame(similar_words, columns=["Similar Word", "Similarity Score"])
                    opposite_df = pd.DataFrame(opposite_words, columns=["Dissimilar Word", "Dissimilarity Score"])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.table(similar_df)
                    
                    with col2:
                        st.table(opposite_df)
                else:
                    st.warning("No valid embeddings found for the input phrases.")
            else:
                st.warning("No valid embeddings found for the input phrases.")
        else:
            st.warning("Please enter a prompt.")

if __name__ == "__main__":
    main()