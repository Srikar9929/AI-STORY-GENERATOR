import streamlit as st
from transformers import pipeline, set_seed

st.title("AI Story Generator (Local GPT-2)")

@st.cache_resource
def load_generator():
    return pipeline('text-generation', model='gpt2')

generator = load_generator()

prompt = st.text_area("Enter your story prompt:", "")

if st.button("Generate Story"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating story..."):
            set_seed(42)
            # Generate a longer story
            result = generator(prompt, max_length=250, num_return_sequences=1, do_sample=True, top_k=50)
            story = result[0]['generated_text']
            st.markdown("### Generated Story")
            st.write(story)
