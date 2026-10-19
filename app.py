import streamlit as st
import joblib
from preprocessor import clean_headline

@st.cache_resource
def load():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('clickbait_model.pkl')
    return model, vectorizer

model, vectorizer = load()

st.title("CLICKBAIT OR LEGITIMATE")
st.write("Enter a headline below to check if it's clickbait or legitimate news:")

user_input = st.text_input("Please Write Your Headline Here", " ")

if st.button("Enter"):
    if user_input.strip() == "":
        st.warning("Please enter a headline first:")
    else:
        cleaned_txt = clean_headline(user_input)
        text_vectorized = vectorizer.transform([cleaned_txt])
        prediction = model.predict(text_vectorized)[0]

        if prediction == 1:
            st.error("CLICKBAIT")

        else:
            st.success("LEGITIMATE")
