import numpy as np
import pickle
import streamlit as st

#with open("logistic_regression.pkl", "rb") as model_file:
 #   lr = pickle.load(model_file)

with open("naive_bayes.pkl", "rb") as pickle_in:
    classifier = pickle.load(pickle_in)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open("label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

def map_prediction(prediction):
    return label_encoder.inverse_transform(prediction)[0]

def resume_classification(resume_description):
    resume_vector = vectorizer.transform([resume_description])

    prediction = classifier.predict(resume_vector)
    
    return map_prediction(prediction)

def main():
    st.title("Classifying resumes into categories based on JD")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Resume Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    jd = st.text_input("Resume Description")
    
    if st.button("Predict"):
        if jd.strip(): 
            result = resume_classification(jd)
            st.success('The category is {}'.format(result))
        else:
            st.error("Please enter a resume description.")

if __name__ == '__main__':
    main()
