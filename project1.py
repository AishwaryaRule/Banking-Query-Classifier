import streamlit as st
import numpy as np
import pandas as pd
import nltk
import re
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

st.set_page_config(page_title="Banking chatbot",layout="wide")

st.subheader("Hi, I am here to help !")
st.title("Enter your query:")
user_input = st.text_input("Write your query here", " ")
submit=st.button("Submit")

if submit:
    user_input=user_input.strip().lower()
    if user_input in ["hi","hello","whats'up","hiii","hellooooo","hello there"]:
        final_output="Hi user, how may I help you?"
    elif user_input in ["bye","byee"]:
        final_output="Ok bye, let me know if you have any query!"
    else:
        stemmer=PorterStemmer()

        sentence=user_input.lower()                            #converting to lower case
        sentence =re.sub(r'[^\w\s]', '', sentence)              #removing punctuations,links,special characters
        sentence = re.sub(r"https\S+|www\S+https\S+", '',sentence, flags=re.MULTILINE)
        sentence = re.sub(r'\@w+|\#','',sentence)
        words=word_tokenize(sentence)                              #tokenization
        words=[stemmer.stem(word) for word in words if word not in stopwords.words("english")]           #removing stop-words and stemming
        processed=" ".join(words)

        vectorizer1=pickle.load(open('Vectorizer.pkl','rb'))
        question_vectors=pickle.load(open('Question_vectors.pkl','rb'))
        df=pickle.load(open('BANK_data.pkl','rb'))

        test_vector=vectorizer1.transform([processed]).toarray()

        cosine_sim=cosine_similarity(question_vectors,test_vector)
        most_sim_idx=np.argmax(cosine_sim)

        final_output=df.iloc[most_sim_idx]['Answer']

    st.write(final_output)