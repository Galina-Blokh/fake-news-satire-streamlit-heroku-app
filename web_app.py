import pickle
import pandas as pd
import numpy as np
from pattern.text import Sentence
from pattern.text.en import sentiment, parse, modality
from transformers import AutoTokenizer, AutoModel
import torch

MODEL_PATH = 'finalized_model.pkl'
import streamlit as st


# Mean Pooling - Take attention mask into account for correct averaging
@st.cache(suppress_st_warning=False)
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


@st.cache(suppress_st_warning=False)
def load_pipeline(sentences):
    """
    Load the Text Processing and Classifier Pipeline
    """
    # Tokenize sentences
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/ce-ms-marco-TinyBERT-L-4")
    model = AutoModel.from_pretrained("sentence-transformers/ce-ms-marco-TinyBERT-L-4")

    encoded_input = tokenizer(sentences.to_list(), padding=True, truncation=True, max_length=128, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Perform pooling. In this case, mean pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    sentiment_train = sentences.apply(lambda x: sentiment(x))
    sentiment_train = pd.DataFrame(sentiment_train.values.tolist(),
                                   columns=['polarity', 'subjectivity'],
                                   index=sentences.index)
    parse_s = sentences.apply(lambda x: parse(x, lemmata=True))
    sent = parse_s.apply(lambda x: Sentence(x))
    modality_s = pd.DataFrame(sent.apply(lambda x: modality(x)))

    meta_df = sentiment_train.merge(modality_s, left_index=True, right_index=True)
    input_matrix = pd.concat([meta_df.reset_index(drop=True), pd.DataFrame(sentence_embeddings)], axis=1)

    with open(MODEL_PATH, 'rb') as file:
        unpickler = pickle.Unpickler(file)
        model_p = unpickler.load()
        class_text = model_p.predict(input_matrix)
        prediction = model_p.predict_proba(input_matrix)
        if class_text == 0:
            my_prediction = "Fake News"
        else:
            my_prediction = "Satire"
        return my_prediction, prediction


st.title('Fake news or Satire. Classificator')
news_story = st.text_area('Enter the Text', height=200)
if st.button('Submit'):
    class_text, probability = load_pipeline(pd.Series(news_story))
    st.write('Your text seems to be a ', class_text, 'with a ',
             round(np.max(probability) * 100, 2), '% probability.')
