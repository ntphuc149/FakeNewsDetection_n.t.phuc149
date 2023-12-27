import pandas as pd
import re
import nltk
import ftfy
import pickle
import streamlit as st
import plotly.express as px
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.title('Detecting fake news with machine learning classification models')
content = st.text_area(label='Nội dung tin tức', placeholder='Nhập nội dung tin tức cần xác thực tại đây', height=200)
btn_verify = st.button('Xác thực')

vectorizer = pickle.load(open('./MODELS/vectorizer.pkl', 'rb'))
log_reg_pretrain = pickle.load(open('./MODELS/LogisticRegression.pkl', 'rb'))
rfc_pretrain = pickle.load(open('./MODELS/RandomForestClassifier.pkl', 'rb'))
xgbc_pretrain = pickle.load(open('./MODELS/XGBoost.pkl', 'rb'))

def restore_text(content):
    fixed_text = ftfy.fix_text(str(content))
    return fixed_text

def remove_links(text):
    link_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return re.sub(link_pattern, '', text)

def final_cleaning(content):
    lemmatizer = WordNetLemmatizer()
    cleaned_content = re.sub('[^a-zA-Z]', ' ', content)
    cleaned_content = cleaned_content.lower()
    tokenized_content = cleaned_content.split()
    lemmatized_content = [lemmatizer.lemmatize(word) for word in tokenized_content if not word in stopwords.words('english')]
    lemmatized_content = ' '.join(lemmatized_content)
    lemmatized_content = re.sub(' +', ' ', lemmatized_content)
    return lemmatized_content

def fake_news_predictor(content):
    fixed_content =restore_text(content)
    fixed_content_no_link = remove_links(fixed_content)
    cleean_data = final_cleaning(fixed_content_no_link)
    input_data = [cleean_data]
    vector_form_raw = vectorizer.transform(input_data)

    log_reg_pred = log_reg_pretrain.predict(vector_form_raw)
    log_reg_pred_proba = log_reg_pretrain.predict_proba(vector_form_raw)

    rfc_pred = rfc_pretrain.predict(vector_form_raw)
    rfc_pred_proba = rfc_pretrain.predict_proba(vector_form_raw)

    xgbc_pred = xgbc_pretrain.predict(vector_form_raw)
    xgbc_pred_proba = xgbc_pretrain.predict_proba(vector_form_raw)

    return log_reg_pred, log_reg_pred_proba, rfc_pred, rfc_pred_proba, xgbc_pred, xgbc_pred_proba

if btn_verify:
    if content != '':
        log_reg_pred, log_reg_pred_proba, rfc_pred, rfc_pred_proba, xgbc_pred, xgbc_pred_proba = fake_news_predictor(content)

        # st.progress(value=round(((log_reg_pred_proba[0][0] + rfc_pred_proba[0][0] + xgbc_pred_proba[0][0])/3), 3),
        #             text=f'Real News probability: {round(((log_reg_pred_proba[0][0] + rfc_pred_proba[0][0] + xgbc_pred_proba[0][0])/3)*100, 3)}')
        st.progress(value=round(((log_reg_pred_proba[0][1] + xgbc_pred_proba[0][1])/3), 2),
                    text=f'Fake News probability: {round(((log_reg_pred_proba[0][1] + xgbc_pred_proba[0][1])/2)*100, 3)}')

        if log_reg_pred[0] == 0:
            st.success(f'Logistic Regression: Tin chuẩn em êiiiii! Chuẩn {round(log_reg_pred_proba[0][0]*100, 2)}%')
        else:
            st.error(f'Logistic Regression: Tin không chuẩn em êiiiii! Không chuẩn {round(log_reg_pred_proba[0][1]*100, 2)}%')
        # if rfc_pred[0] == 0:
        #     st.success(f'Random Forest: Tin chuẩn em êiiiii! Chuẩn {round(rfc_pred_proba[0][0]*100, 2)}%')
        # else:
        #     st.error(f'Random Forest: Tin không chuẩn em êiiiii! Không chuẩn {round(rfc_pred_proba[0][1]*100, 2)}%')
        if xgbc_pred[0] == 0:
            st.success(f'XGBoost: Tin chuẩn em êiiiii! Chuẩn {round(xgbc_pred_proba[0][0]*100, 2)}%')
        else:
            st.error(f'XGBoost: Tin không chuẩn em êiiiii! Không chuẩn {round(xgbc_pred_proba[0][1]*100, 2)}%')
    else:
        st.warning('Thêm tin tức và nhấn "Xác thực" để anh giúp em kiểm tra xem tin này có chuẩn hay không nhé!')

st.subheader('Our models performance on test set')
# st.selectbox(label='Metric', options=['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC', 'All metric'], index=5)

metrics = pd.read_csv(filepath_or_buffer='./Dataset/model_evaluations.csv', sep=',')

fig = px.bar(metrics, ['Logistic Regression', 'Random Forest', 'XGBoost'], ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC-ROC'],
             title='Model Performance Metrics',
             labels={'value': 'Score', 'variable': 'Metric'},
             barmode='group')

st.plotly_chart(fig, use_container_width=True)
