import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from model_methods import predict
classes = {0:'tidak delay',1:'delay kurang dari sama dengan 4 jam',2:'delay > 4 jam'}
class_labels = list(classes.values())
st.title("Prediksi Delay Departure Kapal")
def predict_class():
    data = list(map([DERMAGA, JENIS_KAPAL, PALKA, JUMLAH_CC, DISCHARGE, LOADING, BD, SHIFTING_YARD, WAG, BAD_WEATHER]))
    result, probs = predict(data)
    st.write("The predicted class is ",result)
    probs = [np.round(x,6) for x in probs]
    ax = sns.barplot(probs ,class_labels, palette="winter", orient='h')
    ax.set_yticklabels(class_labels,rotation=0)
    plt.title("Probabilities of the Data belonging to each class")
    for index, value in enumerate(probs):
        plt.text(value, index,str(value))
    st.pyplot()
st.markdown("**Masukkan Karakteristik Kapal yang akan Diprediksi**")
DERMAGA = st.selectbox("Dermaga Sandar:", ("1", "2", "3", "4"))
JENIS_KAPAL = st.selectbox("Jenis Kapal:", ("Feeder", "Direct"))
PALKA = st.number_input("Jumlah Palka (Unit):")
JUMLAH_CC = st.number_input("Jumlah CC (Unit):")
DISCHARGE = st.number_input("Jumlah Petikemas yang Dibongkar (Box):")
LOADING = st.number_input("Jumlah Petikemas yang Dimuat (Box):")
BD = st.number_input("Lama Waktu Breakdown (Menit):")
SHIFTING_YARD = st.number_input("Jumlah Shifting Yard (Box):")
WAG = st.number_input("Lama Waktu WAG (Menit):")
BAD_WEATHER = st.number_input("Lama Waktu Bad Weather (Menit):")
submit = st.button('Predict')
if submit:
    if JENIS_KAPAL == "Direct":
      JENIS_KAPAL = 1
    else:
      JENIS_KAPAL = 0
    if DERMAGA == "1":
      DERMAGA = 1
    elif DERMAGA == "2":
      DERMAGA = 2
    elif DERMAGA == "3":
      DERMAGA = 3
    else:
      DERMAGA = 4
    predict_class()