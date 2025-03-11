#import important package
import streamlit as st
import pandas as pd
import numpy as np
import pickle

#load dataset
data = pd.read_csv("traineddata.csv")
file = open("pipe.pkl", "rb")
pipe = pickle.load(file)
file.close()
st.title("laptop Price predictor")

#take user input
#brand
company = st.selectbox('Brand', data['Company'].unique())

#laptop_type
laptop_type = st.selectbox('Type', data['TypeName'].unique())

#ram
ram = st.selectbox('RAM (in GB)', [2,4,6,8,12,16,24,32,64])

#operating system
os = st.selectbox('OS', data['OpSys'].unique())

#weight
weight = st.number_input("Weight of the laptop")

#touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

#ips
ips = st.selectbox('IPS', ['No', 'Yes'])

#screen size
screen_size = st.number_input('Screen Size:')

#resolution
resolution = st.selectbox('Scrren Resolution', ['1920x1080','1366x900','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU', data['CPU_Name'].unique())

#hdd
hdd = st.selectbox('HDD (in GB)', [0,128,256,512,1024,2048])

#ssd
ssd = st.selectbox('SSD (in GB)', [0,8,128,256,512,1024])

#gpu
gpu = st.selectbox('GPU', data['Gpu_Brand'].unique())

#prediction
if st.button('Predict Price'):
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0
    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x'[0]))
    Y_res = int(resolution.split('x'[1]))

    ppi = ((X_res**2)+(Y_res**2))**0.5/(screen_size)

    query = np.array([company, laptop_type, ram, os, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu])

    prediction = int(np.exp(pipe.predict(query)[0]))

    st.title("Prediction price of the configuration laptop is around" + str(prediction))