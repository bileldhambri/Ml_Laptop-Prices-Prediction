import streamlit as st
import pickle
import numpy as np
#import xgboost
# import the model
model = pickle.load(open('pipe.pkl','rb'))
Data = pickle.load(open('Save_data.pkl','rb'))

st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand',Data['Company'].unique())

# type of laptop
type = st.selectbox('Type',Data['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
#ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.number_input('Screen Size')

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',Data['Cpu_Name'].unique())

#Cpu frequence
cpu_freq = st.selectbox('CPU fréquence',Data['cpu_freq(GHz)'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',Data['Gpu_brand'].unique())

os = st.selectbox('OS',Data['OpSys'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    #if ips == 'Yes':
    #    ips = 1
    #else:
    #    ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,type,ram,os,weight,touchscreen,ppi,hdd,ssd,gpu,cpu,cpu_freq])
    #query = np.array([16,3,8,2,2.00,0,100.454670,0,128,2,2])
    #query = np.array(["Toshiba","Notebook",8,"Windows",2.00	,1,3,0,0,"Intel","Intel Core i5",2.3])
    query = query.reshape(1,12)
    st.title("The predicted price of this configuration is " + str(int(np.exp(model.predict(query)[0]))))
    
#import sklearn   
#print('The scikit-learn version is {}.'.format(sklearn.__version__))
import xgboost as xgb
print(xgb.__version__)
import shap
print(shap.__version__)