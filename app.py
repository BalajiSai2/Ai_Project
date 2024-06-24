import pandas as pd
import numpy as np
import streamlit as st
import pickle as pk

model =pk.load(open('model.pkl','rb'))

import streamlit as st

# Set custom styles using st.markdown() to modify the body background color and header color
st.markdown(
    """
    <style>
.st-emotion-cache-1r4qj8v{
        background-color: rgb(212 160 160); /* Set body background color to green */
        color: blue; /* Set text color to blue */
        font-size: 24px; /* Set font size to 24 pixels */
        font-family: Arial, sans-serif; /* Set font family */
        padding: 20px; /* Add padding around content */
    }
     h1 {
        font-family: 'Roboto', sans-serif; /* Set Roboto for h1 (heading level 1) */
        font-size: 36px; /* Set font size for h1 */
        font-weight: bold; /* Set font weight to bold */
       
    }
    .st-eg {
        background-color: #f0f0f0; /* Set background color of the slider */
        color: #007bff; /* Set text color of the slider */
    }
    .st-emotion-cache-l9bjmx p {
        color: #012d01;
    font-size: 17px;
     }
     .st-emotion-cache-10y5sf6 {
    color: rgb(65 3 32);}
    .st-cx {
    background: linear-gradient(to right, rgb(111 226 7) 0%, rgb(9 225 225) 0%, rgb(13 77 201 / 25%) 0%, rgb(12 87 231 / 25%) 100%);
    }
    .st-emotion-cache-1r4qj8v {
    color: #090976;
     }
     .st-cz {
    margin-left: 5px;
    margin-right: 5px;
     }
     .st-emotion-cache-1vzeuhh {
    background-color: rgb(131 60 9);
    margin-left: 7px;
  }
  .st-emotion-cache-1inwz65 {
    margin-left: 2px;}
   
    </style>
    """
    , unsafe_allow_html=True
)

# Display the header using st.markdown() with custom styles
st.markdown("<h1 >Car Price Prediction ML Model</h1>", unsafe_allow_html=True)
cars_data =pd.read_csv('Cardetails.csv')
cars_data.dropna(inplace=True)
cars_data.drop_duplicates(inplace=True)
def get_brand_name(car_name):
    car_name=car_name.split(' ')[0]
    return car_name.strip()

cars_data['name']=cars_data['name'].apply(get_brand_name)
name=st.selectbox('Select Car Brand',cars_data['name'].unique())
year=st.slider('Car Manufactured Year',1994,2024)
km_driven=st.slider('No of km driven',11,200000)
fuel=st.selectbox('Fuel type',cars_data['fuel'].unique())
seller_type=st.selectbox('Seller type',cars_data['seller_type'].unique())
transmission=st.selectbox('Transmission type',cars_data['transmission'].unique())
owner=st.selectbox('Owner ',cars_data['owner'].unique())
mileage=st.slider('Car Mileage',10,40)
engine=st.slider('Engine cc',700,5000)
max_power=st.slider('Max Power',0,200)
seats=st.slider('No of seats',5,10)

if st.button("Predict"):
    input_data_model=pd.DataFrame([[name,year,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,seats]],
    columns=['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats'])
    
    st.write(input_data_model)
    
    input_data_model['name'].replace(['Maruti','Skoda','Honda','Hyundai','Toyota','Ford','Renault','Mahindra','Tata','Chevrolet','Datsun','Jeep','Mercedes-Benz','Mitsubishi','Audi','Volkswagen','BMW','Nissan','Lexus','Jaguar','Land','MG','Volvo','Daewoo'
    ,'Kia','Fiat','Force','Ambassador','Ashok','Isuzu','Opel',],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],inplace=True)
    input_data_model['transmission'].replace(['Manual' ,'Automatic'],[1,2],inplace=True)
    input_data_model['seller_type'].replace(['Individual', 'Dealer' ,'Trustmark Dealer'],[1,2,3],inplace=True)
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'],[1,2,3,4],inplace=True)
    input_data_model['owner'].replace(['First Owner' ,'Second Owner' ,'Third Owner' ,'Fourth & Above Owner',
    'Test Drive Car'],[1,2,3,4,5],inplace=True)
    

    car_price=model.predict(input_data_model)
    if(car_price[0]>0):
     st.markdown('Car Price is going to be '+str(car_price[0]))
    else:
       st.markdown('<h5 style="color:red;">Invalid details!!</h5>', unsafe_allow_html=True)
