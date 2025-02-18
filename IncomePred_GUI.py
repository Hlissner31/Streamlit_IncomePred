import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
from sklearn.preprocessing import LabelEncoder
import pickle

#load pickle file from disk
model = pickle.load(open('model_V2.pkl', 'rb'))

with open('encoder_V2.pkl', 'rb') as f:
    encoder_dict = pickle.load(f)

#encodeing the data using the encoder_dict
def encode_data(df, encoder_dict):
    category_col = ['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
    for col in category_col:
        if col in encoder_dict:
            le = LabelEncoder()
            le.classes_ = np.array(encoder_dict[col], dtype=object)

            #use transform method on unknown columns
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
            df[col] = le.transform(df[col])
    return df

# Streamlit App Title and Description
st.title('Income Prediction App')
st.write('This app predicts whether a person makes over 50K a year.')


def main():
    # Age field
    st.slider('How old are you?', 0, 90, key="age")

    # Working Class Selection
    st.selectbox('What is your working class?', ['Federal-gov', 'Local-gov', 'Never-worked',
                                                'Private','Self-emp-inc','Self-emp-not-inc',
                                                'State-gov','Without-pay'], key="working_class")
    # Education Level Selection
    st.selectbox('What is your education level?', ['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th',
                                                    'Assoc-acdm', 'Assoc-voc', 'HS-grad', 'Masters', 'Preschool',
                                                    'Prof-school', 'Some-college', 'Bachelors'], key="education_level")
    # Marital Status Selection
    st.selectbox('What is your marital status?', ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse',
                                                'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'], key="marital_status")
    # Occupation Selection
    st.selectbox('What is your occupation?', ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial',
                                            'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct',
                                                'Other-service', 'Priv-house-serv', 'Prof-specialty',
                                                'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving'], key="occupation")
    # Relationship Selection
    st.selectbox('What is your relationship?', ['Husband', 'Not-in-family', 'Other-relative', 'Own-child',
                                                'Unmarried', 'Wife'], key="relationship")
    # Race Selection
    st.selectbox('What is your race?', ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black',
                                        'Other', 'White'], key="race")
    # Sex Selection
    st.selectbox('What is your sex?', ['Female', 'Male'], key="sex")
    # Capital Gain field
    st.number_input('What is your capital gain?', 0, 99999, key="capital_gain")
    # Capital Loss field
    st.number_input('What is your capital loss?', 0, 4356, key="capital_loss")
    # Hours per Week Worked field
    st.slider('How many hours per week do you work?', 0, 99, key="hours_per_week")
    # Country of Origin Selection
    st.selectbox('What is your country of origin?', ['Cambodia', 'Canada', 'China', 'Columbia', 'Cuba',
                                                    'Dominican-Republic', 'Ecuador', 'El-Salvador', 'England',
                                                        'France', 'Germany', 'Greece', 'Guatemala', 'Haiti', 'Holand-Netherlands',
                                                        'Honduras', 'Hong', 'Hungary', 'India', 'Iran', 'Ireland', 'Italy', 'Jamaica',
                                                            'Japan', 'Laos', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru',
                                                            'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South',
                                                                'Taiwan', 'Thailand', 'Trinadad&Tobago', 'United-States',
                                                                'Vietnam', 'Yugoslavia'], key="country_of_origin")
    
    if st.button("Predict"):

        data = { 'age': st.session_state.age, 'workclass': st.session_state.working_class,
                    'education': st.session_state.education_level, 'maritalstatus': st.session_state.marital_status,
                    'occupation': st.session_state.occupation, 'relationship': st.session_state.relationship,
                        'race': st.session_state.race, 'gender': st.session_state.sex, 'capitalgain': st.session_state.capital_gain, 'capitalloss': st.session_state.capital_loss,
                        'hoursperweek': st.session_state.hours_per_week,
                            'nativecountry': st.session_state.country_of_origin }
        
        #convert to dataframe and encode the data using the encoder_dict
        df = pd.DataFrame([data])

        df = encode_data(df, encoder_dict)

        #all features are numerical now, so we can use the model to predict the income level
        features_list = df.values

        prediction = model.predict(features_list)

        output = int(prediction[0])

        if output == 1:
            text = '>50K'
        else:
            text = '<=50K'

        st.success(f'The predicted income level is {text}')


main()
