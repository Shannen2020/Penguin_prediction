import streamlit as st 
import pandas as pd
import pickle
import numpy as np
# from sklearn.ensemble import RandomForestClassifier

st.write("""
# Penguin Prediction App 

This app predicts the **Palmer Penguin** species!

""")

st.sidebar.header("User Input Features")

st.sidebar.markdown("""
[Example CSV input file](http://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

#Collect user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your file here", type=['csv'])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        sex = st.sidebar.selectbox('Sex',('male','female'))
        island = st.sidebar.selectbox('Island', ('Biscoe','Dream', 'Torgersen'))
        bill_length_mm = st.sidebar.slider('Bill length(mm)', 32.1, 59.6, 42.0)
        bill_depth_mm = st.sidebar.slider('Bill depth(mm)', 13.1, 21.5, 14.0)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0, 231.0)
        body_mass_g = st.sidebar.slider('Body mass(g)', 2700, 6300)
        data = {'island':island,
               'bill_length_mm': bill_length_mm,
               'bill_depth_mm': bill_depth_mm,
               'flipper_length_mm': flipper_length_mm,
               'body_mass_g': body_mass_g,
               'sex':sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for encoding phase
penguins_raw =  pd.read_csv('penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df, penguins], axis=0)

# Encode ordinal features
encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # selects only the first row (user input data)

# Displays the user input features

st.subheader('User Input Features')

if uploaded_file is not None:
    st.write(df) # prints the table
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters(shown below)')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction')
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction probability')
st.write(prediction_proba)




