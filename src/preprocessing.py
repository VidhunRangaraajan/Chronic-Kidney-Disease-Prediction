# Requirements - Pandas, Numpy

# Importing required libraries.
import pandas as pd
import numpy as np

# Creating a dataframe from the csv file.
df_data = pd.read_csv('../data/kidney_disease.csv')

# Dropping the 'id' column as pationt id will not determine the presence of CKD.
df_data.drop('id', axis=1, inplace=True)

# Renaming the columns for better understanding and readability.
df_data.columns=['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bateria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium', 'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count', 'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'pedal_edema', 'anemia','class']

# Numerical columns that are stored as string.
text_columns=['packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count']

#Converting the numerical columns that are stored as string to numeric values.
def convert_to_numeric(columns):
    for i in columns:
        df_data[i] = pd.to_numeric(df_data[i], errors='coerce')
convert_to_numeric(text_columns)

# Function to fill the null values in numerical columns with mean value of the respective column.
def mean_value_imputation(df_data, column):
    mean_values = df_data[column].mean()
    df_data[column]=df_data[column].fillna(mean_values)

# Function to fill the null values in categorical columns with mode value of the respective column.
def mode_value_imputation(df_data, column):
    mode_values = df_data[column].mode()[0]
    df_data[column]=df_data[column].fillna(mode_values)

# Identifying numerical or categorical column and adding it to the respective list.
num_cols=[col for col in df_data.columns if df_data[col].dtype in ['float64']]
cat_cols=[col for col in df_data.columns if df_data[col].dtype == 'str']

# Filling the null values in numerical columns with mean value of the respective column.
for col_name in num_cols:
    mean_value_imputation(df_data, col_name)

# Filling the null values in categorical columns with mode value of the respective column.
for col_name in cat_cols:
    mode_value_imputation(df_data, col_name)
    
# Replacing categorical values with consistent formatting
df_data['diabetes_mellitus'] = df_data['diabetes_mellitus'].replace(to_replace={' yes':'yes', '\tyes':'yes','\tno':'no'})
df_data['coronary_artery_disease'] = df_data['coronary_artery_disease'].replace(to_replace='\tno', value='no')
df_data['class'] = df_data['class'].replace(to_replace='ckd\t', value='ckd')

# Replacing categorical values with numerical values
df_data['class']=df_data['class'].map({'ckd':1, 'notckd':0})
df_data['red_blood_cells']=df_data['red_blood_cells'].map({'abnormal':0, 'normal':1})
df_data['pus_cell']=df_data['pus_cell'].map({'abnormal':0, 'normal':1})
df_data['pus_cell_clumps']=df_data['pus_cell_clumps'].map({'present':1, 'notpresent':0})
df_data['bateria']=df_data['bateria'].map({'present':1, 'notpresent':0})
df_data['hypertension']=df_data['hypertension'].map({'yes':1, 'no':0})
df_data['diabetes_mellitus']=df_data['diabetes_mellitus'].map({'yes':1, 'no':0})
df_data['coronary_artery_disease']=df_data['coronary_artery_disease'].map({'yes':1, 'no':0})
df_data['appetite']=df_data['appetite'].map({'good':1, 'poor':0})
df_data['pedal_edema']=df_data['pedal_edema'].map({'yes':1, 'no':0})
df_data['anemia']=df_data['anemia'].map({'yes':1, 'no':0})


df_data.info()
df_data.describe()