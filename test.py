# test.py
# Description: This file is used to test the predict_mode function from the src/predict module.
# Author: Vidhun Rangaraajan J
# Website: https://www.vidhun.com
# Github: https://github.com/VidhunRangaraajan
# Repository: https://github.com/VidhunRangaraajan/Chronic-Kidney-Disease-Prediction
# Requirements: None directly(uses predict_mode function from src/predict.py)
# Usage: To check the functionality of the predict_mode function with sample inputs from out of the file.
# Depends On: src/predict.py
# Input Files: -
# Output Files: -
# Notes: The parameters given for the catogorical columns shows both the possible stings that can be passed to each catogorical column and any int/float can be passed to the numeric data respectively.
# To Do: -

# Importing the predict_mode function from the "src/predict.py" module/file.
from src.predict import predict_mode

print(predict_mode({
  "age": "65",
  "blood_pressure": "150",
  "specific_gravity": "1.010",
  "albumin": "4",
  "sugar": "3",
  "red_blood_cells": "abnormal",
  "pus_cell": "abnormal",
  "pus_cell_clumps": "present",
  "bateria": "present",
  "blood_glucose_random": "250",
  "blood_urea": "90",
  "serum_creatinine": "5.0",
  "sodium": "130",
  "potassium": "6.0",
  "haemoglobin": "8.5",
  "packed_cell_volume": "28",
  "white_blood_cell_count": "12000",
  "red_blood_cell_count": "3.0",
  "hypertension": "yes",
  "diabetes_mellitus": "yes",
  "coronary_artery_disease": "yes",
  "appetite": "poor",
  "pedal_edema": "yes",
  "anemia": "yes",
}
))#Output Sample 1: ['CKD', 1, 1, 1, 1, 1]

print(predict_mode({
  "age": "40",
  "blood_pressure": "110",
  "specific_gravity": "1.025",
  "albumin": "0",
  "sugar": "0",
  "red_blood_cells": "normal",
  "pus_cell": "normal",
  "pus_cell_clumps": "notpresent",
  "bateria": "notpresent",
  "blood_glucose_random": "95",
  "blood_urea": "25",
  "serum_creatinine": "0.9",
  "sodium": "140",
  "potassium": "4.2",
  "haemoglobin": "14.0",
  "packed_cell_volume": "42",
  "white_blood_cell_count": "7500",
  "red_blood_cell_count": "5.2",
  "hypertension": "no",
  "diabetes_mellitus": "no",
  "coronary_artery_disease": "no",
  "appetite": "good",
  "pedal_edema": "no",
  "anemia": "no",
}
))# Output Sample 2: ['Not CKD', 0, 0, 0, 0, 0]

# In the output ["Final by taking the mode of the predictions", 1('CKD')/0('Not CKD') of the best model, 1/0(model-2) , 1/0(model-3), 1/0(model-4), 1/0(model-5)].