from utils import feature_engineering, label_scale, CirosisCNN
import pandas as pd
import torch
import numpy as np
test_df = pd.read_csv('data/test.csv')

id_column = test_df['id']
print(test_df.head())

object_columns = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
float_columns = ['id', 'N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage']

test_df = label_scale(test_df, object_columns, float_columns)
test_df = feature_engineering(test_df)

test_tensor = torch.tensor(test_df.values, dtype= torch.float32)

input_size = test_tensor.shape[1]
num_classes = 3
model = CirosisCNN(input_size, num_classes)
model.load_state_dict(torch.load('Cirrhosis_pred.pth'))

X = test_tensor

with torch.no_grad():
   y_pred =  model(X)
   y_pred = torch.argmax(y_pred, dim=1) 



cirrhosis_classes = ['D', 'C', 'CL']
print(y_pred)


result_df = pd.DataFrame({
    'id': id_column,
    'Obesity': [cirrhosis_classes[label] for label in y_pred]
})
result_df.to_csv('test_predictions.csv', index=False)
print("Test sonuçları kaydedildi: test_predictions.csv")
