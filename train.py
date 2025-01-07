import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch import optim
import numpy as np
from utils import CirosisCNN, feature_engineering, label_scale
# Veriyi okuma
train_df = pd.read_csv('data/train.csv')
print(train_df['Status'].unique())
# Kategorik ve sayısal sütunlar
object_columns = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Status']
float_columns = ['id', 'N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage']

train_df = label_scale(train_df, object_columns, float_columns)
train_df = feature_engineering(train_df)
print(train_df['Status'])

X = train_df.drop(columns=['Status'])
y = train_df['Status']
# Veriyi tensöre dönüştürme
X_tensör = torch.tensor(X.values, dtype= torch.float32)
y_tensör = torch.tensor(y.values, dtype= torch.long)

# Model için gerekli parametreler
num_classes = 3
input_size = X_tensör.shape[1]

# Model, Kayıp Fonksiyonu ve Optimizasyon
model = CirosisCNN(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

X_train, X_val, y_train, y_val = train_test_split(X_tensör, y_tensör,
                                                  test_size=0.2,
                                                  random_state=42)

# Eğitim döngüsü
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    y_pred = model(X_train)
    
    loss = criterion(y_pred, y_train)
    loss.backward()
    
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Modelin doğruluğunu hesaplama
model.eval()
with torch.no_grad():
    y_val_pred = model(X_val)
    y_val_pred = torch.argmax(y_val_pred, dim=1)
    accuracy = (y_val_pred == y_val).float().mean()
    print(f'Validation Accuracy: {accuracy.item():.4f}')

torch.save(model.state_dict(), 'Cirrhosis_pred.pth')