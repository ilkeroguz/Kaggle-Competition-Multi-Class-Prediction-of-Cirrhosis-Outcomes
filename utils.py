import pandas
import torch
from torch import nn
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

class CirosisCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CirosisCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)  # Daha fazla birim
        self.fc2 = nn.Linear(256, 128)         # Daha fazla birim
        self.fc3 = nn.Linear(128, 64)          # Daha fazla birim
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, num_classes)# Çıkış katmanı
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  # LeakyReLU aktivasyon fonksiyonu
        self.dropout = nn.Dropout(0.5)  # Dropout katmanı

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))  # Leaky ReLU
        x = self.dropout(x)  # Dropout
        x = self.leaky_relu(self.fc2(x))  # Leaky ReLU
        x = self.dropout(x)  # Dropout
        x = self.leaky_relu(self.fc3(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc4(x))# Leaky ReLU
        x = self.fc5(x)# Çıkış katmanı
        return torch.softmax(x, dim=1)  # Softmax çıkışı


def feature_engineering(df):
    df['age_per_day'] = df['Age'] / df['N_Days']
    df['bilirubin_cholesterol_ratio'] = df['Bilirubin'] / df['Cholesterol']
    df['albumin_copper'] = df['Albumin'] + df['Copper']
    df['age_bilirubin_interaction'] = df['Age'] * df['Bilirubin']

# Log dönüşüm - Bilirubin
   # df['log_bilirubin'] = np.log1p(df['Bilirubin'])

# Min-Max scaling - Age
    min_max_scaler = MinMaxScaler()
    df['normalized_age'] = min_max_scaler.fit_transform(df[['Age']])
    return df

def label_scale(df, object_columns, float_columns):
    # Label Encoding ve Standard Scaling
    encoder = LabelEncoder()
    scaler = StandardScaler()
    # Kategorik sütunları etiketleme
    for column in object_columns:
      df[column] = encoder.fit_transform(df[column])

# Sayısal sütunları standardize etme
    df[float_columns] = scaler.fit_transform(df[float_columns])
    return df
