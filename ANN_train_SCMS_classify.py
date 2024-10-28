import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import os
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_csv('combined_cell_lines.csv') # change the file name as your own file
data.columns = data.columns.str.strip()
data = data.dropna()

# Manually assign group codes
group_mapping = {'IGR37': 0, 'IGR39': 1, 'IGR39_treated': 2, 'IGR37_treated': 3, 'WM115':4, 'WM115_treated': 5, 'WM2664': 6, 'WM2664_treated': 7}
data['Group'] = data['Group'].map(group_mapping)

# Function to split the data: training, validation and test
def split_data(df, train_size, val_size):
    train_data, temp_data = train_test_split(df, train_size=train_size, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=(1 - val_size / (1 - train_size)), random_state=42)
    return train_data, val_data, test_data

# Splitting data for groups A:IGR37 untreated cells and B: IGR39 untreated cells
train_A, val_A, test_A = split_data(data[data['Group'] == 0], 0.6, 0.2) # 60% of data A for training, 20% of data A for testing
train_B, val_B, test_B = split_data(data[data['Group'] == 1], 0.6, 0.2) # 60% of data B for training, 20% of data B for testing

# Combine splits
train_data = pd.concat([train_A, train_B])
val_data = pd.concat([val_A, val_B])
test_data = pd.concat([test_A, test_B])

# Separate features and target variables
X_train = train_data.drop(columns=['Group', 'sampleid'])
y_train = train_data['Group']
X_val = val_data.drop(columns=['Group', 'sampleid'])
y_val = val_data['Group']
X_test = test_data.drop(columns=['Group', 'sampleid'])
y_test = test_data['Group']

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Custom wrapper class
# Adjust the hyperparameters for your project
class MyKerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.0001, epochs=500, batch_size=32):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, X, y):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        history = self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping])
        return history

    def score(self, X, y):
        return self.model.evaluate(X, y)

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype("int32")

# Wrap model
model = MyKerasClassifier()

# Train model and store history
history = model.fit(X_train, y_train)

# Evaluate model on test data
test_loss, test_accuracy = model.score(X_test, y_test)

# Predict on the rest of groups
# Double check the group mapping code
X_C = scaler.transform(data[data['Group'] == 2].drop(columns=['Group', 'sampleid']))
y_C = data[data['Group'] == 2]['Group']
X_D = scaler.transform(data[data['Group'] == 3].drop(columns=['Group', 'sampleid']))
y_D = data[data['Group'] == 3]['Group']
X_E = scaler.transform(data[data['Group'] == 4].drop(columns=['Group', 'sampleid']))
y_E = data[data['Group'] == 4]['Group']
X_F = scaler.transform(data[data['Group'] == 5].drop(columns=['Group', 'sampleid']))
y_F = data[data['Group'] == 5]['Group']
X_G = scaler.transform(data[data['Group'] == 6].drop(columns=['Group', 'sampleid']))
y_G = data[data['Group'] == 6]['Group']
X_H = scaler.transform(data[data['Group'] == 7].drop(columns=['Group', 'sampleid']))
y_H = data[data['Group'] == 7]['Group']


pred_C = model.predict(X_C)
pred_D = model.predict(X_D)
pred_E = model.predict(X_E)
pred_F = model.predict(X_F)
pred_G = model.predict(X_G)
pred_H = model.predict(X_H)

# Add predictions to the dataframe
data.loc[data['Group'] == 2, 'Prediction'] = pred_C
data.loc[data['Group'] == 3, 'Prediction'] = pred_D
data.loc[data['Group'] == 4, 'Prediction'] = pred_E
data.loc[data['Group'] == 5, 'Prediction'] = pred_F
data.loc[data['Group'] == 6, 'Prediction'] = pred_G
data.loc[data['Group'] == 7, 'Prediction'] = pred_H


# Save results with group mappings for clarity
results_dir = 'Output_folder' #change it to any name you like
os.makedirs(results_dir, exist_ok=True)
data.to_csv(os.path.join(results_dir, 'predictions.csv'), index=False)
with open(os.path.join(results_dir, 'results.csv'), 'w') as f:
    f.write(f'Test Loss: {test_loss}\n')
    f.write(f'Test Accuracy: {test_accuracy}\n')
    f.write(f'Predictions for Group IGR39_treated (2): {pred_C}\n')
    f.write(f'Predictions for Group IGR37_treated (3): {pred_D}\n')
    f.write(f'Predictions for Group WM115 (4): {pred_E}\n')
    f.write(f'Predictions for Group WM115_treated (5): {pred_F}\n')
    f.write(f'Predictions for Group WM2664 (6): {pred_G}\n')
    f.write(f'Predictions for Group WM2664_treated (7): {pred_H}\n')

# Plot and save accuracy and loss graphs
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.savefig(os.path.join(results_dir, 'accuracy.png'))

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig(os.path.join(results_dir, 'loss.png'))

print('Results and plots saved in', results_dir)

print(f'Group mapping: {group_mapping}')

