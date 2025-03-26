# Required Libraries
# Install using: pip install numpy pandas scikit-learn tensorflow

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
import pickle

# Step 1: Load the Dataset
file_path = "C:\\Users\\ashud\\Downloads\\train.csv"  # Update with the correct path
data = pd.read_csv(file_path)

# Check Data Structure
print(data.head())
print(data.info())

# Step 2: Preprocess the Text
def preprocess_text(text):
    """
    Cleans the text by converting it to lowercase and removing special characters.

    Parameters:
        text (str): Input text to preprocess.

    Returns:
        str: Cleaned text.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

data['clean_text'] = data['question_text'].apply(preprocess_text)

# Step 3: Tokenization and Padding
tokenizer = Tokenizer(num_words=20000)  # Limiting vocabulary to 20,000 words
tokenizer.fit_on_texts(data['clean_text'])

# Convert Text to Sequences
sequences = tokenizer.texts_to_sequences(data['clean_text'])
X = pad_sequences(sequences, maxlen=100)

# Extract Target Variable
y = data['target']

# Step 4: Split Data into Training and Validation Sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Definition
model = Sequential([
    Embedding(input_dim=20000, output_dim=128, input_length=100),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Step 6: Train the Model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=1,  # Adjust based on performance
    batch_size=20,
    verbose=1
)

# Step 7: Evaluate the Model
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Step 8: Generate Predictions
predictions = model.predict(X_val)
predicted_classes = (predictions > 0.5).astype(int)

# Step 9: Save the Model and Tokenizer
tokenizer_path = "tokenizer.pkl"
model_path = "spam_filter_model.h5"

with open(tokenizer_path, "wb") as file:
    pickle.dump(tokenizer, file)

model.save(model_path)

print(f"Tokenizer saved to {tokenizer_path}, Model saved to {model_path}")





