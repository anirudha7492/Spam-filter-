#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam


# In[3]:


# Step 1: Load the dataset
file_path = "C:\\Users\\ashud\\Downloads\\train.csv"  # Replace with the actual path to your train file
data = pd.read_csv(file_path)


# In[4]:



# Check the structure of the data
print(data.head())
print(data.info())


# In[5]:


# Step 2: Preprocess the text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    return text

data['clean_text'] = data['question_text'].apply(preprocess_text)


# In[6]:


# Step 3: Tokenize and pad the sequences
tokenizer = Tokenizer(num_words=20000)  # Adjust the vocabulary size as needed
tokenizer.fit_on_texts(data['clean_text'])


# In[7]:


# Convert text to sequences
sequences = tokenizer.texts_to_sequences(data['clean_text'])
X = pad_sequences(sequences, maxlen=100)  # Adjust maxlen based on your dataset


# In[8]:


# Extract target variable
y = data['target']


# In[9]:


# Step 4: Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


# Step 6: Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=1,  # Adjust the number of epochs
    batch_size=20,  # Adjust the batch size
    verbose=1
)


# In[12]:


# Step 7: Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_accuracy:.4f}")


# In[13]:


# Step 8: Make predictions on the validation set
predictions = model.predict(X_val)
predicted_classes = (predictions > 0.5).astype(int)

# Optional: Save the tokenizer and model
import pickle
tokenizer_path = "tokenizer.pkl"
model_path = "spam_filter_model.h5"

with open(tokenizer_path, "wb") as file:
    pickle.dump(tokenizer, file)
    
model.save(model_path)

print(f"Tokenizer saved to {tokenizer_path}, Model saved to {model_path}")


# In[ ]:




