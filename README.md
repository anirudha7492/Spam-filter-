# Spam-filter-
# Quora Spam Filter

This project implements a **Spam Detection System** to identify spam questions on Quora using **Bidirectional LSTM** models. Natural Language Processing (NLP) techniques are used to preprocess the text data, followed by training a deep learning model for binary classification.

## Dataset
- The dataset consists of Quora questions labeled as spam (1) or not spam (0).
- Download the dataset from the given link: [Download Here](https://www.dropbox.com/sh/kpf9z73woodfssv/AAAw1_JIzpuVvwteJCma0xMla?dl=0)

## Project Structure
```bash
- train.csv            # Dataset containing labeled Quora questions
- spam_filter.py        # Main Python script for training and evaluating the model
- README.md             # Project documentation
- tokenizer.pkl         # Saved tokenizer for inference
- spam_filter_model.h5  # Trained model
```

## Requirements
Install the following dependencies using `pip`:
```bash
pip install numpy pandas scikit-learn tensorflow
```

## Solution Concept

### Step 1: Data Preprocessing
- **Text Cleaning**: All questions are converted to lowercase and special characters are removed using **Regular Expressions (Regex)**.
- **Tokenization**: Using Keras' **Tokenizer** to convert text data into sequences of tokens.
- **Padding**: Sequences are padded to a fixed length using `pad_sequences()` to ensure uniformity.

### Step 2: Model Building
- **Embedding Layer**: Converts tokenized words into dense vectors using word embeddings. This allows the model to understand the semantic meaning of words.
- **Bidirectional LSTM**: A type of recurrent neural network that reads input from both directions to capture context more effectively.
- **Dropout Layer**: Prevents overfitting by randomly dropping neurons during training.
- **Dense Layer**: A fully connected layer with a sigmoid activation function to output a binary classification (spam or not spam).

### Model Architecture
```
Embedding -> Bidirectional LSTM -> Dropout -> Dense (Sigmoid)
```
- **Embedding Layer**: Input dimension = 20,000 words, Output dimension = 128  
- **LSTM Layer**: 64 units, bidirectional for better context understanding  
- **Dropout**: 50% neurons dropped to reduce overfitting  
- **Dense**: Single neuron with Sigmoid activation for binary output  

### Step 3: Training
- The model is trained using the **Adam optimizer** with **binary crossentropy** as the loss function.  
- Validation data is used to monitor performance.  

### Step 4: Evaluation and Prediction
- After training, the model's accuracy is evaluated on the validation set.  
- Predictions are made using a threshold of 0.5, classifying questions as spam or not.  

 
