# Import necessary libraries
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json
import pickle

# Load FIR dataset
df = pd.read_csv('/content/drive/MyDrive/FIR4.csv')

# Count of objects in each class
values = dict(df['IPC-Section'].value_counts())
print(values)

# Number of classes
num_classes = len(df['IPC-Section'].unique())
print("Number of classes:", num_classes)

# Convert IPC-Section to strings
df['IPC-Section'] = df['IPC-Section'].astype(str)

# Use LabelEncoder to convert string labels to numerical labels
label_encoder = LabelEncoder()
df['IPC-Section'] = label_encoder.fit_transform(df['IPC-Section'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['Description', 'Offense']], df['IPC-Section'], test_size=0.3, random_state=42)
print("Length of training data: ", len(X_train))
print("Length of testing data: ", len(X_test))

# Load pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def preprocess_text(descriptions, offenses):
    # Concatenate Description and Offense
    texts = descriptions + " " + offenses
    # Convert to list and then to strings
    texts = [str(text) for text in texts.tolist()]
    # Tokenize and preprocess the text
    inputs = tokenizer(texts, return_tensors="tf", padding=True, truncation=True, max_length=128)
    return inputs['input_ids'], inputs['attention_mask']

# Load pre-trained transformer model
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)

# Compile the model
model.compile(optimizer=AdamW(learning_rate=0.00001),
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=[SparseCategoricalAccuracy(name="accuracy")])

# Train the model
model.fit(preprocess_text(X_train['Description'], X_train['Offense']), y_train, epochs=60, batch_size=16, validation_data=(preprocess_text(X_test['Description'], X_test['Offense']), y_test))

# Evaluate the model (training)
train_loss, train_accuracy = model.evaluate(preprocess_text(X_train['Description'], X_train['Offense']), y_train)
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Train Loss: {train_loss}%")

# Evaluate the model (testing)
test_loss, test_accuracy = model.evaluate(preprocess_text(X_test['Description'], X_test['Offense']), y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss}%")

# Save the trained model
model.save('/content/drive/MyDrive/ipc_section_model_with_nlp')

# Save the configuration to config.json
model_config = model.config.to_dict()
with open('/content/drive/MyDrive/ipc_section_model_with_nlp/config.json', 'w') as config_file:
    json.dump(model_config, config_file)

# Save the trained model weights
model.save_weights('/content/drive/MyDrive/ipc_section_model_with_nlp/tf_model.h5')

# Save the weights to a file with shape (768, num_classes)
weights = model.get_layer('bert').get_weights()[0]
with open('/content/drive/MyDrive/ipc_section_model_with_nlp/bert_weights_768x{}.npy'.format(num_classes), 'wb') as weights_file:
    np.save(weights_file, weights)




