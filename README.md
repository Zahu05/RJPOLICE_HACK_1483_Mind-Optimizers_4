FIR ANALYSIS USING AI/ML FOR PREDICTING IPC SECTIONS

#Explanation of model

ipc_model's code trains a text classification model using BERT (Bidirectional Encoder Representations from Transformers) for predicting IPC sections based on FIR (First Information Report) data.

Data Loading and Exploration:

FIR dataset (FIR4.csv) is loaded using pandas.
Class distribution of IPC-Section values is printed.
Number of unique classes is determined.
Data Preprocessing:

IPC-Section values are converted to strings.
LabelEncoder is used to convert string labels to numerical labels.
Dataset is split into training and testing sets.
Text Tokenization and Preprocessing:

BERT tokenizer (bert-base-uncased) is loaded.
Text descriptions and offenses are concatenated and tokenized.
Input IDs and attention masks are extracted for both training and testing data.
Model Definition and Compilation:

BERT-based transformer model for sequence classification (TFBertForSequenceClassification) is loaded.
The model is compiled with AdamW optimizer, SparseCategoricalCrossentropy loss, and SparseCategoricalAccuracy metric.
Model Training:

The model is trained using the training data for 60 epochs with a batch size of 16.
Validation data is used to monitor the model's performance during training.
Model Evaluation:

Model accuracy and loss are evaluated on both the training and testing datasets.
Model Saving:

The trained model is saved to /content/drive/MyDrive/ipc_section_model_with_nlp.
Model configuration is saved to config.json.
Model weights are saved to tf_model.h5.
BERT layer weights are saved to a NumPy file.
This code demonstrates the process of loading FIR data, preprocessing text, training a BERT-based classification model, and saving the trained model for later use. The trained model can be employed for predicting IPC sections from new textual data.

#Explanation of Flask_app.py

Flask web application utilizes various libraries for text processing, OCR, and machine learning to predict IPC sections based on provided text data.
Libraries and Models:

The code imports Flask for creating a web application, Pyngrok for creating a public URL, pandas for data manipulation, TensorFlow for machine learning, and various other libraries for text processing (BERT, easyOCR, Summa).
Data Loading and Preprocessing:

FIR dataset (FIR4.csv) and IPC dataset (FF1.csv) are loaded using pandas. IPC-Section column is encoded using LabelEncoder.
BERT tokenizer and a pre-trained BERT model (ipc_section_model_with_nlp) are loaded.
Application Setup:

The Flask app is set up, and ngrok is used to create a public URL for accessibility.
Functions:

Several functions are defined, including text extraction from images using OCR, saving user details, converting PDF to images, and summarizing English text using the Summa library.
Prediction Function:

There's a function for predicting IPC sections based on input text using the loaded BERT model.
Flask Routes:

Two routes are defined:
/ (home), which renders the main page.
/predict (POST method), which processes the uploaded PDF file, extracts text, summarizes it, predicts IPC sections, and displays the result.
Web Interface:

The web interface (index.html) allows users to upload a PDF file and provide personal details. The application extracts text from the file, summarizes it, predicts IPC sections, and displays the results.
Run the Application:

The application is run on a specified port, and the public URL is printed for access.
Overall, the application combines OCR, NLP, and machine learning to analyze legal text data, predict IPC sections, and present the results through a user-friendly web interface.

