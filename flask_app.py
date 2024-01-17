from flask import Flask, render_template, request
from pyngrok import ngrok
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import fitz  # PyMuPDF
import easyocr
from googletrans import Translator
from summa import summarizer

# Load the FIR dataset
df = pd.read_csv('/content/drive/MyDrive/FIR4.csv')
df['IPC-Section'] = df['IPC-Section'].astype(str)
label_encoder = LabelEncoder()
df['IPC-Section'] = label_encoder.fit_transform(df['IPC-Section'])

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
loaded_model = TFBertForSequenceClassification.from_pretrained('/content/drive/MyDrive/ipc_section_model_with_nlp', from_pt=False)

# Load the IPC dataset with additional columns
ipc_data = pd.read_csv('/content/drive/MyDrive/FF1.csv')

#Set up the output folder
user_uploads_folder = '/content/drive/MyDrive/user_uploads'
os.makedirs(user_uploads_folder, exist_ok=True)

# Set up the user details folder
user_details_folder = '/content/drive/MyDrive/user_details'
os.makedirs(user_details_folder, exist_ok=True)

port_no = 5000

app = Flask(__name__)
ngrok.set_auth_token('2arGLDR2EMWABImIICjP510EKHn_7g87CpnHc1oG2prwVd65K')

public_url = ngrok.connect(port_no).public_url

@app.route("/")
def home():
    return render_template("index.html", result=None, input_text=None)

def save_user_details(name, adhar, phone_no, uploaded_image_path):
    user_details_file = os.path.join(user_details_folder, 'user_details.csv')
    data = {'Name': [name], 'Adhar': [adhar], 'PhoneNo': [phone_no], 'ImagePath': [uploaded_image_path]}
    df_user_details = pd.DataFrame(data)

    if os.path.exists(user_details_file):
        df_user_details.to_csv(user_details_file, mode='a', header=False, index=False)
    else:
        df_user_details.to_csv(user_details_file, index=False)

def convert_pdf_to_images(pdf_path, output_folder='output_images'):
    try:
        pdf_document = fitz.open(pdf_path)

        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            image = page.get_pixmap()

            image_path = f"/content/drive/MyDrive/user_uploads/{output_folder}/page_{page_number + 1}.png"
            image.save(image_path)

        pdf_document.close()

    except Exception as e:
        print(f"Error occurred during PDF to image conversion: {e}")

def extract_hindi_text(image_path):
    try:
        reader = easyocr.Reader(['hi'])
        results = reader.readtext(image_path)
        extracted_text = ' '.join([result[1] for result in results])
        return extracted_text
    except Exception as e:
        print(f"Error occurred during Hindi text extraction: {e}")
        return ''

def translate_hindi_to_english(hindi_text):
    translator = Translator()
    english_text = translator.translate(hindi_text, src='hi', dest='en').text
    return english_text

def summarize_english_text(english_text):
    summary = summarizer.summarize(english_text, ratio=0.2)
    return summary

# Function to predict IPC Section for a given input
def predict_ipc_section(input_text):
    # Tokenize and preprocess the text
    inputs = tokenizer(input_text, return_tensors="tf", padding=True, truncation=True, max_length=128)

    # Make predictions using the loaded model
    predictions = loaded_model(inputs)
    top_indices = np.argsort(predictions.logits)[0][-5:]
    label_encoded = label_encoder.inverse_transform(top_indices)

    results = []

    for i in range(5):
        ipc_section = label_encoded[4-i]

        # Find the row in the IPC dataset with the predicted IPC-Section label
        predicted_row = ipc_data[ipc_data['IPC-Section'] == ipc_section].iloc[0]

        result = {
            "ipc_section": ipc_section,
            "score": predictions.logits[0].numpy()[top_indices[i]],
            "description": predicted_row['Description'],
            "offense": predicted_row['Offense'],
            "punishment": predicted_row['Punishment'],
            "bailable": predicted_row['Bailable'],
            "court": predicted_row['Court'],
        }

        results.append(result)

    return results

@app.route("/predict", methods=["POST"])
def predict():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return render_template("index.html", result=None, input_text=None)

    file = request.files['file']

    # If the user does not select a file, return to the home page
    if file.filename == '':
        return render_template("index.html", result=None, input_text=None)

    if file:
        name = request.form.get('name')
        adhar = request.form.get('aadhar')
        phone_no = request.form.get('mobile')

        file_path = os.path.join(user_uploads_folder, 'uploaded.pdf')
        file.save(file_path)
        # Convert PDF to images
        convert_pdf_to_images(file_path)

        # Extract Hindi text from images
        hindi_text = ""
        image_folder = '/content/drive/MyDrive/user_uploads/output_images'
        for image_filename in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image_filename)
            if os.path.isfile(image_path) and image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                hindi_text += extract_hindi_text(image_path) + ' '

        # Translate Hindi to English
        english_text = translate_hindi_to_english(hindi_text)

        # Summarize English text
        summary = summarize_english_text(english_text)

        # Predict IPC Section
        result = predict_ipc_section(english_text)
        save_user_details(name, adhar, phone_no, file_path)

        return render_template("index.html", result=result, input_text=summary)
print('\n\n')
print("Please visit this public URL to access the app: ", public_url)
if __name__ == "__main__":
    app.run(port=port_no)
