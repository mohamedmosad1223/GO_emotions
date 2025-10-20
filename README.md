# GO_emotions
– Text Emotion Classification Project
📌 Overview

This project aims to classify emotions in text using BERT.
The GoEmotions dataset is used, which contains 28 different emotions such as joy, sadness, anger, love, fear, and more.

🛠️ Tools and Libraries

Python 3.11

PyTorch

Transformers (Hugging Face)

Datasets (Hugging Face)

scikit-learn

pandas & numpy

🗂️ Dataset

Dataset: GoEmotions Cleaned

Number of rows:

Training: ~187,000

Test: ~20,000

Each row contains:

text: the input text

label: numeric value representing the emotion

⚙️ Project Steps

Load the dataset

from datasets import load_dataset
dataset = load_dataset("Keyurjotaniya007/go-emotions-cleaned")


Clean the text

Lowercasing

Removing punctuation and numbers

Removing extra whitespace

Split the data

Create train / validation sets

Ensure class balance using stratify

Tokenize the text

Use BERT tokenizer to convert text into input IDs

Apply padding & truncation (max length 128)

Fine-tune the model

Model: BertForSequenceClassification

Number of labels: 28

Hyperparameters:

Batch size: 32

Learning rate: 2e-5

Epochs: 3

fp16 for faster GPU training

Evaluation

Metrics: Accuracy & F1-weighted

Validation results:

Accuracy: 41%

F1-weighted: 0.389

📈 Notes

The model could benefit from more training or using DistilBERT for faster performance.

Results can be improved with text augmentation or longer fine-tuning.

🔗 Important Links

Hugging Face Dataset: GoEmotions Cleaned

Transformers Documentation: https://huggingface.co/docs/transformers

👨‍💻 How to Run

Install required libraries:

pip install torch transformers datasets scikit-learn pandas numpy


Run the training script (train.py) or notebook.

After training, use trainer.predict() to make predictions on new text.
