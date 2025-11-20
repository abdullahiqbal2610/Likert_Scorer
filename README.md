# Psychological Likert Scale Analyzer

An AI-powered tool for automatically scoring free-text responses on a 1-5 Likert scale using fine-tuned transformer models.

## Features

- Automated Likert scale scoring (1-5)
- Fine-tuned DistilBERT model
- Batch response processing
- Confidence scoring
- Professional visualization
- Research-grade data export

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <project-folder>
Install dependencies:

bash
pip install -r requirements.txt
Generate training data and train model:

bash
python generate_improved_dataset.py
python fine_tune_likert_model.py
Run the application:

bash
streamlit run Likert_Scorer_Fixed.py
Project Structure
text
project/
├── generate_improved_dataset.py   # Dataset generation
├── fine_tune_likert_model.py     # Model training
├── Likert_Scorer_Fixed.py        # Streamlit application
├── requirements.txt              # Dependencies
├── README.md                     # This file
└── .gitignore                   # Git ignore rules
Usage
Select a question category and specific item

Enter responses (single or batch)

Click "Analyze Responses" to get AI-powered scoring

View detailed results and export data

Model Performance
Accuracy: ~82-85%

F1-Score: ~0.82-0.85

5-class classification (Likert 1-5)

Technologies Used:

PyTorch

Hugging Face Transformers

Streamlit

Scikit-learn

Plotly
