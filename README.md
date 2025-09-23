# AI-Driven Phishing Attack Detection and Graph-Based Email Classification

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![BERT](https://img.shields.io/badge/HuggingFace-Transformers-green)](https://huggingface.co/)

## Overview
This repository contains an AI-driven solution for detecting phishing emails using advanced natural language processing (NLP) and graph-based machine learning techniques. The project leverages BERT embeddings for contextual text understanding, combined with graph-based representations using NetworkX to model relationships (e.g., word co-occurrence or email feature networks). Built with PyTorch, it trains neural networks on GPU to classify emails as "Safe" or "Phishing" with improved accuracy over traditional methods.

Key features:
- Data preprocessing and feature extraction using TF-IDF and BERT.
- Graph-based enhancements for relationship modeling.
- Evaluation with precision, recall, and F1-score metrics.

## Technologies
- **Languages/Frameworks**: Python, PyTorch, Hugging Face Transformers (BERT), scikit-learn, NetworkX
- **Data Handling**: pandas, datasets
- **Visualization**: matplotlib, seaborn

## Dataset
- Utilizes a CSV dataset of emails labeled as "Safe Email" or "Phishing Email" (columns: 'Email Text', 'Email Type').
- Source: [Add link to your dataset, e.g., Kaggle or anonymized version in /data/emails.csv].
- Size: [e.g., 10,000 samples]. Note: Sensitive data should be anonymized before uploading.

## Installation & Setup
1. Clone the repository:
git clone https://github.com/yourusername/ai-phishing-detection.git
cd ai-phishing-detection
text2. Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
text3. Install dependencies:
pip install -r requirements.txt
text4. Download the dataset to `/data/` (or mount Google Drive as in the notebook).
5. Ensure GPU support for efficient BERT training (e.g., Google Colab). Install PyTorch with CUDA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
text## Usage
- Open `pad.ipynb` in Jupyter Notebook or Google Colab.
- Run cells sequentially: Data loading → Preprocessing → BERT/TF-IDF features → Graph construction → Model training → Evaluation.
- Predict on new emails using the trained model in the final cells.

## License

2025 9th International Conference on Computational System and Information Technology for Sustainable Solutions (CSITSS






















