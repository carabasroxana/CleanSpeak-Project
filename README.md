# CleanSpeak-Project

CleanSpeak is a machine learning pipeline designed to detect and rewrite offensive language into non-offensive (polite) alternatives. The project uses both custom-annotated datasets and automated data generated via ChatGPT for training and evaluation.

## 🔍 Project Goals

    Detect offensive language in text.

    Translate or rewrite offensive phrases into polite equivalents.

    Compare the performance of models trained on different data sources (manual annotations vs. ChatGPT-generated corpora).


## ⚙️ Setup

    Clone the repository:

git clone https://github.com/carabasroxana/CleanSpeak.git]

cd CleanSpeak

    Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  
**On Windows:** venv\Scripts\activate

    Install dependencies:

pip install -r requirements.txt

## 🚀 Usage
**1. Data Preparation**

Run preprocessing and annotation scripts:

python data/download_datasets.py

python data/extract_offensive.py

python data/prepare_annotation.py

python data/auto_annotate.py


**2. Training**

Train the model on manually or automatically annotated corpora:

python polite-bot/train.py

For fine-tuning with HuggingFace:

python polite-bot/finetune_hf.py

**3. Evaluation**

python polite-bot/evaluate_metrics.py

**4. Serve Model**

python polite-bot/serve.py

## 🧪 Experiment: ChatGPT-Based Corpus

The script offensive_transl_chatgpt.py evaluates how ChatGPT-generated annotations compare with the manually or automatically annotated corpus.

Use it to generate polite alternatives using ChatGPT and test model behavior without training.