import os
import pandas as pd
import torch
import kagglehub
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from datasets import Dataset
import kaggle
from sklearn.model_selection import train_test_split
from utils import generate_qa_pairs, preprocess_for_qa


def download_dataset(dataset_name="hk7797/stock-market-india", data_path="./data"):
    """Download the stock market dataset from Kaggle"""
    os.makedirs(data_path, exist_ok=True)
    try:
        kagglehub.dataset_download(dataset_name,path=data_path,)
        print(f"Successfully downloaded dataset to {data_path}")
        return pd.read_csv(os.path.join(data_path, 'STOCK_DATA.csv'))
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        if os.path.exists(os.path.join(data_path, 'STOCK_DATA.csv')):
            print("Using existing dataset file")
            return pd.read_csv(os.path.join(data_path, 'STOCK_DATA.csv'))
        else:
            raise FileNotFoundError("Dataset file not found. Please check your Kaggle API setup or download the file manually.")

def train_qa_model(model_name="bert-base-uncased", output_dir="./stock_qa_model", 
                  data_path="./data", num_train_epochs=3, batch_size=8):
    """Train a question answering model on stock market data"""
    
    df = download_dataset(data_path=data_path)
    
    qa_dataset = generate_qa_pairs(df)
    print(f"Generated {len(qa_dataset)} QA pairs")
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    train_raw, eval_raw = train_test_split(qa_dataset, test_size=0.2, random_state=42)
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_raw))
    eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_raw))
    
    print("Preprocessing train dataset...")
    train_dataset = train_dataset.map(
        lambda examples: preprocess_for_qa(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    print("Preprocessing evaluation dataset...")
    eval_dataset = eval_dataset.map(
        lambda examples: preprocess_for_qa(examples, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=500,
        logging_steps=100,
        learning_rate=3e-5,
        weight_decay=0.01,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DefaultDataCollator(),
    )
    
    trainer.train()
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    download_dataset()
    MODEL_DIR = "./stock_qa_model"
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    train_qa_model(output_dir=MODEL_DIR)