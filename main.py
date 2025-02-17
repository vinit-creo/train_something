import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from datasets import Dataset
import kaggle
import os
from sklearn.model_selection import train_test_split

class StockMarketQA:
    def __init__(self, model_name="bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
    def download_dataset(self):
        """Download the stock market dataset from Kaggle"""
        dataset_name = "hk7797/stock-market-india"
        kaggle.api.dataset_download_files(dataset_name, path='./data', unzip=True)
        return pd.read_csv('./data/STOCK_DATA.csv')

    def prepare_qa_dataset(self, df):
        """Convert stock data into QA format"""
        qa_pairs = []
        
        # Generate QA pairs for each stock entry
        for _, row in df.iterrows():
            context = f"On {row['Date']}, {row['Symbol']} stock opened at {row['Open']}, "
            context += f"reached a high of {row['High']} and a low of {row['Low']}, "
            context += f"and closed at {row['Close']} with a trading volume of {row['Volume']}."
            
            # Create multiple QA pairs for each data point
            qa_pairs.extend([
                {
                    "context": context,
                    "question": f"What was the closing price of {row['Symbol']} on {row['Date']}?",
                    "answer": str(row['Close'])
                },
                {
                    "context": context,
                    "question": f"What was the trading volume of {row['Symbol']} on {row['Date']}?",
                    "answer": str(row['Volume'])
                },
                {
                    "context": context,
                    "question": f"What was the highest price for {row['Symbol']} on {row['Date']}?",
                    "answer": str(row['High'])
                }
            ])
        
        return Dataset.from_pandas(pd.DataFrame(qa_pairs))

    def preprocess_function(self, examples):
        """Preprocess the data for the model"""
        questions = [q.strip() for q in examples["question"]]
        contexts = [c.strip() for c in examples["context"]]

        inputs = self.tokenizer(
            questions,
            contexts,
            max_length=384,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        answers = examples["answer"]
        start_positions = []
        end_positions = []

        for i, context in enumerate(contexts):
            answer = answers[i]
            answer_start = context.find(answer)
            answer_end = answer_start + len(answer)
            
            start_position = inputs.char_to_token(i, answer_start)
            end_position = inputs.char_to_token(i, answer_end - 1)
            
            if start_position is None:
                start_position = 0
            if end_position is None:
                end_position = 0
                
            start_positions.append(start_position)
            end_positions.append(end_position)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def train(self, dataset, output_dir="./stock_qa_model"):
        """Train the model"""
        train_dataset, eval_dataset = train_test_split(dataset, test_size=0.2)
        train_dataset = Dataset.from_dict(train_dataset)
        eval_dataset = Dataset.from_dict(eval_dataset)

        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        eval_dataset = eval_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=eval_dataset.column_names
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps",
            eval_steps=500,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            save_steps=500,
            logging_steps=100,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DefaultDataCollator(),
        )

        trainer.train()
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def answer_question(self, question, context):
        """Use the model to answer a question"""
        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            max_length=384,
            truncation=True,
            padding="max_length"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits)

        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0][answer_start:answer_end+1]
            )
        )

        return answer

def main():
    qa_system = StockMarketQA()
    
    df = qa_system.download_dataset()
    dataset = qa_system.prepare_qa_dataset(df)
    
    qa_system.train(dataset)
    
    context = dataset[0]["context"]
    question = "What was the closing price?"
    answer = qa_system.answer_question(question, context)
    print(f"\nQuestion: {question}")
    print(f"Context: {context}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()