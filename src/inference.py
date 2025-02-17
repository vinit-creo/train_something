import os
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import pandas as pd
from utils import get_context_for_date_symbol

class StockMarketQA:
    def __init__(self, model_path="./stock_qa_model", data_path="./data"):
        """Initialize the QA system with the trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory {model_path} not found. Make sure to train the model first.")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.model.eval()  # Set to evaluation mode
        
        self.data_path = data_path
        self.df = pd.read_csv(os.path.join(data_path, 'STOCK_DATA.csv'))
        print(f"Loaded stock data with {len(self.df)} entries")
        
        self.dates = sorted(self.df['Date'].unique())
        self.symbols = sorted(self.df['Symbol'].unique())
        
    def answer_question(self, question, context):
        """Get answer for a specific question"""
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
        
        if answer_end < answer_start:
            answer_end = answer_start
        
        tokens = self.tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0][answer_start:answer_end+1]
        )
        
        answer = self.tokenizer.convert_tokens_to_string(tokens)
        
        answer = answer.strip()
        
        # If answer is empty or just punctuation/special chars, return "No answer found"
        if not answer or all(c in '.,!?;:()[]{}<>-' for c in answer):
            return "No answer found in the context"
        
        return answer

    def get_available_dates(self, n=5):
        """Return a sample of available dates"""
        return self.dates[:n]
    
    def get_available_symbols(self, n=5):
        """Return a sample of available symbols"""
        return self.symbols[:n]

def main():
    try:
        qa_system = StockMarketQA()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run train.py first to train the model.")
        return
    
    sample_dates = qa_system.get_available_dates()
    sample_symbols = qa_system.get_available_symbols()
    
    print("\nStock Market QA System (type 'exit' to quit)")
    print(f"Sample dates: {', '.join(sample_dates)}")
    print(f"Sample symbols: {', '.join(sample_symbols)}")
    print("Full list available in STOCK_DATA.csv")
    
    while True:
        date = input("\nEnter date (YYYY-MM-DD): ")
        if date.lower() == 'exit':
            break
            
        symbol = input("Enter stock symbol (e.g., TCS, RELIANCE): ")
        if symbol.lower() == 'exit':
            break
            
        question = input("Enter your question: ")
        if question.lower() == 'exit':
            break

        try:
            context = get_context_for_date_symbol(qa_system.df, date, symbol)
            if context:
                answer = qa_system.answer_question(question, context)
                
                print("\nContext:", context)
                print("Question:", question)
                print("Answer:", answer)
            else:
                print(f"No data found for {symbol} on {date}")
                print(f"Available dates: {', '.join(qa_system.get_available_dates())}")
                print(f"Available symbols: {', '.join(qa_system.get_available_symbols())}")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Please check if the date and symbol are correct.")

if __name__ == "__main__":
    main()