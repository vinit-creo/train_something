import pandas as pd

def generate_qa_pairs(df):
    """Convert stock data into QA format"""
    qa_pairs = []
    
    for _, row in df.iterrows():
        context = get_context_for_row(row)
        
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
            },
            {
                "context": context,
                "question": f"What was the lowest price for {row['Symbol']} on {row['Date']}?",
                "answer": str(row['Low'])
            },
            {
                "context": context,
                "question": f"What was the opening price of {row['Symbol']} on {row['Date']}?",
                "answer": str(row['Open'])
            }
        ])
        
        if 'Prev Close' in row:
            qa_pairs.append({
                "context": context,
                "question": f"What was the previous closing price of {row['Symbol']}?",
                "answer": str(row['Prev Close'])
            })
    
    return qa_pairs

def get_context_for_row(row):
    """Generate context string from a row of stock data"""
    context = f"On {row['Date']}, {row['Symbol']} stock opened at {row['Open']}, "
    context += f"reached a high of {row['High']} and a low of {row['Low']}, "
    context += f"and closed at {row['Close']} with a trading volume of {row['Volume']}."
    
    if 'Prev Close' in row:
        context += f" The previous day's closing price was {row['Prev Close']}."
    
    return context

def get_context_for_date_symbol(df, date, symbol):
    """Get the stock context for a specific date and symbol"""
    filtered = df[(df['Date'] == date) & (df['Symbol'] == symbol)]
    if len(filtered) == 0:
        return None
    
    row = filtered.iloc[0]
    return get_context_for_row(row)

def preprocess_for_qa(examples, tokenizer):
    """Preprocess the data for question answering model"""
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]

    inputs = tokenizer(
        questions,
        contexts,
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_mapping = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answer"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        answer = answers[sample_idx]
        context = contexts[sample_idx]
        
        start_char = context.find(answer)
        end_char = start_char + len(answer)
        
        sequence_ids = inputs.sequence_ids(i)
        
        context_start = 0
        while sequence_ids[context_start] != 1:
            context_start += 1
        context_end = len(sequence_ids) - 1
        while sequence_ids[context_end] != 1:
            context_end -= 1
            
        if start_char == -1 or not (offset[context_start][0] <= start_char and end_char <= offset[context_end][1]):
            start_positions.append(0)
            end_positions.append(0)
        else:
            token_start = context_start
            while token_start < len(offset) and offset[token_start][0] <= start_char:
                token_start += 1
            token_start -= 1
            
            token_end = token_start
            while token_end < len(offset) and offset[token_end][1] <= end_char:
                token_end += 1
            token_end -= 1
            
            start_positions.append(token_start)
            end_positions.append(token_end)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs