import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import logging
from datasets import load_dataset

# Configure memory settings for Mac M1/M2
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  
torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_test_model():
    model_name = "meta-llama/Llama-3.2-1B"
    output_dir = "./prompt_llm_model"
    

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    logger.info("Loading model...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False
    ).to(device)
    
    model.gradient_checkpointing_enable()
    
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    
    logger.info("Loading dataset...")

    dataset = load_dataset("stevez80/Sci-Fi-Books-gutenberg")
    small_dataset = dataset['train'].select(range(5)) 
    
    logger.info("Preparing data...")
    def format_prompt(examples):
       if "text" in examples:
            return examples["text"]
       elif "content" in examples:
             return  examples["contents"]
       else:
            text_columns = [col for col in examples.keys() if isinstance(examples[col][0], str) and len(examples[col][0]) > 100]
            if text_columns:
                return examples[text_columns[0]]
            else:
                logger.error("Could not find text column in dataset")
                return {"input_ids": [], "attention_mask": []}
    
    formatted_dataset = small_dataset.map(
        format_prompt,
        batched=True,
        remove_columns=small_dataset.column_names
        
    )
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=100,
            return_tensors="pt"
        )
    
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
        
    )
    
    logger.info("Setting up training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        logging_steps=1,
        save_strategy="no", 
        save_total_limit=1,
        remove_unused_columns=False,
        optim="adamw_torch",
        max_grad_norm=0.5,
        learning_rate=2e-5,
        warmup_ratio=0.03,
        gradient_checkpointing=True,
        report_to="none",   
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Test training complete!")

if __name__ == "__main__":
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        train_test_model()
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        logger.info("Attempting to train on CPU...")
        os.environ['PYTORCH_DEVICE'] = 'cpu'
        train_test_model()