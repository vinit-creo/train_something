import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_test_model():
    # Load a very small model and only few examples for testing
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir = "./prompt_llm_model"
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    
    logger.info("Loading dataset...")
    dataset = load_dataset("fka/awesome-chatgpt-prompts")
    small_dataset = dataset['train'].select(range(10))
    
    logger.info("Preparing data...")
    def format_prompt(examples):
        prompts = []
        for act, prompt in zip(examples['act'], examples['prompt']):
            formatted = f"Act as {act}\n\nPrompt: {prompt}\n\nResponse:"
            prompts.append(formatted)
        return {'text': prompts}
    
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
            max_length=128,
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
        num_train_epochs=1,  
        per_device_train_batch_size=2,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=1,
        remove_unused_columns=False
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
        train_test_model()
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())