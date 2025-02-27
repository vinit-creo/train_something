import os
import argparse
import logging
import random
import torch
import traceback
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_fix.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps",)
        logger.info("::: Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("::: Using CUDA GPU")
    else:
        device = torch.device("cpu")
        logger.info("::: Using CPU (training will be slow)")
    return device

def prepare_insurance_qa_dataset(tokenizer, args):
    """Prepare the Insurance QA dataset avoiding NumPy dependency issues."""
    
    predefined_examples = [
        {"question": "What is insurance?", 
         "answer": "Insurance is a contract represented by a policy in which an individual or entity receives financial protection or reimbursement against losses from an insurance company."},
        {"question": "What is a premium?", 
         "answer": "A premium is the amount of money an individual or business pays for an insurance policy."},
        {"question": "What is a deductible?", 
         "answer": "A deductible is the amount of money you pay out of pocket before your insurance coverage kicks in."},
        {"question": "What is coinsurance?", 
         "answer": "Coinsurance is the percentage of costs that you pay for covered services after you've met your deductible."},
    ]
    
    try:
        logger.info(f"Loading Insurance QA dataset...")
        dataset = load_dataset("rvpierre/insurance-qa-en")
        logger.info(f"Dataset loaded with splits: {list(dataset.keys())}")
        
        train_split = "train"
        if train_split not in dataset:
            logger.warning(f"Train split not found in dataset, using predefined examples")
            examples = predefined_examples
        else:
            logger.info(f"Processing examples from {train_split} split")
            examples = []
            
            use_answer_pool = False
            if "answers_pool" in dataset[train_split].features:
                use_answer_pool = True
                answers_pool = dataset[train_split].features["answers_pool"]
                logger.info(f"Using answers pool with {len(answers_pool)} answers")
            
            count = 0
            for i, example in enumerate(dataset[train_split]):
                if args.max_examples and count >= args.max_examples:
                    break
                
                if "question" not in example or not example["question"]:
                    continue
                    
                question = example["question_en"]
                
                if "answers" in example and example["answers"] and len(example["answers"]) > 0:
                    if use_answer_pool and isinstance(example["answers"][0], int):
                        answer_idx = example["answers"][0] 
                        if 0 <= answer_idx < len(answers_pool):
                            answer = answers_pool[answer_idx]
                            examples.append({"question": question, "answer": answer})
                            count += 1
                    elif isinstance(example["answers"][0], str):
                        answer = example["answers"][0]
                        examples.append({"question": question, "answer": answer})
                        count += 1
            
            if not examples:
                logger.warning("No valid examples extracted from dataset, using predefined examples")
                examples = predefined_examples
            else:
                logger.info(f"Successfully extracted {len(examples)} examples from dataset")
                
    except Exception as e:
        logger.error(f"Error loading or processing dataset: {e}")
        logger.error(traceback.format_exc())
        logger.warning("Using predefined examples instead")
        examples = predefined_examples
    
    # Format examples for Llama-3 format
    formatted_texts = []
    
    for ex in examples:
        formatted_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are an insurance expert that provides clear, accurate, and helpful information about insurance concepts, policies, and practices.\n\n<|start_header_id|>user<|end_header_id|>\n{ex['question']}\n\n<|start_header_id|>assistant<|end_header_id|>\n{ex['answer']}<|end_of_text|>"
        formatted_texts.append(formatted_text)
    
    logger.info(f"Created {len(formatted_texts)} formatted examples")
    
    # Split into train and eval (90/10 split)
    train_size = max(1, int(0.9 * len(formatted_texts)))
    train_texts = formatted_texts[:train_size]
    eval_texts = formatted_texts[train_size:] if len(formatted_texts) > train_size else [formatted_texts[0]]
    
    logger.info(f"Split into {len(train_texts)} training and {len(eval_texts)} evaluation examples")
    
    try:
        logger.info(f"Tokenizing examples (max_length={args.max_length})...")
        
        def tokenize_and_to_lists(texts):
            encodings = tokenizer(
                texts, 
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt"
            )
            
            return {
                "input_ids": encodings["input_ids"].tolist(),
                "attention_mask": encodings["attention_mask"].tolist(),
                "labels": encodings["input_ids"].tolist()  # Same as input_ids for causal LM
            }
        
        train_dict = tokenize_and_to_lists(train_texts)
        eval_dict = tokenize_and_to_lists(eval_texts)
        
        train_dataset = Dataset.from_dict(train_dict)
        eval_dataset = Dataset.from_dict(eval_dict)
        
        logger.info(f"Created datasets with {len(train_dataset)} training and {len(eval_dataset)} evaluation examples")
        
        def convert_to_tensors(example):
            return {
                "input_ids": torch.tensor(example["input_ids"]),
                "attention_mask": torch.tensor(example["attention_mask"]),
                "labels": torch.tensor(example["labels"])
            }
        
        train_dataset.set_transform(convert_to_tensors)
        eval_dataset.set_transform(convert_to_tensors)
        
        return train_dataset, eval_dataset
        
    except Exception as e:
        logger.error(f"Error during tokenization or dataset creation: {e}")
        logger.error(traceback.format_exc())
        raise

def load_model_and_tokenizer(args):
    """Load Llama-3 model optimized for Mac."""
    logger.info(f"Loading model: {args.model}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Using EOS token as padding token")
        else:
            tokenizer.pad_token = tokenizer.bos_token if tokenizer.bos_token else "<pad>"
            logger.info(f"Set padding token to {tokenizer.pad_token}")
    
    device = get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32  # Only use float16 for CUDA
    
    try:
        logger.info(f"Loading model on {device} with dtype {dtype}")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=dtype,
                device_map={"": device}
            )
            logger.info("Model loaded with device mapping")
        except Exception as e:
            logger.warning(f"Error loading model with device mapping: {e}")
            logger.info("Trying standard loading method...")
            
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=dtype
            )
            model = model.to(device)
            logger.info("Model loaded with standard method")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(traceback.format_exc())
        raise

def train_model(model, tokenizer, train_dataset, eval_dataset, args):
    """Train the model with Mac-friendly settings."""
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = get_device()
    
    # FIX: Only use fp16 with CUDA, not with MPS
    use_fp16 = device.type == "cuda" and not args.disable_fp16
    if device.type == "mps":
        logger.info("MPS detected - disabling fp16 as it's not compatible")
        use_fp16 = False
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=use_fp16,
        optim="adamw_torch",
        dataloader_num_workers=0,  
        report_to="tensorboard",
        gradient_checkpointing=args.gradient_checkpointing,
        logging_first_step=True,
    )
    
    logger.info(f"Training configuration: {training_args}")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    logger.info("Starting training...")
    try:
        trainer.train()
        
        logger.info(f"Saving model to {args.output_dir}")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        return trainer
    except Exception as e:
        logger.error(f"Training error: {e}")
        logger.error(traceback.format_exc())
        
        try:
            logger.info("Attempting to save checkpoint despite error...")
            trainer.save_model(os.path.join(args.output_dir, "checkpoint-error"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "checkpoint-error"))
        except:
            logger.error("Could not save error checkpoint")
        
        raise

def main():
    """Main entry point with robust error handling."""
    
    parser = argparse.ArgumentParser(description="Train Llama-3 on insurance QA data")
    
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B",
                       help="Model to fine-tune (default: meta-llama/Llama-3.2-1B)")
    parser.add_argument("--output_dir", type=str, default="./llama3-insurance-model",
                       help="Directory to save model checkpoints")
    
    # Data arguments
    parser.add_argument("--max_examples", type=int, default=50,
                       help="Maximum number of examples to process")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Training batch size")
    parser.add_argument("--grad_accum", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Mac-specific arguments @vinit try this for non mac envs also
    parser.add_argument("--disable_fp16", action="store_true",
                       help="Disable mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing to save memory")
    
    args = parser.parse_args()
    
    # Main try-except block for the entire training pipeline
    try:
        # Set random seed
        set_seed(args.seed)
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args)
        
        # Prepare dataset
        train_dataset, eval_dataset = prepare_insurance_qa_dataset(tokenizer, args)
        
        # Train model
        trainer = train_model(model, tokenizer, train_dataset, eval_dataset, args)
        
        logger.info("Training completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())
        
        if "memory" in str(e).lower() or "cuda" in str(e).lower() or "mps" in str(e).lower():
            logger.error("This appears to be a memory-related error. Try reducing batch size, sequence length, or model size.")
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)