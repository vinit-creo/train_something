import datetime
import json
import os
import random
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import (
    LoraConfig, 
    prepare_model_for_kbit_training,
    get_peft_model,
    TaskType
)
import bitsandbytes as bnb
from tqdm import tqdm
import argparse
from datasets import load_dataset
import logging
from datasets import load_dataset

# Configure memory settings for Mac M1/M2
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  
torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',)
logger = logging.getLogger(__name__)



## make sepereate files to parse the logs in the taring file


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuneing LLaMA for insurance document QA")
    
    # Model arguments: know thew meaning of each values present here @vinit
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B",
                        help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="./llama-insurance-model",
                        help="Directory to save model checkpoints")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=512, 
                        help="Maximum sequence length for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    
    # Data arguments
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Maximum number of training samples")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Maximum number of evaluation samples")
    parser.add_argument("--use_insurance_qa", action="store_true", 
                        help="Use the Insurance QA dataset")
    parser.add_argument("--docs_dir", type=str, default=None,
                        help="Directory containing insurance documents")
    
    # Evaluation arguments
    parser.add_argument("--eval_only", action="store_true", 
                        help="Only run evaluation on a trained model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint to load for evaluation")
    parser.add_argument("--cpu_only", action="store_true", help="Force CPU-only training. We broke homie !")

    
    return parser.parse_args()


def prepare_insurance_qa_dataset(tokenizer, args):
    """Prepare the Insurance QA dataset for fine-tuning."""
    logger.info("Loading Insurance QA dataset...")
    try:
        dataset = load_dataset("rvpierre/insurance-qa-en")
        
        # Print dataset structure
        logger.info(f"Dataset structure: {dataset.keys()}")
        for split in dataset.keys():
            logger.info(f"{split} split has {len(dataset[split])} examples")
        
        # Process examples
        train_data = []
        eval_data = []
        
        # Process train split
        logger.info("Processing training data...")
        for i, example in enumerate(tqdm(dataset["train"])):
            if args.max_train_samples and i >= args.max_train_samples:
                break
            
            question = example["question_en"]
            
            # Get answers (may be indices or strings)
            if "answers" in example and example["answers"] and len(example["answers"]) > 0:
                if isinstance(example["answers"][0], int):
                    # Look up answers in the pool
                    if "answers_pool" in dataset["train"].features:
                        answers_pool = dataset["train"].features["answers_pool"]
                        # Get the first answer only for now
                        answer_idx = example["answers"][0]
                        if 0 <= answer_idx < len(answers_pool):
                            answer = answers_pool[answer_idx]
                        else:
                            continue  # Skip if invalid index
                    else:
                        continue  # Skip if no answer pool
                else:
                    # Direct string answers
                    answer = example["answers"][0]
            else:
                continue  # Skip examples without answers
            
            # Format for instruction fine-tuning
            instruction = "You are an insurance expert. Answer the following insurance-related question:"
            formatted_example = {
                "instruction": instruction,
                "input": question,
                "output": answer
            }
            
            train_data.append(formatted_example)
        
        # Process validation or test split for evaluation
        eval_split = "validation" if "validation" in dataset else "test"
        logger.info(f"Processing evaluation data from {eval_split} split...")
        
        for i, example in enumerate(tqdm(dataset[eval_split])):
            if args.max_eval_samples and i >= args.max_eval_samples:
                break
            
            question = example["question_en"]
            
            # Get answers (similar to train processing)
            if "answers" in example and example["answers"] and len(example["answers"]) > 0:
                if isinstance(example["answers"][0], int):
                    if "answers_pool" in dataset[eval_split].features:
                        answers_pool = dataset[eval_split].features["answers_pool"]
                        answer_idx = example["answers"][0]
                        if 0 <= answer_idx < len(answers_pool):
                            answer = answers_pool[answer_idx]
                        else:
                            continue
                    else:
                        continue
                else:
                    answer = example["answers"][0]
            else:
                continue
            
            # Format for evaluation
            instruction = "You are an insurance expert. Answer the following insurance-related question:"
            formatted_example = {
                "instruction": instruction,
                "input": question,
                "output": answer
            }
            
            eval_data.append(formatted_example)
        
        logger.info(f"Created {len(train_data)} training examples and {len(eval_data)} evaluation examples")
        
        # Check if we have any examples at all
        if len(train_data) == 0:
            logger.warning("No training examples were created! Check dataset format and filtering conditions.")
            # Create at least one dummy example to prevent tokenization errors
            train_data.append({
                "instruction": "You are an insurance expert. Answer the following insurance-related question:",
                "input": "What is insurance?",
                "output": "Insurance is a contract represented by a policy in which an individual or entity receives financial protection or reimbursement against losses from an insurance company."
            })
        
        if len(eval_data) == 0:
            logger.warning("No evaluation examples were created! Check dataset format and filtering conditions.")
            # Create at least one dummy example
            eval_data.append({
                "instruction": "You are an insurance expert. Answer the following insurance-related question:",
                "input": "What is a premium?",
                "output": "A premium is the amount of money an individual or business pays for an insurance policy."
            })
        
        # Format data for tokenization
        def format_for_tokenization(examples):
            texts = []
            for ex in examples:
                if ex["input"]:
                    text = f"### Instruction: {ex['instruction']}\n\n### Input: {ex['input']}\n\n### Response: {ex['output']}"
                else:
                    text = f"### Instruction: {ex['instruction']}\n\n### Response: {ex['output']}"
                texts.append(text)
            return texts
        
        # Tokenize function with error handling
        def tokenize_function(texts):
            if not texts:
                logger.error("Attempted to tokenize empty text list!")
                raise ValueError("Cannot tokenize empty text list")
                
            try:
                tokenized = tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=args.max_seq_length,
                    return_tensors="pt",
                )
                return tokenized
            except Exception as e:
                logger.error(f"Tokenization error: {e}")
                # Print some sample texts for debugging
                for i, text in enumerate(texts[:3]):
                    logger.error(f"Sample text {i}: {text[:100]}...")
                raise
        
        # Tokenize datasets
        train_texts = format_for_tokenization(train_data)
        eval_texts = format_for_tokenization(eval_data)
        
        logger.info(f"Tokenizing {len(train_texts)} training texts and {len(eval_texts)} evaluation texts")
        
        train_tokenized = tokenize_function(train_texts)
        eval_tokenized = tokenize_function(eval_texts)
        
        # Convert to HF datasets
        from datasets import Dataset
        
        train_dataset = Dataset.from_dict({
            "input_ids": train_tokenized["input_ids"],
            "attention_mask": train_tokenized["attention_mask"],
            "labels": train_tokenized["input_ids"].clone(),  # For causal LM, labels are the same as input_ids
        })
        
        eval_dataset = Dataset.from_dict({
            "input_ids": eval_tokenized["input_ids"],
            "attention_mask": eval_tokenized["attention_mask"],
            "labels": eval_tokenized["input_ids"].clone(),
        })
        
        return train_dataset, eval_dataset, train_data, eval_data
        
    except Exception as e:
        logger.error(f"Error preparing dataset: {e}")
        logger.error(traceback.format_exc())
        
        # Create fallback dummy datasets to allow the program to continue
        logger.warning("Creating fallback dummy datasets")
        
        # Create one dummy example
        dummy_text = "### Instruction: You are an insurance expert. Answer the following question.\n\n### Input: What is insurance?\n\n### Response: Insurance is a contract represented by a policy in which an individual or entity receives financial protection or reimbursement against losses from an insurance company."
        
        # Tokenize the dummy example
        dummy_tokenized = tokenizer(
            [dummy_text],
            padding="max_length",
            truncation=True,
            max_length=args.max_seq_length,
            return_tensors="pt",
        )
        
        # Create dummy datasets
        from datasets import Dataset
        
        dummy_dataset = Dataset.from_dict({
            "input_ids": dummy_tokenized["input_ids"],
            "attention_mask": dummy_tokenized["attention_mask"],
            "labels": dummy_tokenized["input_ids"].clone(),
        })
        
        dummy_data = [{
            "instruction": "You are an insurance expert. Answer the following question.",
            "input": "What is insurance?",
            "output": "Insurance is a contract represented by a policy in which an individual or entity receives financial protection or reimbursement against losses from an insurance company."
        }]
        
        return dummy_dataset, dummy_dataset, dummy_data, dummy_data

def setup_model_and_tokenizer(args):
    """Set up the model and tokenizer for training."""
    logger.info(f"Loading base model: {args.base_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Added padding token: {tokenizer.pad_token}")

    if torch.cuda.is_available() and not args.cpu_only:
        try:
            import bitsandbytes as bnb   
            bnb_config = bnb.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,)
            
            model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",)
    
            model = prepare_model_for_kbit_training(model)
            logger.info("Using quantized 4-bit model with GPU acceleration")
        except Exception as e:
            logger.warning(f"Error setting up quantized model: {e}")
            logger.info("Falling back to standard model loading")
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
    else:
        # CPU-only training
        logger.info("Using CPU for training (this will be slow)")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="cpu",
        )
    

    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    
    model = get_peft_model(model, lora_config)
    
    # Log trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer


def train(args, model, tokenizer, train_dataset, eval_dataset):
    """Train the model."""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"insurance-llama-{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        fp16=True,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        report_to="tensorboard",
    )
    
    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training arguments
    with open(os.path.join(output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    return output_dir

def evaluate(args, model, tokenizer, eval_data):
    """Evaluate the model on test examples."""
    logger.info("Running evaluation...")
    
    # Set the model to evaluation mode
    model.eval()
    
    results = {
        "total": len(eval_data),
        "correct": 0,
        "examples": []
    }
    
    # Evaluate on a sample of examples
    for idx, example in enumerate(tqdm(eval_data[:100])):  # Limit to 100 examples for speed
        question = example["input"]
        correct_answer = example["output"]
        
        # Format prompt
        if example["instruction"]:
            prompt = f"### Instruction: {example['instruction']}\n\n### Input: {question}\n\n### Response:"
        else:
            prompt = f"### Input: {question}\n\n### Response:"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response (after "### Response:")
        if "### Response:" in generated_text:
            model_answer = generated_text.split("### Response:")[-1].strip()
        else:
            model_answer = generated_text
        
        # Simple exact match evaluation - could be improved
        # with semantic similarity or other metrics
        is_correct = model_answer.lower() == correct_answer.lower()
        if is_correct:
            results["correct"] += 1
        
        # Store result
        results["examples"].append({
            "question": question,
            "correct_answer": correct_answer,
            "model_answer": model_answer,
            "is_correct": is_correct
        })
        
        # Log some examples
        if idx < 5:
            logger.info(f"\nQuestion: {question}")
            logger.info(f"Correct answer: {correct_answer}")
            logger.info(f"Model answer: {model_answer}")
            logger.info(f"Correct: {is_correct}")
    
    # Calculate overall metrics
    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
    
    logger.info(f"\nEvaluation results:")
    logger.info(f"Total examples: {results['total']}")
    logger.info(f"Correct: {results['correct']}")
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    
    # Save results
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    return results

"""
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
"""

def set_seed(seed):
    """Set random seed for reproducibility across Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Optional: for complete reproducibility, set deterministic operations
    # Note: this can impact performance
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")
    
    
    
def main():
    args = parse_args()
    
    set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.eval_only and args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        model, tokenizer = AutoModelForCausalLM.from_pretrained(args.checkpoint), AutoTokenizer.from_pretrained(args.checkpoint)
    else:
        model, tokenizer = setup_model_and_tokenizer(args)
    
    train_dataset, eval_dataset, train_data, eval_data = prepare_insurance_qa_dataset(tokenizer, args)
    
    if not args.eval_only:
        output_dir = train(args, model, tokenizer, train_dataset, eval_dataset)
        logger.info(f"Training completed. Model saved to {output_dir}")
    
    results = evaluate(args, model, tokenizer, eval_data)
    logger.info("Evaluation completed")
    
if __name__ == "__main__":
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        main()
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        logger.info("Attempting to train on CPU...")
        os.environ['PYTORCH_DEVICE'] = 'cpu'
        main()