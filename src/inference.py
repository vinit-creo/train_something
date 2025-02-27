import os
import argparse
import logging
import torch
import time
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def get_device():
    """Get the most appropriate device for inference."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU (inference may be slow)")
    return device

def load_model(model_path, device_type="auto"):
    """
    Load the model and tokenizer from the specified path.
    
    Args:
        model_path: Path to the model directory
        device_type: Type of device to use ("auto", "cpu", "cuda", "mps")
        
    Returns:
        model, tokenizer: The loaded model and tokenizer
    """
    logger.info(f"Loading model from {model_path}")
    
    if device_type == "auto":
        device = get_device()
    else:
        device = torch.device(device_type)
        logger.info(f"Using specified device: {device}")
    
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise
    
    try:
        model = None
        
        try:
            # Method 1: With device mapping
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map={"": device}
            )
            logger.info("Model loaded with device mapping")
        except Exception as e1:
            logger.warning(f"Error loading model with device mapping: {e1}")
            
            try:
                # Method 2: Standard loading
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=dtype
                )
                model = model.to(device)
                logger.info("Model loaded with standard method")
            except Exception as e2:
                logger.error(f"Error loading model with standard method: {e2}")
                
                # Method 3: CPU fallback with no dtype
                logger.info("Trying CPU fallback...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="cpu"
                )
                logger.info("Model loaded on CPU as fallback")
        
        if model is None:
            raise ValueError("Failed to load model with all methods")
            
        return model, tokenizer, device
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def format_prompt(question, model_type="llama"):
    """
    Format the prompt according to the model's expected format.
    
    Args:
        question: The insurance question to ask
        model_type: Type of model format to use ("llama", "llama3", "general")
        
    Returns:
        Formatted prompt text
    """
    system_message = "You are an insurance expert that provides clear, accurate, and helpful information about insurance concepts, policies, and practices."
    
    if model_type == "llama3":
        if model_type == "llama3-alt":
            prompt = f"<s>system\n{system_message}\n\nuser\n{question}\n\nassistant\n"
        else:
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_message}\n\n<|start_header_id|>user<|end_header_id|>\n{question}\n\n<|start_header_id|>assistant<|end_header_id|>"
    elif model_type == "llama":
        prompt = f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{question} [/INST]"
    else:
        prompt = f"### Instruction: {system_message}\n\n### Input: {question}\n\n### Response:"
    
    return prompt

def generate_response(model, tokenizer, question, device, args):
    """
    Generate a response to the given question using the loaded model.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        question: The question to answer
        device: The device to use
        args: Command-line arguments
        
    Returns:
        The generated answer text
    """
    prompt = format_prompt(question, args.model_type)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    generation_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            generation_config=generation_config
        )
    
    end_time = time.time()
    generation_time = end_time - start_time
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if args.verbose:
        logger.info(f"Raw generated text: {generated_text}")
    
    if args.model_type == "llama3":
        if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
            answer = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[1].strip()
            if "<|end_of_text|>" in answer:
                answer = answer.split("<|end_of_text|>")[0].strip()
        elif "system" in generated_text and "assistant" in generated_text:
            parts = generated_text.split("assistant")
            if len(parts) > 1:
                answer = parts[1].strip()
                answer = answer.replace("user", "").replace("system", "")
                answer = ''.join(c for c in answer if c.isprintable() and c != '!')
            else:
                answer = "I apologize, but I couldn't generate a proper response to your question about cosigning."
        else:
            answer = generated_text.strip()
    elif args.model_type == "llama":
        if "[/INST]" in generated_text:
            answer = generated_text.split("[/INST]")[1].strip()
        else:
            answer = generated_text.strip()
    else:
        if "### Response:" in generated_text:
            answer = generated_text.split("### Response:")[1].strip()
        else:
            answer = generated_text.strip()
    
    markers = ["system", "user", "assistant", "<|start_header_id|>", "<|end_header_id|>", "<|begin_of_text|>", "<|end_of_text|>"]
    for marker in markers:
        answer = answer.replace(marker, "")
    
    while "!!" in answer:
        answer = answer.replace("!!", "!")
    
    answer = " ".join(answer.split())
    
    if args.verbose:
        num_input_tokens = inputs.input_ids.shape[1]
        num_output_tokens = outputs.shape[1] - num_input_tokens
        tokens_per_second = num_output_tokens / generation_time if generation_time > 0 else 0
        
        logger.info(f"Generated {num_output_tokens} tokens in {generation_time:.2f}s ({tokens_per_second:.2f} tokens/s)")
    
    return answer

def interactive_mode(model, tokenizer, device, args):
    """
    Run in interactive mode, allowing the user to ask multiple questions.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        device: The device to use
        args: Command-line arguments
    """
    print("\n===== Insurance LLaMA Interactive Mode =====")
    print("Type 'exit', 'quit', or 'q' to end the session")
    print("Type 'verbose' to toggle verbose mode")
    print("================================================\n")
    
    history = []
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['exit', 'quit', 'q']:
            print("\nExiting interactive mode. Goodbye!")
            break
            
        if question.lower() == 'verbose':
            args.verbose = not args.verbose
            print(f"Verbose mode {'enabled' if args.verbose else 'disabled'}")
            continue
            
        if not question:
            continue
            
        try:
            print("\nThinking...")
            answer = generate_response(model, tokenizer, question, device, args)
            print(f"\nAnswer: {answer}")
            
            history.append({"question": question, "answer": answer})
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            print("\nSorry, I encountered an error while generating a response.")
    
    if history and input("\nSave conversation history? (y/n): ").lower() == 'y':
        filename = f"conversation_history_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w') as f:
            for i, entry in enumerate(history, 1):
                f.write(f"Q{i}: {entry['question']}\n\n")
                f.write(f"A{i}: {entry['answer']}\n\n")
                f.write("-" * 50 + "\n\n")
        print(f"Conversation history saved to {filename}")

def process_file(model, tokenizer, device, args):
    """
    Process questions from a file, one per line.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        device: The device to use
        args: Command-line arguments
    """
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        return
        
    logger.info(f"Processing questions from {args.input_file}")
    
    with open(args.input_file, 'r') as f:
        questions = [line.strip() for line in f if line.strip()]
        
    output_file = args.output_file or f"answers_{os.path.basename(args.input_file)}"
    
    with open(output_file, 'w') as f:
        for i, question in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{len(questions)}")
            
            try:
                answer = generate_response(model, tokenizer, question, device, args)
                
                f.write(f"Q: {question}\n\n")
                f.write(f"A: {answer}\n\n")
                f.write("-" * 50 + "\n\n")
                
            except Exception as e:
                logger.error(f"Error generating response for question {i}: {e}")
                f.write(f"Q: {question}\n\n")
                f.write(f"A: [Error generating response]\n\n")
                f.write("-" * 50 + "\n\n")
    
    logger.info(f"Responses written to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned Insurance LLaMA model")
    
    parser.add_argument("--model_path", type=str, default="./llama3-insurance-model",
                       help="Path to the fine-tuned model")
    parser.add_argument("--model_type", type=str, default="llama3", 
                      choices=["llama", "llama3", "llama3-alt", "general"],
                       help="Type of model format to use")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"],
                       help="Device to use for inference")
    
    parser.add_argument("--max_tokens", type=int, default=256,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for sampling (0 for greedy)")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                       help="Repetition penalty")
    
    parser.add_argument("--question", type=str, default=None,
                       help="Single question to answer (non-interactive mode)")
    parser.add_argument("--input_file", type=str, default=None,
                       help="File containing questions to answer (one per line)")
    parser.add_argument("--output_file", type=str, default=None,
                       help="File to write answers to (when using input_file)")
    
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--skip_special_tokens", action="store_true", default=True,
                       help="Skip special tokens in output (default: True)")
    
    args = parser.parse_args()
    
    try:
        model, tokenizer, device = load_model(args.model_path, args.device)
        
        if args.input_file:
            process_file(model, tokenizer, device, args)
        elif args.question:
            answer = generate_response(model, tokenizer, args.question, device, args)
            print(f"Q: {args.question}\n")
            print(f"A: {answer}")
        else:
            interactive_mode(model, tokenizer, device, args)
            
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        logger.error(f"{''.join(traceback.format_exception(type(e), e, e.__traceback__))}")
        return 1
        
    return 0

if __name__ == "__main__":
    import traceback
    
    try:
        exit_code = main()
        exit(exit_code)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        traceback.print_exc()
        exit(1)