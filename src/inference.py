import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MentalHealthCounselorBot:
    def __init__(self, model_path="./prompt_llm_model"):
        """Initialize the counseling model"""
        logger.info("Initializing Mental Health Counselor...")
        
        self.device = torch.device("cpu")  # Force CPU usage for stability
        logger.info(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
            
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def generate_response(self, context, user_input, max_length=200):
        """Generate a counseling response"""
        try:
            input_text = (
                f"Context: {context}\n"
                f"User: {user_input}\n"
                f"Assistant:"
            )
            
            encoded = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=100  
                
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=encoded['input_ids'],
                    max_length=max_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,  
                    num_beams=1,     
                    repetition_penalty=1.0 
                )
            
            response = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            response = response.replace(input_text, "").strip()
            
            return response if response else "I understand. Could you tell me more about that?"
            
        except Exception as e:
            logger.error(f"Error in generate_response: {e}", exc_info=True)
            return "I'm currently having trouble processing that. Could you try rephrasing?"

def main():
    print("\n" + "="*80)
    print("IMPORTANT: This is an experimental AI model and should NOT replace professional mental health care.")
    print("="*80 + "\n")
    
    try:
        print("Initializing counseling bot...")
        counselor = MentalHealthCounselorBot()
        
        context = "a mental health counselor who provides understanding and guidance"
        
        while True:
            print("\n" + "="*50)
            user_input = input("\nShare what's on your mind (or type 'exit' to quit): ").strip()
            
            if user_input.lower() == 'exit':
                print("\nTake care of yourself.")
                break
            
            if len(user_input) < 3:
                print("Please share a bit more about what you're feeling.")
                continue
            
            try:
                print("\nThinking...")
                response = counselor.generate_response(context, user_input)
                print("\nResponse:", response)
                    
            except Exception as e:
                print(f"\nError generating response. Please try again.")
                logger.error(f"Generation error: {e}", exc_info=True)
                
    except Exception as e:
        print(f"\nError: {e}")
        print("Please ensure the model is properly trained and saved.")
        sys.exit(1)

if __name__ == "__main__":
    main()