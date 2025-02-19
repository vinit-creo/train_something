import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class PromptInference:
    def __init__(self, model_path="./prompt_llm_model"):
        """Initialize with the trained model"""
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        self.model.eval()
        
    def generate_response(self, act, prompt, max_length=200):
        """Generate a response for a given prompt"""
        input_text = f"Act as {act}\n\nPrompt: {prompt}\n\nResponse:"
        
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(input_text, "").strip()
        return response

def main():
    print("Initializing model (this might take a moment)...")
    model = PromptInference()
    
    print("\nPrompt Interface (type 'exit' to quit)")
    print("You can specify a role and prompt for the AI to respond to.")
    
    while True:
        print("\n" + "="*50)
        act = input("\nEnter the role (e.g., 'Life Coach', 'Accountant'): ")
        if act.lower() == 'exit':
            break
            
        prompt = input("Enter your prompt: ")
        if prompt.lower() == 'exit':
            break
        
        try:
            print("\nGenerating response...\n")
            response = model.generate_response(act, prompt)
            print("Response:", response)
        except Exception as e:
            print(f"Error generating response: {e}")

if __name__ == "__main__":
    main()