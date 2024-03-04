import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time


def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    return model, tokenizer

# Tokenize input prompt
def tokenize_input(tokenizer, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    return input_ids

# Run inference with warm-up
def run_inference(model, input_ids, num_tokens=128, max_length=128, temperature=1.0, top_k=50):
    model.eval()
    with torch.no_grad():
        # Set pad token id explicitly
        model.config.pad_token_id = model.config.eos_token_id

        # Generate attention mask
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)

        # Warm-up
        _ = model.generate(input_ids,
                           attention_mask=attention_mask,
                           max_length=input_ids.size(1) + num_tokens)

        # Start inference and measure time
        start_time = time.time()
        output = model.generate(input_ids,
                                attention_mask=attention_mask,
                                max_length=max_length,
                                temperature=temperature,
                                top_k=top_k,
                                no_repeat_ngram_size=2)  # Set no_repeat_ngram_size to avoid repeated n-grams
        end_time = time.time()
        inference_time = end_time - start_time

        return output, inference_time

# Main function
def main():
    model_path = input("Enter the Hugging Face model path: ")
    prompt = input("Enter the prompt: ")

    # Load the model
    model, tokenizer = load_model(model_path)

    # Tokenize input
    input_ids = tokenize_input(tokenizer, prompt)

    # Run inference
    output, inference_time = run_inference(model, input_ids)

    # Decode and print the output
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Model response:", decoded_output)
    print("Inference time:", inference_time)

if __name__ == "__main__":
    main()
