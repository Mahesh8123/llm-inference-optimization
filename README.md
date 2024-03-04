# llm-inference-optimization
Problem Statement: Large Language Models (LLMs) are powerful tools for Natural Language Processing (NLP) tasks but often come with high computational costs. The task is to design and develop a script capable of optimizing and running an LLM based on the GPT2LMHeadModel architecture.




# LLM Inference Optimization

## Overview
This script is designed to optimize and run a Large Language Model (LLM) based on the MistralForCausalLM architecture. It accepts a Hugging Face model path as input, optimizes the model for faster inference, waits for user input for a prompt, runs the model on the prompt, and outputs the model response along with the inference time.

## Requirements
- Python 3.x
- PyTorch
- Transformers library from Hugging Face

## Usage
1. Clone the repository to your local machine:
   
   git clone <repository_url>
  

2. Navigate to the directory containing the script:

   cd LLM-Inference-Optimization
   

3. Install the required dependencies:

   pip install torch transformers


4. Run the script:
   ```bash
   python inference_optimization.py
   ```

5. Follow the on-screen prompts to enter the Hugging Face model path and the desired prompt.

6. The script will then run the model on the prompt, display the model response, and provide the inference time.

## Additional Notes
- Ensure that you have access to a GPU with specifications compatible with the script requirements.
- Depending on the model size and GPU capabilities, the optimization process and inference time may vary.
- Feel free to modify the script as needed for your specific use case or to experiment with different optimization techniques.
