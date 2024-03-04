# llm-inference-optimization
Problem Statement: Large Language Models (LLMs) are powerful tools for Natural Language Processing (NLP) tasks but often come with high computational costs. The task is to design and develop a script capable of optimizing and running an LLM based on the GPT2LMHeadModel architecture.


# Running LLM Inference Optimization Script on Google Colab

This script is designed to optimize and run a Large Language Model (LLM) using MistralForCausalLM architecture on Google Colab. It accepts a Hugging Face model path as input, optimizes the model for faster inference, waits for user input for a prompt, runs the model on the prompt, and outputs the model response along with performance metrics.

## How to Use on Google Colab

1. Open Google Colab: Visit [Google Colab](https://colab.research.google.com/) and sign in with your Google account.

2. **Create a New Notebook**: Click on "File" -> "New Notebook" to create a new Colab notebook.

   
3.Upload Script: Upload the provided Python script (`llm_inference_optimization.py`) to your Google Drive.

4. Install Required Libraries: Install the Transformers library from Hugging Face by running the following code cell:
   
   !pip install transformers
   

5. **Run the Script**: Execute the script by running the following code cell:
  
   !python /content/drive/MyDrive/llm_inference_optimization.py
 

6. Follow Instructions: Follow the prompts to enter the Hugging Face model path and the desired prompt.

8. View Output: The script will run the model on the prompt, display the model response, and provide performance metrics including inference time.

## Additional Notes

- Ensure that you have a stable internet connection and sufficient runtime on Google Colab to execute the script.
- Depending on the model size and Colab's GPU capabilities, the optimization process and inference time may vary.
- Feel free to modify the script as needed for your specific use case or to experiment with different optimization techniques.

