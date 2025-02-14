## Imports
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

## Keep in mind that mixtral is a fairly large model for most laptops and requires ~25+ GB RAM,
## so if you need a smaller model,
## try using one like llama-13b-chat-gguf (model_name="TheBloke/Llama-2-13B-chat-GGUF"; model_file="llama-2-13b-chat.Q4_K_M.gguf")
## or mistral-7b-openorca-gguf (model_name="TheBloke/Mistral-7B-OpenOrca-GGUF"; model_file="mistral-7b-openorca.Q4_K_M.gguf").

## Download the GGUF model
model_name = "TheBloke/Llama-2-13B-chat-GGUF"
model_file = "llama-2-13b-chat.Q4_K_M.gguf"
model_path = hf_hub_download(model_name, filename=model_file)

## Instantiate model from downloaded file
llm = Llama(
    model_path=model_path,
    n_ctx=16000,  # Context length to use
    n_threads=12,            # Number of CPU threads to use
    n_gpu_layers=0        # Number of model layers to offload to GPU
)

## Generation kwargs
generation_kwargs = {
    "max_tokens":20000,
    "stop":["</s>"],
    "echo":False, # Echo the prompt in the output
    "top_k":1 # This is essentially greedy decoding, since the model will always return the highest-probability token. Set this value > 1 for sampling decoding
}

## Run inference
prompt = "The meaning of life is "
res = llm(prompt, **generation_kwargs) # Res is a dictionary

## Unpack and the generated text from the LLM response dictionary and print it
print(res["choices"][0]["text"])
