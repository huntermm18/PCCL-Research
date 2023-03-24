import requests
from _parameters import *

# MODELS = ['bigscience/bloom-7b1', 'google/flan-t5-xxl', 'gpt2-xl', 'facebook/opt-6.7b', 'EleutherAI/gpt-j-6B', 'LLAMA65', 'LLAMA7', 'LLAMA13']


def do_queries(prompt_list, max_tokens=512, temperature=.8):
    response = requests.post(f'http://172.17.0.1:{PORT}/LLModels/', json={"model":MODEL, "prompt": prompt_list, "num_tokens": max_tokens, "temperature": temperature})
    try:
        return response.json() # returns a list of completions
    except:
        return response





# =========================================================

# server commands:
# screen -x 2795402.pts-20.sivri

# req = ["In order to cook the pizza you need \n 1 cup of cheese\n 1 cup of tomatoe sauce", "My favorite color is"]

# # Available models are bloom, flan, gpt2, facebook, or eleuther
# # You can change the num_tokens or temperature if needed
# response = requests.post(' http://172.17.0.1:8000/LLModels/',json={"model":MODEL, "prompt": req[0], "num_tokens": 40,"temperature": .7})
# # It returns two lists
# # The first list is the prompt with the generated text
# # The second list is the generated text
# response = response.json()
# prompt_with_response = response[0]
# only_response = response[1]
# for r in prompt_with_response:
#     print(r)
