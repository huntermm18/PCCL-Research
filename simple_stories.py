import openai
import pandas as pd
import time
from foi_simple import *
from tqdm import tqdm
from keys import *
from _parameters import *

openai.api_key = keys['openAI']

def gen_backstory(pid, df):
    person = df.iloc[pid]
    id = person['ids']
    backstory = BACKSTORY_START

    for k in foi_keys:
        df_val = person[k]
        elem_template = fields_of_interest[k]['template']
        elem_map = fields_of_interest[k]['valmap']

        if len(elem_map) == 0:
            newval = str(df_val)
        elif df_val in elem_map:
            newval = elem_map[df_val]
        else:
            newval = str(df_val)

        newval = newval.replace("<1:[RECORD VERBATIM]>:", "")
        backstory += " " + elem_template.replace('XXX', newval)

    if backstory[0] == ' ':
        backstory = backstory[1:]

    return id, backstory


def do_query(prompt, max_tokens=MAX_TOKENS, engine="text-davinci-003"):
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
        top_p=1,
        logprobs=100,
    )
    return response.choices[0]['text']

if USE_TURBO:
    def do_query(prompt, max_tokens=MAX_TOKENS):
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            max_tokens=max_tokens,
            temperature=TEMPERATURE,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return res['choices'][0]['message']['content']


# ====================================================================================

# df = pd.read_csv("./NSYRdones.csv")
df = pd.read_csv(PEOPLE_CSV)
foi_keys = "age gender ethnicity".split()  # XXX income paredu parsedu religup

ids = []
prompt_ids = []
prompts = []
responses = []
ethnicities = []

for pid in range(len(df)):
    for i in range(ITERATIONS):
        prompt_ids.append(i)
        ethnicities.append(df.iloc[pid]['ethnicity'])
        id, prompt = gen_backstory(pid, df)
        prompt += BACKSTORY_END
        prompts.append(prompt)
        ids.append(id)

if USE_OPENAI:
    for prompt in tqdm(prompts):
        if PRINT_STORIES:
            print("---------------------------------------------------")
            print(prompt)

        done = False
        while not done:
            try:
                response = do_query(prompt, max_tokens=128)
                if PRINT_STORIES: print(response)              
                responses.append(response)
                done = True
            except:
                print('In except. Sleeping for 5...')
                time.sleep(5.0)

else: # use other model
    from languageModel import do_queries
    print(f'\nUsing {MODEL} language model... Passing in {len(prompts)} prompts')
    responses = do_queries(prompts, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)


# Output data to csv
newdf = pd.DataFrame({'ids': ids, 'pids': prompt_ids, 'prompt': prompts, 'response': responses, 'ethnicity': ethnicities})
newdf.to_csv(STORIES_OUTPUT_CSV)
# newdf.to_csv('./data/deleteme.csv')
print('simple_stories.py finished')

