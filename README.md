## Getting Started
Add openAI key to keys.py

Set parameters in _parameters.py

## Pipeline
Run "simple_stories.py"

`python3 simple_stories.py`

Run "strans.py" in the docker container

```
sh docker-script.sh
python3 strans.py
exit
```

Run "vis.py" which generates the final plots and outputs them to the result_viz folder. The plotly charts will be HTML files

`python3 vis.py`


## Language Model
The language model that will be used in the pipeline is defined in _parameters.py

If USE_OPENAI == True then the model will be either text-davinci-003 or gpt-3.5-turbo (based on USE_TURBO)

If USE_OPENAI == False then instead of API calls to OpenAI the calls will be made to the local server on Sivri and will use the model from the MODEL variable


## Docker container
run `sh docker-script.sh` with the optional gpu number parameter after