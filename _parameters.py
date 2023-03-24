

#                   PIPELINE PARAMETERS


# ========================== Global ==========================================

USE_OPENAI = True # If True use OpenAI APIs. If False use locally hosted models
USE_TURBO = True # If True use GPT-3.5 Turbo (Won't do anything if USE_OPENAI is False). If False use text-davinci-003
MAX_TOKENS = 512 # Max tokens for the model to generate per prompt
TEMPERATURE = 0.7 # Temperature for the model
STORIES_OUTPUT_CSV = "./data/results_simple_multipass_text-davinci-003.csv" # './data/racism_stores.csv'
NUM_DIMENTIONS = 2 # 2 or 3 (dimentions for the output graphs)




# ========================== Simple Stories Parameters ==============================================

ITERATIONS = 3 # Iterations per person
PRINT_STORIES = True # print the story to the console each time it is generated
PEOPLE_CSV = './data/fake-people.csv' # CSV file with the people data

BACKSTORY_START = """When asked about how they have experienced racism, a person who is"""
BACKSTORY_END = """ responded with: \""""

MODEL = 'LLAMA65' # if USE_OPENAI is False (see languageModel.py)
PORT = '8007' # if USE_OPENAI is False (port for local server with models)




# ============================ Vis ==================================================

SEMTIMENT_ANALISER = 'textblob' # 'textblob', 'fliar' (textblob is faster, but flair tends to be more accurate)
SKIP_SUMMARIES = False # Use GPT to generate summaries for the stories (skips this summary step if True)
FEATURE_TO_SORT_ON = 'ethnicity'
OUTPUT_FILE_NAMES  = { # Change the values not the keys
    'plotly_graph':'plotly-graph.html',
    'plotly_graph_clusters':'plotly-graph-clusters.html',
    'plotly_graph_clusters_dbscan':'plotly-graph-clusters-dbscan.html',
    'graphs-pdf':'output-graphs.pdf',
    'plotly_graph_clusters_summaries': 'plotly-graph-clusters-summaries.html'
}