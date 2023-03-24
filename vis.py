import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from _parameters import *
plt.ion()
print('Note: Make sure that the last run of strans.py was on the same input csv as this run of vis.py or the results will be mismatched...') 

df = pd.read_csv(STORIES_OUTPUT_CSV)

if STORIES_OUTPUT_CSV == 'data/racism_stores.csv': # Case where ethnicity is not in the csv (shouldn't be needed)
    ethnicities = []
    for prompt in df['prompt']:
        if 'black' in prompt.lower()[430:]:
            ethnicities.append('Black')
        elif 'white' in prompt.lower()[430:]:
            ethnicities.append('White')
        elif 'asian' in prompt.lower()[430:]:
            ethnicities.append('Asian')
        elif 'hispanic' in prompt.lower()[430:]:
            ethnicities.append('Hispanic')
        else:
            ethnicities.append('Other')
    df['ethnicity'] = ethnicities


sentences = []
for pid in tqdm(range(len(df))):
    person = df.iloc[pid]
    sentences.append(person['response'])

rid = 0
rids = {}
rrids = {}
rtexts = []
rcodes = []

for pid in tqdm(range(len(df))):
    person = df.iloc[pid]
    ethnicity = person[FEATURE_TO_SORT_ON]

    if type(ethnicity) != str:
        ethnicity = 'Other'
    ethnicity = ethnicity.strip().lower()
    rtexts.append(ethnicity)
    if not ethnicity in rids:
        rids[ethnicity] = rid
        rid += 1

    rcodes.append(rids[ethnicity])

rcodes = np.array(rcodes)

for k, v in rids.items():
    rrids[v] = k

projections = np.load("./data/simple_multipass_projections.npy")

plt.clf()

legend_entries = []
for r in range(rid):
    inds = rcodes == r
    legend_entries.append(rrids[r])
    plt.scatter(projections[inds, 0], projections[inds, 1], alpha=0.5)
plt.legend(legend_entries)






# ========================== get sentiment data ==========================
sentiments = {}

if SEMTIMENT_ANALISER == 'textblob':
    from textblob import TextBlob
    print('Using TextBlob for sentiment analysis')

    # Function to calculate sentiment polarity
    def sentiment_polarity(text, include_neutral=False):
        sentiment = TextBlob(text).sentiment.polarity
        if sentiment < 0:
            return "NEGATIVE"
        elif sentiment == 0 and include_neutral:
            return "NEUTRAL"
        else:
            return "POSITIVE"

    stories = {}
    sentiment_means = {}
    ethnicities = df['ethnicity'].unique()
    for e in ethnicities:
        stories[e] = df['response'].loc[(df['ethnicity'] == e)]
    for e in ethnicities:
        s = []
        sentiment_list = []
        for response in stories[e]:
            sentiment_list.append(sentiment_polarity(response))
            s.append(TextBlob(response).sentiment.polarity)
        sentiment_means[e] = np.mean(s)
        sentiments[e] = sentiment_list

elif SEMTIMENT_ANALISER == 'flair':
    from flair.data import Sentence
    from flair.nn import Classifier
    tagger = Classifier.load('sentiment')
    print('Using fliar for sentiment analysis')

    stories = {}
    sentiment_means = {}
    ethnicities = df['ethnicity'].unique()
    for e in ethnicities:
        stories[e] = df['response'].loc[(df['ethnicity'] == e)]
    for e in tqdm(ethnicities):
        s = []
        sentiment_list = []
        for response in stories[e]:
            sentence = Sentence(response)
            tagger.predict(sentence)
            polarity = sentence.score if sentence.tag == 'POSITIVE' else -sentence.score
            sentiment_list.append(sentence.tag)
            s.append(polarity)
        sentiment_means[e] = np.mean(s)
        sentiments[e] = sentiment_list

else:
    print('Unknown sentiment analyser')

# matplot lib bar chart
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages(f"./result_viz/{OUTPUT_FILE_NAMES['graphs-pdf']}")
data = sentiment_means
ind = np.arange(len(data))
fig = plt.figure()
plt.bar(ind, list(data.values()))
plt.xticks(ind, list(data.keys()))
plt.show()
pdf.savefig(fig)

# pie charts
for ethnicity in stories.keys():
#   sentiment_list = [sentiment_polarity(story) for story in stories[ethnicity]]
    sentiment_list = sentiments[ethnicity]

    sentiment_keys = ['POSITIVE', 'NEGATIVE']
    if len(sentiment_list) == 0:
        continue
    values = [sentiment_list.count('POSITIVE') / len(sentiment_list), sentiment_list.count('NEGATIVE') / len(sentiment_list)]

    # Plotting the results as a pie chart
    fig = plt.figure()
    plt.pie(values, labels=sentiment_keys, startangle=90, counterclock=False, autopct='%1.1f%%', shadow=True)
    plt.axis('equal')
    plt.title(f'Sentiment Analysis Results: {ethnicity}')
    plt.show()
    pdf.savefig(fig)

# save figures to pdf
pdf.close()
# ========================== End get sentiment data ==========================





# ========================== clustering ======================================

# kmeans
from sklearn.cluster import KMeans
def cluster_data(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(data)
    return kmeans.labels_
cluster_labels = cluster_data(projections, 5)

# dbscan
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.27, min_samples=2)
dbscan.fit(projections)
dbscan_cluster_labels = dbscan.labels_
n_clusters = len(set(dbscan_cluster_labels)) - (1 if -1 in dbscan_cluster_labels else 0)
print("Clusters found: ", n_clusters)

# Agglomerative
number_clusters = 12
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters=number_clusters).fit(projections)
agglomerative_cluster_labels = clustering.labels_

# ========================== End clustering ==================================






# ========================== cluster summaries ===============================

import openai
from keys import *
openai.api_key = keys['openAI']

def do_query(prompt, max_tokens=512, engine='text-davinci-003'):
    response = openai.Completion.create(engine=engine, prompt=prompt, temperature=0.6, max_tokens=max_tokens)
    return response.choices[0]['text']

summaries = []
prompts = []
for cluster_num in range(number_clusters):
    indexes = []
    for i in range(len(agglomerative_cluster_labels)):
        if agglomerative_cluster_labels[i] == cluster_num: indexes.append(i)
    responses = df.iloc[indexes]['response']
    responses = '\n\n'.join(responses)
    prompt = f'{responses[:7500]} \nThe best single label for all of the previous sentences that captures the single idea that best summarizes them is:"""'
    prompts.append(prompt)

if not SKIP_SUMMARIES:
    print('Summarizing clusters...')
    if USE_OPENAI:
        for prompt in prompts:
            summary = do_query(prompt, max_tokens=20)
            print(f'Summary: {summary.strip()}')
            summaries.append(summary.strip())
    else:
        from languageModel import do_queries
        print(f'\nUsing other language model... Passing in {len(prompts)} prompts')
        summaries = do_queries(prompts, max_tokens=512, temperature=TEMPERATURE)
        summaries = [s.strip() for s in summaries]


# ========================== end cluster summaries ======================





# ========================== plotly =====================================

import plotly.express as px

# add linebreaks
def split_string(string, parts=4):
    n = len(string)
    return [string[i*n//parts:(i+1)*n//parts] for i in range(parts)]


s1, s2, s3, s4 = [], [], [], []
for s in sentences:
    x = split_string(s, 4)
    s1.append(x[0])
    s2.append(x[1])
    s3.append(x[2])
    s4.append(x[3])

if NUM_DIMENTIONS == 3:
    fig = px.scatter_3d(
        projections,
        x=0, y=1, z=2,
        color=rtexts,
        color_discrete_sequence=px.colors.qualitative.Prism,
        # hover_name=np.array(sentences), # was long, so I broke into 4 parts
        hover_data=[s1, s2, s3, s4]
    )
    # fig.show()
    fig.write_html(f"./result_viz/{OUTPUT_FILE_NAMES['plotly_graph']}")
else:
    fig = px.scatter(
        projections,
        x=0, y=1,
        color=rtexts,
        color_discrete_sequence=px.colors.qualitative.Prism,
        hover_data=[s1, s2, s3, s4]
    )
    fig.write_html(f"./result_viz/{OUTPUT_FILE_NAMES['plotly_graph']}")


# color by cluster kmeans
if NUM_DIMENTIONS == 3:
    fig = px.scatter_3d(
        projections,
        x=0, y=1, z=2,
        color=cluster_labels,
        color_discrete_sequence=px.colors.qualitative.Prism,
        hover_name=rtexts,
        hover_data=[s1, s2, s3, s4]
    )
    fig.write_html(f"./result_viz/{OUTPUT_FILE_NAMES['plotly_graph_clusters']}")
else:
    fig = px.scatter(
        projections,
        x=0, y=1,
        color=cluster_labels,
        color_discrete_sequence=px.colors.qualitative.Prism,
        hover_name=rtexts,
        hover_data=[s1, s2, s3, s4]
    )
    fig.write_html(f"./result_viz/{OUTPUT_FILE_NAMES['plotly_graph_clusters']}")


# color by cluster dbscan
if NUM_DIMENTIONS == 3:
    fig = px.scatter_3d(
        projections,
        x=0, y=1, z=2,
        color=dbscan_cluster_labels,
        color_discrete_sequence=px.colors.qualitative.Prism,
        hover_name=rtexts,
        hover_data=[s1, s2, s3, s4]
    )
    fig.write_html(f"./result_viz/{OUTPUT_FILE_NAMES['plotly_graph_clusters_dbscan']}")
else:
    fig = px.scatter(
        projections,
        x=0, y=1,
        color=dbscan_cluster_labels,
        color_discrete_sequence=px.colors.qualitative.Prism,
        hover_name=rtexts,
        hover_data=[s1, s2, s3, s4]
    )
    fig.write_html(f"./result_viz/{OUTPUT_FILE_NAMES['plotly_graph_clusters_dbscan']}")

# color by cluster summaries
if not SKIP_SUMMARIES:
    if NUM_DIMENTIONS == 3:
        point_summaries = [summaries[i] for i in agglomerative_cluster_labels]
        fig = px.scatter_3d(
            projections,
            x=0, y=1, z=2,
            color=point_summaries,
            color_discrete_sequence=px.colors.qualitative.Prism,
            hover_name=rtexts,
            hover_data=[s1, s2, s3, s4]
        )
        fig.write_html(f"./result_viz/{OUTPUT_FILE_NAMES['plotly_graph_clusters_summaries']}")
    else:
        point_summaries = [summaries[i] for i in agglomerative_cluster_labels]
        fig = px.scatter(
            projections,
            x=0, y=1,
            color=point_summaries,
            color_discrete_sequence=px.colors.qualitative.Prism,
            hover_name=rtexts,
            hover_data=[s1, s2, s3, s4]
        )
        fig.write_html(f"./result_viz/{OUTPUT_FILE_NAMES['plotly_graph_clusters_summaries']}")

# ========================== End plotly =====================================

print('finished')
