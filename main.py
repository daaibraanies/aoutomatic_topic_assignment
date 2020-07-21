import pandas as pd
import numpy as np
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from rake_nltk import Rake
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import collections
import warnings
warnings.simplefilter("ignore")
pd.set_option('display.max_columns',25)

df = pd.read_csv('dataset.csv',header = None)

significant_frequency = 1
stop_words = stopwords.words('english')
stop_words.extend(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l','m','n','o','p','q','r','s','t', 'u', 'v', 'w', 'x', 'y', 'z', "about", "across", "after", "all", "also", "an", "and", "another", "added",
"any", "are", "as", "at", "basically", "be", "because", 'become', "been", "before", "being", "between","both", "but", "by","came","can","come","could","did","do","does","each","else","every","either","especially", "for","from","get","given","gets",
'give','gives',"got","goes","had","has","have","he","her","here","him","himself","his","how","if","in","into","is","it","its","just","lands","like","make","making", "made", "many","may","me","might","more","most","much","must","my","never","provide",
"provides", "perhaps","no","now","of","on","only","or","other", "our","out","over","re","said","same","see","should","since","so","some","still","such","seeing", "see", "take","than","that","the","their","them","then","there",
"these","they","this","those","through","to","too","under","up","use","using","used", "underway", "very","want","was","way","we","well","were","what","when","where","which","while","whilst","who","will","with","would","you","your",
'etc', 'via', 'eg'])
#Uncomment if wordnet is absent on your computer
#nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
vectorizer = CountVectorizer(analyzer='word',ngram_range=(1,2))

vectors = []
topic_words = []
duplicate_words = []
topic_words_processed = []
results = []

df['std_text'] = df[0].str.lower().str.replace(r'^([\w]\s)*', ' ')\
                .str.replace(r'[\s]+',' ')\
                .str.replace(r'[.0-9-:;,@#$%^&*()\\\/\.~\!\?\[\]]','')\
                .str.strip()
df['std_text'] = df['std_text'].apply(lambda x: nltk.word_tokenize(x))
df['std_text'] = df['std_text'].apply(lambda x:[word for word in x if word not in stop_words])
df['std_text'] = df['std_text'].apply(lambda x:[lemmatizer.lemmatize(word) for word in x])

for i,row in df.iterrows():
    vectors.append(", ".join(row['std_text']))

vectorized = vectorizer.fit_transform(vectors)
lda_model = LatentDirichletAllocation(n_components=10,
                                       random_state=1,
                                       evaluate_every=-1)
lda_output = lda_model.fit_transform(vectorized)
df_document_topic = pd.DataFrame(np.round(lda_output,2),
                                 columns=["Topic "+str(i) for i in range(lda_model.n_components)])

dominant_topic = (np.argmax(df_document_topic.values,axis=1))
df_document_topic['dominant_topic'] = dominant_topic
df = pd.merge(df,df_document_topic,left_index=True,right_index=True,how='outer')

keywords = np.array(vectorizer.get_feature_names())
for topic_weights in lda_model.components_:
    top_keywords_locs = (-topic_weights).argsort()[:20]
    topic_words.append(keywords.take(top_keywords_locs))

duplicate_words = [word for topic in topic_words for word in topic]
duplicate_words = [item for item, count in collections.Counter(duplicate_words).items() if count > 1]

#perfhaps would have been better to use top N TF-IDF instead
#word-isolation still works
for topic in topic_words:
    single_appearance = [word for word in topic if word not in duplicate_words]
    if single_appearance == []:
        pass
    topic_words_processed.append(single_appearance)

df_topic_keywords = pd.DataFrame(topic_words_processed)
df_topic_keywords.columns = ['Term '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords['topic_keywords'] = df_topic_keywords.values.tolist()
df_topic_keywords['topic_number'] = df_topic_keywords.index
df_topic_keywords = df_topic_keywords[['topic_keywords','topic_number']]
df_topic_keywords['topic_keywords'] = [[word for word in keyword_lsit if word] for keyword_lsit in df_topic_keywords['topic_keywords']]

df = pd.merge(df,df_topic_keywords,left_on='dominant_topic',right_on='topic_number')
del df_topic_keywords
print(df.head())

for topic in df['dominant_topic'].unique():
    key_words = []
    tmp_keywords = []
    tmp = []
    current_topic = df[df['dominant_topic'] == topic]
    current_topic = current_topic.copy()
    current_topic[0] = current_topic[0].str.lower().str.replace(r'^([\w]\s)*', ' ') \
        .str.replace(r'[\s]+', ' ') \
        .str.replace(r'[.0-9-:;,@#$%^&*()\\\/\.~\!\?\[\]]', '') \
        .str.strip()

    for j in current_topic[0].values.tolist():
        rake = Rake()
        rake_extracted = rake.extract_keywords_from_text(j)
        ranked_phrases = rake.get_ranked_phrases_with_scores()
        key_words = [word for word in ranked_phrases if word not in key_words]

    key_words = pd.DataFrame(key_words,columns=['score','term'])
    key_words = key_words.sort_values('score',ascending=False)
    key_words = key_words.drop_duplicates(subset=['term'])
    key_words['topic_number'] = topic
    key_words['term_list'] = key_words['term'].apply(lambda x:x.split())

    for j in key_words.values.tolist():
        tmp = []
        bigrams = ngrams(j[3],2)

        for g in bigrams:
            tmp.append(' '.join(g))

        for word in j[3]:
            tmp.append(word)
            tmp.append(lemmatizer.lemmatize(word))

        j.remove(j[3])
        j.append(list(set(tmp)))
        tmp_keywords.append(j)

    key_words = pd.DataFrame(tmp_keywords,columns=['score','term','topic_number','term_list'])
    topic_words = current_topic['std_text'].values.tolist()
    topic_words = [word for sublist in topic_words for word in sublist]
    topic_words = list(set(topic_words))

    tmp = []
    for topic in topic_words:
        mask = key_words['term_list'].apply(lambda x: topic in x)
        key_word_processed = key_words[mask]

        if key_word_processed.empty:
            pass
        else:
            for j in key_word_processed[['score','term','topic_number']].values.tolist():
                if j not in tmp:
                    tmp.append(j)


    key_words = pd.DataFrame(tmp,columns=['score','term','topic_number'])
    top_key_words = key_words[key_words.score == key_words['score'].max()]
    remaining_keywords = key_words[key_words.score != key_words['score'].max()]
    top_key_words = top_key_words.copy()
    top_key_words = top_key_words.groupby(['score', 'topic_number']).agg({'term': lambda x: ' / '.join(map(str, x))})
    top_key_words = top_key_words.reset_index()
    top_key_words['parent'] = ''
    remaining_keywords = remaining_keywords.copy()
    remaining_keywords['topic_number'] = remaining_keywords['topic_number'] + 0.1
    remaining_keywords['parent'] = top_key_words['term'].values.tolist()[0]
    all_topics = pd.concat([top_key_words, remaining_keywords], sort=False)

    for t in all_topics.to_dict(orient='records'):
        results.append(t)

results.append({'score': 1000, 'topic_number': 0.0, 'term': '', 'parent': ''})

all_topics_df = pd.DataFrame(results)
all_topics_df = all_topics_df.sort_values('topic_number', ascending=True)
all_topics_df = all_topics_df.loc[all_topics_df['score'] > 0]

print(all_topics_df.head())

fig = go.Figure(go.Sunburst(
    labels = all_topics_df['term'].values.tolist(),
    parents = all_topics_df['parent'].values.tolist(),
    values = all_topics_df['score'].values.tolist()
))

fig.update_layout(margin = dict(t = 0, l = 0,  r = 0, b = 0))
fig.write_html("topic_graph.html")
fig.show()
