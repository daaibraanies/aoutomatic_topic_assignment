import pandas as pd
import numpy as np
import re
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
results = []

df['std_text'] = df[0].str.lower().str.replace(r'[^\w\s]', ' ').str.replace(r'\s+',' ').str.strip()
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

for i in topic_words:
    print(i)
#Not descriptive enough topics

df_topic_keywords = pd.DataFrame(topic_words)
df_topic_keywords.columns = ["Term "+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords['topic_keywords'] = df_topic_keywords.values.tolist()
df_topic_keywords['topic_number'] = df_topic_keywords.index
df_topic_keywords = df_topic_keywords[['topic_keywords','topic_number']]

tmp = []
for i in df_topic_keywords['topic_keywords']:
    tmp.append([x for x in i if x is not None])
df_topic_keywords['topic_keywords'] = tmp

df = pd.merge(df,df_topic_keywords,left_on='dominant_topic',right_on='topic_number')
del df['topic_number']

for i in df['dominant_topic'].unique():
    key_words = []
    tmp_kwd = []

    topic = df[df['dominant_topic']==i]
    topic = topic.copy()

    for j in topic[0].values.tolist():
        rake = Rake()
        kwd = rake.extract_keywords_from_text(j)
        phrase_ranks = rake.get_ranked_phrases_with_scores()

        for k in phrase_ranks:
            if k not in key_words:
                key_words.append(k)

    key_words = pd.DataFrame(key_words,columns=['score','term'])
    key_words = key_words.sort_values('score',ascending=False)
    key_words = key_words.drop_duplicates(subset=['term'])
    key_words['topic_number'] = i
    key_words['term_list'] = key_words['term'].apply(lambda x:x.split())

    for row in key_words.values.tolist():
        tmp = []

        #row[3] - termlist
        bigrams = ngrams(row[3],2)

        for bg in bigrams:
            tmp.append(' '.join(bg))

        for k in row[3]:
            tmp.append(k)
            tmp.append(lemmatizer.lemmatize(k))

        row.remove(row[3])
        row.append(list(set(tmp)))

        tmp_kwd.append(row)

    key_words = pd.DataFrame(tmp_kwd,columns=['score', 'term', 'topic_number', 'term_list'])

    topic_words = topic['topic_keywords'].values.tolist()
    topic_words = [item for sublist in topic_words for item in sublist]
    topic_words = list(set(topic_words))

    tmp = []

    for t in topic_words:
        mask = key_words['term_list'].apply(lambda x: t in x)
        key_words_processed = key_words[mask]

        if key_words_processed.empty:
            pass
        else:
            for j in key_words_processed[['score', 'term', 'topic_number']].values.tolist():
                if j not in tmp:
                    tmp.append(j)

        key_words = pd.DataFrame(tmp,columns=['score', 'term', 'topic_number'])
        top_key_words = key_words[key_words['score'] == key_words['score'].max()]

        remaining_keywords = key_words[key_words['score'] != key_words['score'].max()]
        top_key_words = top_key_words.copy()
        top_key_words = top_key_words.groupby(['score', 'topic_number']).agg({'term': lambda x: ' / '.join(map(str, x))})
        top_key_words = top_key_words.reset_index()

        remaining_keywords = remaining_keywords.copy()
        remaining_keywords['topic_number'] = remaining_keywords['topic_number'] + 0.1

        all_topics = pd.concat([top_key_words, remaining_keywords], sort=False)
        for t in all_topics.to_dict(orient='records'):
            results.append(t)

        all_topics_df = pd.DataFrame(results)
        all_topics_df = all_topics_df.sort_values('topic_number', ascending=True)


print(df.head())