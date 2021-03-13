import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

beatly_data = pd.read_csv('model/final_merged_data_2.csv', low_memory=False)
beatly_data['overview'] = (beatly_data['date'] + ' ' + beatly_data['genres'] + ' ' + beatly_data['artist'] + ' ' + beatly_data['album_name'])

tfidf_vector = TfidfVectorizer(stop_words='english')
beatly_data['overview'] = beatly_data['overview'].fillna('')
tfidf_matrix = tfidf_vector.fit_transform(beatly_data['overview'])
sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(beatly_data.index, index=beatly_data['album_name']).drop_duplicates()

def content_based_recommender(album_name, sim_scores=sim_matrix):
    idx = indices[album_name]
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:9]
    album_indices = [i[0] for i in sim_scores]
    return beatly_data['album_name'].iloc[album_indices]
