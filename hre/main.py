from flask import Flask, request, jsonify
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise.reader import Reader

app = Flask(__name__)

# Load data
data = pd.read_csv('Merged file.csv')
reader = Reader(rating_scale=(1, 5))

# Create Dataset object from pandas dataframe
music_data = Dataset.load_from_df(data[['user_id', 'track_id']], reader)
trainset, testset = train_test_split(music_data, test_size=0.2)

# Train SVD algorithm for collaborative filtering
algo = SVD()
algo.fit(trainset)

# Use TF-IDF vectorizer for content-based filtering
arabic_stop_words = ['في', 'من', 'على', 'إلى', 'عن', 'أن', 'هذا', 'هذه', 'هذان', 'هؤلاء', 'ذلك', 'ذلكم', 'ذلكما', 'ذلكن', 'هناك', 'وهو', 'وهي', 'كما', 'لكن', 'وا', 'نحن', 'أنا', 'أنت', 'أنتما', 'أنتم', 'أنتن', 'إياك', 'إياكم', 'إياكما', 'إياكن', 'ما', 'منها', 'منه', 'ذا', 'ذي', 'أولئك']
tfidf = TfidfVectorizer(stop_words=arabic_stop_words)
tfidf_matrix = tfidf.fit_transform(data['name'])

# Calculate cosine similarity matrix for content-based filtering
cosine_sim = cosine_similarity(tfidf_matrix)

# Get song indices for content-based filtering
indices = pd.Series(data.index, index=data['name'])

# Define function to get top n recommendations based on hybrid filtering
def hybrid_recommendations(user_id, name, n):
    # Get song indices for content-based filtering
    indices = pd.Series(data.index, index=data['name'])
    idx = indices[name]

    # Get cosine similarity scores for content-based filtering
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort songs by cosine similarity scores for content-based filtering
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]

    # Get song indices and play counts for collaborative filtering
    collab_indices = []
    for track_id in data['track_id'].unique():
        collab_indices.append((data[data['track_id'] == track_id].index[0], algo.predict(user_id, track_id).est))

    # Sort songs by predicted play counts for collaborative filtering
    collab_indices = sorted(collab_indices, key=lambda x: x[1], reverse=True)
    collab_indices = collab_indices[:n]

    # Combine indices and scores from both filters
    indices = [i[0] for i in sim_scores]
    indices += [i[0] for i in collab_indices]
    scores = [i[1] for i in sim_scores]
    scores += [i[1] for i in collab_indices]

    hybrid_scores = pd.DataFrame({'index': indices, 'score': scores})
    hybrid_scores = hybrid_scores.groupby('index').sum().reset_index()
    hybrid_scores = hybrid_scores.sort_values('score', ascending=False).head(n)

    # Get top n recommended songs based on hybrid filtering
    recommended_songs = data.iloc[hybrid_scores['index']][['name', 'artist','preview_url',]].drop_duplicates()
    recommended_songs['hybrid_score'] = hybrid_scores['score'].values
    recommended_songs = recommended_songs.sort_values('hybrid_score', ascending=False)

    return recommended_songs.head(n)

# Define Flask API endpoints
@app.route('/')
def home():
    return 'Welcome to the music recommendations API!'

@app.route('/recommend/<user_id>/<name>')
def recommend(user_id, name):
    n = request.args.get('n', default=10, type=int)
    recommended_songs = hybrid_recommendations(user_id, name, n)
    result = recommended_songs.to_dict('records')
    return jsonify(result)

# Run the Flask app
if __name__ == '__main__': 
    app.run(debug=True)