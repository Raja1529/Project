import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import os

nltk.download('stopwords')
from nltk.corpus import stopwords

file_path = 'famous_bollywood_movies_2000_2025.csv'

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"❌ File not found at: {file_path}")
    exit()

def combine_features(row):
    return f"{row['genre']} {row['cast']} {row['description']}"

df['combined_features'] = df.apply(combine_features, axis=1)

vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
count_matrix = vectorizer.fit_transform(df['combined_features'])

cosine_sim = cosine_similarity(count_matrix)

def recommend(movie_title, num_recommendations=5):
    movie_title_lower = movie_title.lower()
    title_mapping = {title.lower(): title for title in df['title'].values}

    if movie_title_lower not in title_mapping:
        print(f"\n❌ Movie '{movie_title}' not found.")
        print("✅ Available sample movie titles:")
        for title in df['title'].sample(10):  # print a few titles for reference
            print(f" - {title}")
        return

    actual_title = title_mapping[movie_title_lower]
    movie_index = df[df.title == actual_title].index[0]
    similarity_scores = list(enumerate(cosine_sim[movie_index]))
    sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]

    print(f"\n🎬 Because you liked '{actual_title}', you may also like:")
    for i, (index, score) in enumerate(sorted_similar_movies):
        print(f"{i+1}. {df.iloc[index]['title']} (Similarity Score: {score:.2f})")

if __name__ == "__main__":
    print("🎞️ Welcome to the Bollywood Movie Recommender!")
    print("Type the name of a movie to get recommendations (or type 'exit' to quit).\n")

    while True:
        user_input = input("🔍 Enter a movie title: ").strip()
        if user_input.lower() == 'exit':
            print("👋 Goodbye!")
            break
        recommend(user_input)
        print()
