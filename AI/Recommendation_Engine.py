import numpy as np
import ast
from sklearn.preprocessing import normalize
from keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity as cs
import pandas as pd
embedding_dim = 4105
top_k = 10  # how many recommendations to return
model = load_model("AI/Model/Recommendation_Model.h5",compile=False)

# ---- Load data ----
DB_Embeds = pd.read_csv("Data/Embedded-Products-Databse.csv")
DB_User = pd.read_csv("Data/Users_Interactions_Database.csv")

DB_Embeds["Embedding"] = DB_Embeds["Vectors"].apply(lambda x: np.array(ast.literal_eval(x)))

# Normalize embeddings for stability
DB_Embeds["Embedding"] = DB_Embeds["Embedding"].apply(lambda x: normalize(x.reshape(1, -1))[0])

# Create brand lookup
brand2vec = dict(zip(DB_Embeds.index, DB_Embeds["Embedding"]))

# ---- Helper functions ----
def get_user_profile(row):
    liked_vecs = [brand2vec[b] for b in row["Liked_brands"] if b in brand2vec]
    disliked_vecs = [brand2vec[b] for b in row["Disliked_brands"] if b in brand2vec]

    if len(liked_vecs) == 0:
        return np.zeros(embedding_dim)

    user_like_avg = np.mean(liked_vecs, axis=0)
    user_dislike_avg = np.mean(disliked_vecs, axis=0) if disliked_vecs else np.zeros(embedding_dim)
    user_vec = user_like_avg - user_dislike_avg
    user_vec = user_vec / np.linalg.norm(user_vec)
    return user_vec

def recommend(user_vec, top_k=10):
    pred_vec = model.predict(user_vec.reshape(1, embedding_dim))[0]
    pred_vec = pred_vec / np.linalg.norm(pred_vec)  # normalize

    all_embeddings = np.vstack(DB_Embeds["Embedding"].values)
    sims = np.dot(all_embeddings, pred_vec)
    top_indices = sims.argsort()[::-1][:top_k]
    top_brands = DB_Embeds.iloc[top_indices].index.tolist()
    return top_brands

# ---- Example ----
user_vec = get_user_profile(DB_User.iloc[0])
top_brands = recommend(user_vec, top_k=10)
print("Top recommended brand IDs:", top_brands)


  
