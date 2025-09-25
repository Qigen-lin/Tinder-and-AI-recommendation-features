import pandas as pd
import ast
import numpy as np
from sklearn.preprocessing import normalize
from keras.models import Sequential, load_model
from keras.layers import Dense, Input

embedding_dim = 4105
top_k = 10  # how many recommendations to return

# ---- Load data ----
DB_Embeds = pd.read_csv("Data/Embedded-Products-Databse.csv")
DB_User = pd.read_csv("Data/Users_Interactions_Database.csv")
print(DB_Embeds["Vectors"].head())
# Convert stringified lists to real lists
DB_User["Liked_brands"] = DB_User["Liked_brands"].apply(lambda x: ast.literal_eval(x))
DB_User["Disliked_brands"] = DB_User["Disliked_brands"].apply(lambda x: ast.literal_eval(x))
def safe_parse(x):
    try:
        return np.array(ast.literal_eval(x))
    except Exception:
        print("Bad row:", x)
        return np.zeros(embedding_dim)  # fallback so your code keeps working

DB_Embeds["Embedding"] = DB_Embeds["Vectors"].apply(safe_parse)

# Normalize embeddings for stability
DB_Embeds["Embedding"] = DB_Embeds["Embedding"].apply(lambda x: normalize(x.reshape(1, -1))[0])

# Create brand lookup
brand2vec = dict(zip(DB_Embeds.index, DB_Embeds["Embedding"]))

# ---- Prepare training data ----
X_train = []
y_train = []

for idx, row in DB_User.iterrows():
    liked_vecs = [brand2vec[b] for b in row["Liked_brands"] if b in brand2vec]
    disliked_vecs = [brand2vec[b] for b in row["Disliked_brands"] if b in brand2vec]
    
    if len(liked_vecs) == 0:
        continue

    # user profile: avg(liked) - avg(disliked)
    user_like_avg = np.mean(liked_vecs, axis=0)
    user_dislike_avg = np.mean(disliked_vecs, axis=0) if disliked_vecs else np.zeros(embedding_dim)
    user_vec = user_like_avg - user_dislike_avg
    user_vec = user_vec / np.linalg.norm(user_vec)  # normalize

    target_vec = np.mean(liked_vecs, axis=0)  # average liked brand embedding
    target_vec = target_vec / np.linalg.norm(target_vec)  # normalize

    X_train.append(user_vec)
    y_train.append(target_vec)

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Training samples: {X_train.shape[0]}")

# ---- Build dense model ----
model = Sequential([
    Input(shape=(embedding_dim,)),
    Dense(1024, activation="relu"),
    Dense(embedding_dim, activation="linear")  # outputs embedding
])
model.compile(optimizer="adam", loss="mse")

# ---- Train ----
model.fit(X_train, y_train, epochs=30, batch_size=8)

# ---- Save model ----
model.save("AI/Model/Recommendation_Model.h5")

