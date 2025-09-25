#libraries
import time
import pandas as pd
import numpy as np
from huggingface_hub import InferenceClient
from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.preprocessing import StandardScaler

# Load DB
DB_brands = pd.read_csv("Data/Products-DataBase.csv")
DB_Embeds = pd.DataFrame()
DB_vectors = pd.DataFrame()
Scaler = StandardScaler()

client = InferenceClient(
    provider="auto",
    api_key="[YOUR HF TOKEN]",
)

# ----- Country -> Region One-hot -----
# Countries_Map = {
#     "North America": ["USA"],
#     "South America": ["Brazil"],
#     "Asia": ["Japan","India","China"],
#     "Europe": ["Italy","France","Uk","UK","Germany"],
#     "Australia": ["Australia"]
# }
# Countries_Vectors = {
#     "North America": [1,0,0,0,0],
#     "South America": [0,1,0,0,0],
#     "Asia": [0,0,1,0,0],
#     "Europe": [0,0,0,1,0],
#     "Australia": [0,0,0,0,1]
# }
# Embeds = []
# for country in DB_brands["Country"]:
#     for region, countries in Countries_Map.items():
#         if country in countries:
#             Embeds.append(Countries_Vectors[region])
# DB_Embeds["Country"] = Embeds

# ----- Sustainability Rating -----
Sus_Score_Map = {"A":0.75, "B":0.5, "C":0.25, "D":0}
DB_Embeds["Sustainability_Rating"] = [
    [round(Sus_Score_Map[r],4)] for r in DB_brands["Sustainability_Rating"]
]

# # ----- Year (normalized age) -----
# Current_Year = time.localtime().tm_year
# years = DB_brands["Year"].to_numpy()
# Embeds = Current_Year - years
# Embeds = Scaler.fit_transform(Embeds.reshape(-1,1))
# Embeds = np.round(Embeds,4).astype(float).tolist()
# DB_Embeds["Year"] = Embeds

# ----- Material Type (LLM embeddings) -----
text_inputs = [
    f"Brand: {b}, Product: {p}, Material: {m}, Colour: {c}"
    for b,p,m,c in zip(DB_brands["brand"], DB_brands["products"], DB_brands["Material_Type"], DB_brands["colour"])
]
Embeds = np.array(client.feature_extraction(
    text_inputs,
    model="Qwen/Qwen3-Embedding-8B",
))
DB_Embeds["text_embed"] = Embeds.tolist()

# ----- Numeric columns -----
def scaled_column(col):
    vals = DB_brands[col].to_numpy().reshape(-1,1)
    return np.round(Scaler.fit_transform(vals),4).astype(float).tolist()

DB_Embeds["Carbon_Footprint_MT"] = scaled_column("Carbon_Footprint_MT")
DB_Embeds["Water_Usage_Liters"]  = scaled_column("Water_Usage_Liters")
DB_Embeds["Waste_Production_KG"] = scaled_column("Waste_Production_KG")
DB_Embeds["price"] = scaled_column("price")
# DB_Embeds["avg_rating"] = scaled_column("avg_rating")

# ----- Certifications -----
Certifiations_Map = {
    "Fair Trade": [1,0,0,0],
    "B Corp": [0,1,0,0],
    "OEKO-TEX": [0,0,1,0],
    "GOTS": [0,0,0,1]
}
Embeds = []
for cert in DB_brands["Certifications"]:
    if cert in Certifiations_Map:
        Embeds.append(Certifiations_Map[cert])
    else:
        Embeds.append([0,0,0,0])  # fallback if unknown
DB_Embeds["Certifications"] = Embeds
# recycling
# recycling_Map = {
#     "Yes": 1,
#     "No" : 0
# }
# Embeds = []
# for cert in DB_brands["Recycling_Programs"]:
#     if cert in recycling_Map:
#         Embeds.append(recycling_Map[cert])
# DB_Embeds["Recycling_Programs"] = Embeds

# ----- Merge all into final vectors -----
vec = []
for i in DB_Embeds.index:
    arr = []
    for j in DB_Embeds.columns:
        val = DB_Embeds.at[i,j]
        if isinstance(val, (list, np.ndarray)):
            arr.extend(val)
        else:
            arr.append(val)
    vec.append(arr)

DB_vectors["Vectors"] = vec
DB_vectors.to_csv("Data/Embedded-Products-Databse.csv", index=False)
