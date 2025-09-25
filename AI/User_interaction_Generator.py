import pandas as pd
import random
brands_df = pd.read_csv("Data/Products-DataBase.csv")
# --- Define user taste profiles ---
user_profiles = {
    1: {"like": {"brand": "Roadster"}, "dislike": {"Sustainability_Rating": "D"}},
    2: {"like": {"price": ("high", 1500)}, "dislike": {"price": ("low", 1000)}},
    3: {"like": {"products": "Jeans"}, "dislike": {"colour": "Blue"}},
    4: {"like": {"Material_Type": ["Bamboo Fabric", "Organic Cotton"]}, "dislike": {"Material_Type": ["Recycled Polyester"]}},
    5: {"like": {"price": ("low", 1500)}, "dislike": {"Average_Price_USD": ("high", 1800)}},
    6: {"like": {"Sustainability_Rating": "A"}, "dislike": {"Sustainability_Rating": "D"}},
    7: {"like": {"brand": ["Roadster", "Flying Machine"]}, "dislike": {"brand": ["DOLCE CRUDOz"]}},
    8: {"like": {"Certifications": "Fair Trade"}, "dislike": {"Certifications": "None"}},
    9: {"like": {"Carbon_Footprint_MT": ("low", 200)}, "dislike": {"Carbon_Footprint_MT": ("high", 400)}},
    10: {"like": {"Water_Usage_Liters": ("low", 2000000)}, "dislike": {"Water_Usage_Liters": ("high", 4000000)}}
}

# --- Filtering function based on conditions ---
def filter_brands(condition, df):
    filtered_idx = []
    for key, value in condition.items():
        if isinstance(value, tuple):  
            # Handle numerical conditions: (mode, threshold)
            mode, threshold = value
            if key in df.columns:
                if mode == "high":
                    filtered_idx.extend(df[df[key] >= threshold].index.tolist())
                elif mode == "low":
                    filtered_idx.extend(df[df[key] <= threshold].index.tolist())
        elif isinstance(value, list):  
            filtered_idx.extend(df[df[key].isin(value)].index.tolist())
        else:  
            filtered_idx.extend(df[df[key] == value].index.tolist())
    return set(filtered_idx)

# --- Build fake interactions ---
user_data_taste = []

for user_id, profile in user_profiles.items():
    liked_idx = list(filter_brands(profile["like"], brands_df))
    disliked_idx = list(filter_brands(profile["dislike"], brands_df))

    liked_sample = random.sample(liked_idx, min(len(liked_idx), random.randint(20, 60))) if liked_idx else []
    disliked_sample = random.sample(disliked_idx, min(len(disliked_idx), random.randint(20, 60))) if disliked_idx else []

    user_data_taste.append({
        "User_id": user_id,
        "Liked_brands": liked_sample,
        "Disliked_brands": disliked_sample
    })

# Convert to DataFrame
user_df_taste = pd.DataFrame(user_data_taste)
print(user_df_taste)
user_df_taste.to_csv("Data/Users_Interactions_Database.csv")
