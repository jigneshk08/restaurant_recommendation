import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------
# STEP 1: LOAD DATASET
# ---------------------------------------------------

# STEP 1: LOAD DATASET

df = pd.read_csv("restaurants.csv")

print("Dataset loaded successfully!")
print("Number of restaurants:", len(df))

# Show all available cities
print("\nAvailable Cities:")
print(df["City"].unique())


# ---------------------------------------------------
# STEP 2: SELECT IMPORTANT COLUMNS
# ---------------------------------------------------

columns = [
    "Restaurant Name",
    "City",
    "Cuisines",
    "Price range",
    "Aggregate rating",
    "Votes",
    "Has Online delivery"
]

df = df[columns].copy()


# ---------------------------------------------------
# STEP 3: HANDLE MISSING VALUES
# ---------------------------------------------------

df["Cuisines"] = df["Cuisines"].fillna("Unknown")
df["City"] = df["City"].fillna("Unknown")
df["Aggregate rating"] = df["Aggregate rating"].fillna(0)

print("\nMissing values handled successfully!")


# ---------------------------------------------------
# STEP 4: CLEAN TEXT DATA
# ---------------------------------------------------

df["Cuisines"] = df["Cuisines"].str.lower().str.strip()
df["City"] = df["City"].str.lower().str.strip()


# ---------------------------------------------------
# STEP 5: CREATE CONTENT FEATURE
# ---------------------------------------------------

df["content"] = (
    df["City"] + " " +
    df["Cuisines"] + " " +
    "price_" + df["Price range"].astype(str)
)


# ---------------------------------------------------
# STEP 6: CONVERT TEXT INTO NUMBERS
# ---------------------------------------------------

vectorizer = TfidfVectorizer()

feature_matrix = vectorizer.fit_transform(df["content"])


# ---------------------------------------------------
# STEP 7: CALCULATE SIMILARITY
# ---------------------------------------------------

similarity_matrix = cosine_similarity(feature_matrix)


# ---------------------------------------------------
# STEP 8: RECOMMENDATION FUNCTION
# ---------------------------------------------------

def recommend_restaurants(
    city,
    cuisine,
    price_range,
    min_rating=0,
    online_delivery="Any"
):

    city = city.lower().strip()
    cuisine = cuisine.lower().strip()

    # Create user preference text
    user_preference = (
        city + " " +
        cuisine + " " +
        "price_" + str(price_range)
    )

    # Convert user preference into vector
    user_vector = vectorizer.transform([user_preference])

    # Calculate similarity between user preference
    # and all restaurants
    similarity_scores = cosine_similarity(
        user_vector,
        feature_matrix
    ).flatten()

    # Copy dataset
    results = df.copy()

    # Add similarity score
    results["Similarity"] = similarity_scores

    # ------------------------------------------------
    # FILTER BY RATING
    # ------------------------------------------------

    results = results[
        results["Aggregate rating"] >= min_rating
    ]

    # ------------------------------------------------
    # FILTER BY ONLINE DELIVERY
    # ------------------------------------------------

    if online_delivery.lower() != "any":

        results = results[
            results["Has Online delivery"].str.lower()
            == online_delivery.lower()
        ]

    # ------------------------------------------------
    # SORT RESULTS
    # ------------------------------------------------

    results = results.sort_values(
        by=["Similarity", "Aggregate rating", "Votes"],
        ascending=False
    )

    # Return top 10 restaurants
    return results.head(10)


# ---------------------------------------------------
# STEP 9: TAKE USER INPUT
# ---------------------------------------------------

print("\n====================================")
print(" RESTAURANT RECOMMENDATION SYSTEM ")
print("====================================")

city = input(
    "\nEnter your preferred city: "
)

cuisine = input(
    "Enter your preferred cuisine: "
)

price_range = input(
    "Enter price range (1-4): "
)

min_rating = input(
    "Enter minimum rating (0-5): "
)

online_delivery = input(
    "Online delivery? (Yes/No/Any): "
)


# Convert values
price_range = int(price_range)
min_rating = float(min_rating)


# ---------------------------------------------------
# STEP 10: GET RECOMMENDATIONS
# ---------------------------------------------------

recommendations = recommend_restaurants(
    city,
    cuisine,
    price_range,
    min_rating,
    online_delivery
)


# ---------------------------------------------------
# STEP 11: DISPLAY RESULTS
# ---------------------------------------------------

print("\n====================================")
print(" RECOMMENDED RESTAURANTS ")
print("====================================")

if len(recommendations) == 0:

    print("\nNo restaurants found.")
    print("Try different preferences.")

else:

    for index, row in recommendations.iterrows():

        print("\nRestaurant:", row["Restaurant Name"])
        print("City:", row["City"])
        print("Cuisine:", row["Cuisines"])
        print("Price Range:", row["Price range"])
        print("Rating:", row["Aggregate rating"])
        print("Votes:", row["Votes"])
        print("Online Delivery:", row["Has Online delivery"])
        print("------------------------------------")
print(df["City"].unique())
