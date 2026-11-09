import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error


# =====================================================
# 1. LOAD AND PREPROCESS DATA
# =====================================================

def load_data():

    print("Loading datasets...")

    books = pd.read_csv(
        "Books.csv",
        encoding="latin-1",
        dtype={"ISBN": str},
        low_memory=False
    )

    ratings = pd.read_csv(
        "Ratings.csv",
        encoding="latin-1",
        dtype={"ISBN": str},
        low_memory=False
    )

    # 1. Clean ISBNs
    books["ISBN"] = (
        books["ISBN"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"[^0-9X]", "", regex=True)
    )
    ratings["ISBN"] = (
        ratings["ISBN"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"[^0-9X]", "", regex=True)
    )

    # Remove duplicates
    books = books.drop_duplicates("ISBN")
    ratings = ratings.drop_duplicates(["User-ID", "ISBN"])

    # Clean rating values
    ratings["Book-Rating"] = pd.to_numeric(
        ratings["Book-Rating"], errors="coerce"
    )
    ratings = ratings.dropna(subset=["User-ID", "ISBN", "Book-Rating"])

    # Keep explicit ratings (1-10)
    ratings = ratings[ratings["Book-Rating"].between(1, 10)].copy()

    # Match metadata
    ratings = ratings[ratings["ISBN"].isin(books["ISBN"])].copy()

    # Check remaining ratings count before thresholding
    print(f"Total explicit ratings matching metadata: {len(ratings)}")

    # 2. Dynamic thresholding based on dataset size
    # For small/sample CSVs, threshold is 1; for full datasets, threshold is 5
    min_user_ratings = 5 if len(ratings) > 500 else 1
    min_book_ratings = 5 if len(ratings) > 500 else 1

    if min_user_ratings > 1:
        while True:
            old_length = len(ratings)

            user_counts = ratings["User-ID"].value_counts()
            ratings = ratings[
                ratings["User-ID"].isin(
                    user_counts[user_counts >= min_user_ratings].index
                )
            ]

            book_counts = ratings["ISBN"].value_counts()
            ratings = ratings[
                ratings["ISBN"].isin(
                    book_counts[book_counts >= min_book_ratings].index
                )
            ]

            if len(ratings) == old_length:
                break

    if ratings.empty:
        raise ValueError(
            "No ratings remain. Ensure your CSV files contain sufficient non-zero rating data."
        )

    print("Preprocessing completed!")
    print("Users:", ratings["User-ID"].nunique())
    print("Books:", ratings["ISBN"].nunique())
    print("Ratings:", len(ratings))

    return books, ratings

# =====================================================
# 2. TRAIN-TEST SPLIT
# =====================================================

def split_data(data, test_size=0.2, random_state=42):

    rng = np.random.default_rng(random_state)

    train_parts = []
    test_parts = []

    # Split each user's ratings separately
    for _, group in data.groupby("User-ID"):

        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)

        n_test = max(1, int(round(len(indices) * test_size)))
        n_test = min(n_test, len(indices) - 1)

        if n_test <= 0:
            train_parts.append(group)
        else:
            test_parts.append(data.loc[indices[:n_test]])
            train_parts.append(data.loc[indices[n_test:]])

    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True) if test_parts else pd.DataFrame(columns=data.columns)

    return train, test


# =====================================================
# 3. BUILD RECOMMENDATION SYSTEM
# =====================================================

class BookRecommender:

    def __init__(self, train, books):

        self.train = train
        self.books = books

        self.global_mean = train["Book-Rating"].mean() if not train.empty else 5.0

        # Create user and book indexes
        self.users = pd.Index(train["User-ID"].unique())
        self.items = pd.Index(train["ISBN"].unique())

        self.user_index = {
            user: i for i, user in enumerate(self.users)
        }

        self.item_index = {
            isbn: i for i, isbn in enumerate(self.items)
        }

        # Create sparse user-book matrix
        rows = train["User-ID"].map(self.user_index)
        cols = train["ISBN"].map(self.item_index)

        self.user_item = csr_matrix(
            (
                train["Book-Rating"].astype(float),
                (rows, cols)
            ),
            shape=(len(self.users), len(self.items))
        )

        # Store each user's reading history
        self.history = {
            user: group.set_index("ISBN")["Book-Rating"].to_dict()
            for user, group in train.groupby("User-ID")
        }

        # Prepare popularity scores
        self.create_popularity_model()

        # Prepare content-based model
        self.create_content_model()

        # Prepare collaborative model
        self.create_collaborative_model()

    # =================================================
    # MODEL 1: POPULARITY-BASED FILTERING
    # =================================================

    def create_popularity_model(self):

        stats = self.train.groupby("ISBN")["Book-Rating"].agg(
            ["mean", "count"]
        )

        # Bayesian smoothing
        m = 10

        stats["score"] = (
            (stats["count"] / (stats["count"] + m))
            * stats["mean"]
            +
            (m / (stats["count"] + m))
            * self.global_mean
        )

        self.popularity_scores = stats["score"].to_dict()

    # =================================================
    # MODEL 2: CONTENT-BASED FILTERING
    # =================================================

    def create_content_model(self):

        metadata = (
            self.books.drop_duplicates("ISBN")
            .set_index("ISBN")
            .reindex(self.items)
        )

        title = metadata["Book-Title"].fillna("").astype(str)

        author = metadata["Book-Author"].fillna("").astype(str)

        if "Publisher" in metadata.columns:
            publisher = metadata["Publisher"].fillna("").astype(str)
        else:
            publisher = pd.Series("", index=metadata.index)

        content = title + " " + author + " " + publisher

        tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2)
        )

        tfidf_matrix = tfidf.fit_transform(content)

        self.content_similarity = cosine_similarity(
            tfidf_matrix,
            dense_output=False
        ).tocsr()

    # =================================================
    # MODEL 3: COLLABORATIVE FILTERING
    # =================================================

    def create_collaborative_model(self):

        # Transpose: rows become books, columns become users
        item_user = self.user_item.T.tocsr()

        self.collaborative_similarity = cosine_similarity(
            item_user,
            dense_output=False
        ).tocsr()

    # =================================================
    # COMMON NEIGHBOR PREDICTION FUNCTION
    # =================================================

    def neighbor_prediction(
        self, user, isbn, similarity_matrix, k=20
    ):

        if user not in self.history:
            return self.global_mean

        if isbn not in self.item_index:
            return self.global_mean

        target_idx = self.item_index[isbn]

        user_ratings = self.history[user]

        rated_indices = []
        rated_values = []

        for rated_isbn, rating in user_ratings.items():

            idx = self.item_index.get(rated_isbn)

            if idx is not None and rated_isbn != isbn:
                rated_indices.append(idx)
                rated_values.append(float(rating))

        if not rated_indices:
            return self.global_mean

        similarities = similarity_matrix[
            target_idx, rated_indices
        ].toarray().ravel()

        # Use only positive similarities
        valid = np.where(similarities > 0)[0]

        if len(valid) == 0:
            return self.global_mean

        # Select top K similar books
        valid = valid[
            np.argsort(similarities[valid])[-k:]
        ]

        scores = similarities[valid]
        ratings = np.array(rated_values)[valid]

        sum_scores = np.sum(scores)
        if sum_scores == 0:
            return self.global_mean

        prediction = np.average(
            ratings,
            weights=scores
        )

        return np.clip(prediction, 1, 10)

    # =================================================
    # GENERATE PREDICTIONS
    # =================================================

    def predict(self, user, isbn, model):

        if model == "popularity":

            return np.clip(
                self.popularity_scores.get(
                    isbn, self.global_mean
                ),
                1, 10
            )

        elif model == "content":

            return self.neighbor_prediction(
                user,
                isbn,
                self.content_similarity
            )

        elif model == "collaborative":

            return self.neighbor_prediction(
                user,
                isbn,
                self.collaborative_similarity
            )

        elif model == "hybrid":

            pop = self.predict(user, isbn, "popularity")

            content = self.predict(user, isbn, "content")

            collaborative = self.predict(
                user, isbn, "collaborative"
            )

            prediction = (
                0.20 * pop
                + 0.40 * content
                + 0.40 * collaborative
            )

            return np.clip(prediction, 1, 10)

        else:
            raise ValueError("Unknown recommendation model")

    # =================================================
    # GENERATE TOP-K RECOMMENDATIONS
    # =================================================

    def recommend(self, user, model, k=10):

        if user not in self.history:
            return []

        seen = set(self.history[user])

        candidates = [
            isbn for isbn in self.items
            if isbn not in seen
        ]

        scores = []

        for isbn in candidates:

            score = self.predict(user, isbn, model)

            scores.append((isbn, score))

        scores.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return [isbn for isbn, score in scores[:k]]


# =====================================================
# 4. RATING PREDICTION EVALUATION
# =====================================================

def evaluate_rating(model, test):

    results = []

    predictions = {}

    actual = test["Book-Rating"].to_numpy(dtype=float)

    methods = [
        "popularity",
        "content",
        "collaborative",
        "hybrid"
    ]

    for method in methods:

        print("Evaluating:", method)

        predicted = np.array([
            model.predict(user, isbn, method)
            for user, isbn in zip(
                test["User-ID"],
                test["ISBN"]
            )
        ], dtype=float)

        predictions[method] = predicted

        rmse = np.sqrt(
            mean_squared_error(actual, predicted)
        )

        mae = mean_absolute_error(actual, predicted)

        results.append({
            "Model": method.title(),
            "RMSE": rmse,
            "MAE": mae
        })

    return pd.DataFrame(results), predictions


# =====================================================
# 5. TOP-K RANKING EVALUATION
# =====================================================

def ranking_metrics(recommended, relevant, k=10):

    recommended = recommended[:k]

    relevant = set(relevant)
    if not relevant or not recommended:
        return 0.0, 0.0, 0.0

    hits = [
        1 if item in relevant else 0
        for item in recommended
    ]

    precision = sum(hits) / k

    recall = (
        sum(hits) / len(relevant)
    )

    dcg = sum(
        hit / np.log2(rank + 1)
        for rank, hit in enumerate(hits, start=1)
    )

    ideal_hits = min(len(relevant), k)

    idcg = sum(
        1 / np.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    ndcg = dcg / idcg if idcg > 0 else 0.0

    return precision, recall, ndcg


def evaluate_ranking(model, test, k=10, max_users=100):

    relevant = (
        test[test["Book-Rating"] >= 7]
        .groupby("User-ID")["ISBN"]
        .apply(set)
        .to_dict()
    )

    eligible_users = [
        user for user in relevant
        if user in model.history
    ][:max_users]

    methods = [
        "popularity",
        "content",
        "collaborative",
        "hybrid"
    ]

    results = []

    for method in methods:

        precision_values = []
        recall_values = []
        ndcg_values = []
        recommended_items = set()

        print("Ranking evaluation:", method)

        for user in eligible_users:

            recs = model.recommend(
                user, method, k
            )

            precision, recall, ndcg = ranking_metrics(
                recs,
                relevant[user],
                k
            )

            precision_values.append(precision)
            recall_values.append(recall)
            ndcg_values.append(ndcg)

            recommended_items.update(recs)

        results.append({
            "Model": method.title(),
            f"Precision@{k}": np.mean(precision_values) if precision_values else 0.0,
            f"Recall@{k}": np.mean(recall_values) if recall_values else 0.0,
            f"NDCG@{k}": np.mean(ndcg_values) if ndcg_values else 0.0,
            "Catalog Coverage": (
                len(recommended_items) / len(model.items) if len(model.items) > 0 else 0.0
            )
        })

    return pd.DataFrame(results)


# =====================================================
# 6. VISUALIZATION
# =====================================================

def create_visualizations(data, rating_results, ranking_results):

    os.makedirs("outputs", exist_ok=True)

    # Rating distribution
    plt.figure(figsize=(9, 5))

    data["Book-Rating"].value_counts().sort_index().plot(
        kind="bar"
    )

    plt.title("Distribution of Book Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Number of Ratings")

    plt.tight_layout()

    plt.savefig(
        "outputs/rating_distribution.png",
        dpi=300
    )

    plt.close()

    # RMSE and MAE comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    rating_results.plot(
        x="Model",
        y="RMSE",
        kind="bar",
        ax=axes[0],
        legend=False
    )

    axes[0].set_title("RMSE Comparison")
    axes[0].set_ylabel("Error (lower is better)")
    axes[0].tick_params(axis="x", rotation=25)

    rating_results.plot(
        x="Model",
        y="MAE",
        kind="bar",
        ax=axes[1],
        legend=False
    )

    axes[1].set_title("MAE Comparison")
    axes[1].set_ylabel("Error (lower is better)")
    axes[1].tick_params(axis="x", rotation=25)

    plt.tight_layout()

    plt.savefig(
        "outputs/model_accuracy.png",
        dpi=300
    )

    plt.close()

    # Ranking metrics
    ranking_results.set_index("Model")[
        ["Precision@10", "Recall@10", "NDCG@10"]
    ].plot(
        kind="bar",
        figsize=(10, 5)
    )

    plt.title("Top-10 Recommendation Quality")
    plt.ylabel("Score")
    plt.xticks(rotation=25)
    plt.tight_layout()

    plt.savefig(
        "outputs/ranking_comparison.png",
        dpi=300
    )

    plt.close()


# =====================================================
# 7. MAIN PROGRAM
# =====================================================

def main():

    books, data = load_data()

    train, test = split_data(data)

    print("\nTraining recommendation models...")

    recommender = BookRecommender(train, books)

    # Rating evaluation
    rating_results, predictions = evaluate_rating(
        recommender, test
    )

    # Ranking evaluation on a sample of users
    ranking_results = evaluate_ranking(
        recommender,
        test,
        k=10,
        max_users=100
    )

    # Save results
    os.makedirs("outputs", exist_ok=True)

    rating_results.to_csv(
        "outputs/rating_metrics.csv",
        index=False
    )

    ranking_results.to_csv(
        "outputs/ranking_metrics.csv",
        index=False
    )

    # Generate graphs
    create_visualizations(
        data,
        rating_results,
        ranking_results
    )

    print("\n===== RATING EVALUATION =====")

    print(rating_results.to_string(index=False))

    print("\n===== RANKING EVALUATION =====")

    print(ranking_results.to_string(index=False))

    # Example recommendations
    example_user = train["User-ID"].iloc[0]

    print("\nExample recommendations for user:", example_user)

    for method in [
        "popularity",
        "content",
        "collaborative",
        "hybrid"
    ]:

        recommendations = recommender.recommend(
            example_user,
            method,
            k=10
        )

        print("\n", method.upper())

        details = (
            books.drop_duplicates("ISBN")
            .set_index("ISBN")
            .reindex(recommendations)
        )

        print(
            details[["Book-Title", "Book-Author"]]
            .to_string()
        )

    print("\nAll outputs saved in the outputs folder!")


if __name__ == "__main__":
    main()
