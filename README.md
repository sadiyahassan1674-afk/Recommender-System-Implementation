# 📚 Book Recommendation System
### Comparative Analysis and Implementation of Recommendation Algorithms

## 📌 Project Overview

This project implements and compares different recommendation system techniques using the **Book-Crossing Dataset**.

The objective is to understand how recommendation systems analyze user preferences, identify similarities between books, and generate personalized recommendations.

Four recommendation approaches are implemented and evaluated:

1. Popularity-Based Filtering
2. Content-Based Filtering
3. Collaborative Filtering (Item-Item)
4. Hybrid Recommendation System

The models are evaluated using rating prediction metrics and Top-K recommendation metrics to compare their effectiveness.

---

## 🎯 Project Objectives

- Understand the working principles of recommendation systems.
- Explore different recommendation system paradigms.
- Clean and preprocess book-rating data.
- Implement four recommendation algorithms using Python.
- Generate personalized book recommendations.
- Evaluate model performance using standard evaluation metrics.
- Compare the strengths and weaknesses of different recommendation approaches.
- Visualize the results for better interpretation.

---

## 📂 Dataset Description

This project uses the **Book-Crossing Dataset**, which contains information about books and user ratings.

Two CSV files are required:

| Dataset | Description |
|---|---|
| `Books.csv` | Contains book information such as ISBN, title, author, and publisher. |
| `Ratings.csv` | Contains user IDs, ISBNs, and book ratings. |

### Important Dataset Features

**Books.csv**
- ISBN
- Book-Title
- Book-Author
- Year-Of-Publication
- Publisher

**Ratings.csv**
- User-ID
- ISBN
- Book-Rating

Book ratings range from 1 to 10.

A rating of 0 represents implicit or unspecified feedback rather than an explicit rating. Therefore, zero ratings are excluded during preprocessing.

---

## ⚙️ Technologies and Libraries Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data loading and preprocessing |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| SciPy | Sparse matrix operations |
| Scikit-learn | TF-IDF, cosine similarity, and evaluation metrics |

### Installation

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib scipy scikit-learn
```

---

## 🔄 System Workflow

The recommendation system follows this workflow:

```text
          START
            |
            v
     Load CSV Datasets
            |
            v
      Data Preprocessing
            |
            v
       Train-Test Split
            |
            v
    Build Recommendation Models
            |
     +------+------+------+
     |      |      |      |
     v      v      v      v
   Popularity Content Collaborative
                    |
               Hybrid Model
                    |
                    v
           Generate Predictions
                    |
                    v
            Model Evaluation
                    |
          +---------+---------+
          |                   |
          v                   v
    Rating Metrics       Ranking Metrics
     RMSE, MAE        Precision, Recall,
                         NDCG, Coverage
          |                   |
          +---------+---------+
                    |
                    v
             Visualizations
                    |
                    v
               END
```

---

## 🧹 1. Data Preprocessing

Before building recommendation models, the dataset undergoes several preprocessing steps.

### ISBN Cleaning

ISBN values are converted into strings, stripped of unnecessary spaces, converted to uppercase, and cleaned of unwanted characters.

This ensures consistent matching between the book and ratings datasets.

### Duplicate Removal

Duplicate book records and duplicate user-book rating combinations are removed.

### Rating Filtering

Only explicit ratings between 1 and 10 are retained.

### Metadata Matching

Ratings are matched with the available book information using ISBN.

### Dynamic Filtering

To improve efficiency:

- For datasets containing more than 500 matching ratings, users and books with fewer than 5 ratings are filtered out.
- For smaller datasets, the minimum threshold is reduced to 1.

Filtering is repeated until the dataset becomes stable.

---

## 📊 2. Train-Test Split

The dataset is divided into:

- **80% Training Data**
- **20% Testing Data**

A user-wise splitting approach is used so that users with multiple ratings retain training history while some ratings are reserved for testing.

A random seed of 42 is used to make the experiment reproducible.

The recommendation models are trained using the training dataset and evaluated against the test dataset.

---

# 🤖 Recommendation Algorithms

## 3. Model 1: Popularity-Based Filtering

### Concept

Popularity-based recommendation suggests books based on their average ratings.

Instead of considering individual user preferences, it identifies books that are generally well-rated.

### Implementation

The model calculates the mean rating and number of ratings for each book.

It uses **Bayesian Smoothing** to prevent books with very few ratings from receiving unrealistically high scores.

The formula is:

\[
Score(b)=
\frac{n_b}{n_b+m}R_b+
\frac{m}{n_b+m}C
\]

Where:

- \(R_b\) = Average rating of a book.
- \(n_b\) = Number of ratings received by the book.
- \(C\) = Global average rating.
- \(m\) = Smoothing parameter (10).

### Advantages
- Simple to implement.
- Computationally efficient.
- Useful for identifying generally popular books.

### Disadvantages
- Does not consider individual user interests.
- Recommendations may be similar for every user.
- Less effective for users with specific preferences.

---

## 4. Model 2: Content-Based Filtering

### Concept

Content-based filtering recommends books based on their characteristics.

The system analyzes:

- Book titles
- Authors
- Publishers

It uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert textual information into numerical feature vectors.

### TF-IDF

TF-IDF assigns importance to words based on how frequently they occur in a document and how uncommon they are across documents.

### Cosine Similarity

Cosine similarity measures the similarity between two book feature vectors.

\[
Similarity(A,B)=
\frac{A\cdot B}{\|A\|\|B\|}
\]

A higher similarity score indicates that two books have more similar textual characteristics.

The system identifies similar books and uses the user's previous ratings to predict ratings for other books.

### Advantages
- Provides personalized recommendations.
- Does not require finding similar users.
- Can recommend books based on their metadata.

### Disadvantages
- Depends on the availability and quality of book metadata.
- May recommend books that are too similar to previously read books.
- Cannot easily discover completely different interests.

---

## 5. Model 3: Collaborative Filtering

### Concept

Collaborative filtering recommends books based on user-rating patterns.

The assumption is that books receiving similar ratings from users may have similar appeal.

This project implements **Item-Item Collaborative Filtering**.

### Implementation

First, a User-Item Rating Matrix is constructed.

Rows represent users, columns represent books, and values represent ratings.

The matrix is transposed to create an Item-User Matrix.

Cosine similarity is then calculated between books based on their rating patterns across users.

When predicting a user's rating for a book, the system:

1. Identifies books the user has previously rated.
2. Finds books similar to the target book.
3. Selects the top 20 similar books.
4. Calculates a similarity-weighted average of the user's previous ratings.

### Prediction Formula

\[
\hat r_{u,i}=
\frac{\sum_{j\in N(i,u)}s(i,j)r_{u,j}}
{\sum_{j\in N(i,u)}s(i,j)}
\]

Where:

- \(r_{u,j}\) = User's rating for book j.
- \(s(i,j)\) = Similarity between books i and j.
- \(N(i,u)\) = Similar books rated by user u.

### Advantages
- Learns from collective user behaviour.
- Does not depend on book descriptions.
- Can discover relationships between different books.

### Disadvantages
- Suffers from sparse rating data.
- New books may have insufficient rating history.
- Requires more computational resources than popularity-based filtering.

---

## 6. Model 4: Hybrid Recommendation System

### Concept

The hybrid recommendation system combines the predictions of all three previous models.

The objective is to balance popularity, book characteristics, and collaborative user behaviour.

### Implementation

The model uses a weighted combination:

\[
Hybrid =
0.20(P)+0.40(CB)+0.40(CF)
\]

Where:

- P = Popularity prediction.
- CB = Content-based prediction.
- CF = Collaborative filtering prediction.

### Advantages
- Combines multiple recommendation strategies.
- Balances general popularity and individual preferences.
- Can reduce the limitations of relying on a single approach.

### Disadvantages
- More complex to implement.
- Requires weight selection.
- Performance depends on the quality of the individual models.

**Note:** The weights are fixed experimental values in this implementation. They are not claimed to be mathematically optimal.

---

# 📈 7. Model Evaluation

Two categories of evaluation metrics are used to measure recommendation performance.

## A. Rating Prediction Metrics

### RMSE (Root Mean Squared Error)

RMSE measures the square root of the average squared difference between actual and predicted ratings.

\[
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}
\]

A lower RMSE indicates better prediction accuracy.

### MAE (Mean Absolute Error)

MAE measures the average absolute difference between actual and predicted ratings.

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|
\]

A lower MAE indicates that predictions are closer to actual ratings on average.

## B. Ranking Evaluation Metrics

### Precision@10

Measures how many of the top 10 recommended books are relevant.

\[
Precision@10=\frac{\text{Relevant Recommended Books}}{10}
\]

### Recall@10

Measures how many of the user's relevant books were successfully recommended in the top 10.

\[
Recall@10=
\frac{\text{Relevant Recommended Books}}
{\text{Total Relevant Books}}
\]

### NDCG@10

Normalized Discounted Cumulative Gain evaluates both the relevance and ranking position of recommended books.

Higher NDCG indicates that relevant books appear closer to the top of the recommendation list.

### Catalog Coverage

Measures the proportion of available books that appear in the recommendations generated for the evaluated users.

\[
Coverage=
\frac{\text{Unique Recommended Books}}
{\text{Books Available in Model Catalog}}
\]

For ranking evaluation, ratings of 7 or higher are considered relevant.

The current implementation evaluates up to 100 eligible users to limit computation time.

---

## 📉 8. Data Visualization

The program automatically generates three visualizations.

| Visualization | Purpose |
|---|---|
| Rating Distribution | Displays the distribution of explicit book ratings. |
| Model Accuracy | Compares RMSE and MAE across all four models. |
| Ranking Comparison | Compares Precision@10, Recall@10, and NDCG@10. |

All graphs are saved in the `outputs` directory in PNG format with 300 DPI resolution.

---

## 📁 Project Structure

After running the program, the project directory will look like this:

```text
Book-Recommendation-System/
│
├── Books.csv
├── Ratings.csv
│
├── recommender.py
│
├── outputs/
│   ├── rating_metrics.csv
│   ├── ranking_metrics.csv
│   ├── rating_distribution.png
│   ├── model_accuracy.png
│   └── ranking_comparison.png
│
└── README.md
```

---

## ▶️ 9. How to Run the Project

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the Project Directory

```bash
cd Book-Recommendation-System
```

### Step 3: Install Dependencies

```bash
pip install pandas numpy matplotlib scipy scikit-learn
```

### Step 4: Add the Dataset

Place `Books.csv` and `Ratings.csv` in the project root directory.

### Step 5: Execute the Program

```bash
python recommender.py
```

### Step 6: View Results

After execution, check the `outputs` folder for evaluation metrics and generated graphs.

---

## 📋 10. Expected Program Outputs

The program prints:

- Number of users.
- Number of books.
- Number of ratings after preprocessing.
- Rating prediction results for all four models.
- Ranking evaluation results.
- Example recommendations for a selected user.

It also saves evaluation tables and visualizations for further analysis.

**Important:** Actual RMSE, MAE, Precision, Recall, NDCG, and Coverage values depend on the dataset and must be obtained by running the program. No performance results are assumed or fabricated in this README.

---

## ⚖️ 11. Comparative Analysis

| Model | Personalization | Main Strength | Main Limitation |
|---|---|---|---|
| Popularity-Based | Low | Simplicity and speed | No individual preferences |
| Content-Based | High | Uses book characteristics | Limited by metadata |
| Collaborative | High | Learns user behaviour | Sparse data |
| Hybrid | High | Combines multiple signals | More complex |

The evaluation metrics can be used to identify which model performs best in terms of rating prediction and recommendation ranking.

A model with lower RMSE does not necessarily have better Top-K recommendation quality. Therefore, both types of evaluation are important.

---

## ⚠️ 12. Limitations and Future Improvements

### Current Limitations

- The system relies on explicit ratings.
- Content-based filtering uses limited book metadata.
- Collaborative filtering may struggle with new users and books.
- The hybrid model uses fixed weights.
- Ranking evaluation is limited to a sample of 100 users by default.
- The system does not currently use advanced deep-learning recommendation techniques.

### Future Improvements

The project can be extended by:

1. Implementing Matrix Factorization using SVD.
2. Adding user-user collaborative filtering.
3. Incorporating book descriptions and genres into content-based filtering.
4. Optimizing hybrid model weights using validation data.
5. Adding Precision@K and Recall@K evaluations for larger user samples.
6. Developing a web interface using Streamlit or Flask.
7. Integrating a live book recommendation feature.

---

## 🎓 Conclusion

This project demonstrates the implementation and comparison of four major recommendation system approaches using a real-world book-rating dataset.

By combining popularity-based, content-based, collaborative, and hybrid recommendation techniques, the project explores how different algorithms identify user interests and generate personalized suggestions.

The use of RMSE, MAE, Precision@10, Recall@10, NDCG@10, and Catalog Coverage provides a broader understanding of recommendation system performance.

The project highlights the importance of selecting appropriate recommendation strategies based on data availability, user behaviour, and evaluation objectives.

---

## 👩‍💻 Author

**Book Recommendation System – Recommendation Algorithm Analysis and Implementation**

Developed as part of a project on recommender systems, algorithm implementation, and performance evaluation.

---

⭐ If you find this project useful, consider giving the repository a star!
