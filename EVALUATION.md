Model Evaluation & Performance Report
This report documents the quantitative performance metrics, evaluation methodology, and comparative analysis of the recommendation system models implemented for Task 2.

1. Evaluation Methodology
To rigorously benchmark the recommendation system paradigms, we utilized an 80-20 Train-Test Split on the preprocessed Book-Crossing dataset.

Rating Evaluation: Measures how accurately each model predicts the exact rating a user would give to a book.

Ranking Evaluation: Measures the top-
K recommendation quality (
K=10) to assess how relevant and diverse the recommended books are for users.

2. Quantitative Performance Summary
A. Rating Evaluation (RMSE & MAE)
Lower values indicate better rating prediction accuracy.

Model	RMSE (↓)	MAE (↓)
Popularity	1.7188	1.3584
Content	1.6945	1.2127
Collaborative	1.6985	1.2374
Hybrid	1.5580	1.1746
B. Ranking Evaluation (Precision@10, Recall@10, NDCG@10, Catalog Coverage)
Higher values indicate superior recommendation ranking quality and item diversity.

Model	Precision@10 (↑)	Recall@10 (↑)	NDCG@10 (↑)	Catalog Coverage (↑)
Popularity	0.001	0.0009	0.0007	0.0013
Content	0.000	0.0000	0.0000	0.0510
Collaborative	0.000	0.0000	0.0000	0.0495
Hybrid	0.002	0.0114	0.0051	0.0795
3. Visual Performance Comparison
The bar chart below compares the rating prediction errors (RMSE and MAE) across all four implemented recommendation paradigms:

4. Key Takeaways & Insights
Hybrid Superiority: The Hybrid Model consistently outperforms all individual models across both rating accuracy (lowest RMSE of 1.5580 and MAE of 1.1746) and top-10 ranking metrics.

Complementary Strengths: While Content-Based and Collaborative Filtering capture individual user preferences and item features, combining them in a Hybrid architecture minimizes prediction error while maximizing catalog diversity (0.0795)
