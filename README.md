# Cognifyz Technologies - Machine Learning Internship

![Cognifyz](https://img.shields.io/badge/Cognifyz-ML%20Internship-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)

## 📌 About
This repository contains all the completed tasks for the **Cognifyz Technologies Machine Learning Internship Program**. The internship involves working with a real-world restaurant dataset to build ML models and derive insights.

```

---

## 📊 Dataset Overview
| Column | Description |
|--------|-------------|
| Restaurant ID/Name | Unique identifier and name |
| City, Locality | Location info |
| Longitude, Latitude | Geo-coordinates |
| Cuisines | Type of cuisines served |
| Average Cost for two | Dining cost |
| Price range | 1 (cheap) to 4 (expensive) |
| Aggregate rating | Target variable (0–5) |
| Votes | Number of customer votes |
| Has Table booking / Online delivery | Service features |

---

## ✅ Tasks Completed

### Task 1 – Restaurant Rating Prediction
- **Objective:** Predict aggregate restaurant ratings using regression models.
- **Algorithms:** Linear Regression, Decision Tree, **Random Forest** ✅ (Best), Gradient Boosting
- **Metrics:** MSE, RMSE, MAE, R²
- **Key Finding:** Votes and Price Range are the most influential features.

### Task 2 – Restaurant Recommendation System
- **Objective:** Content-based filtering recommendation system.
- **Approach:** Cosine similarity on cuisine + numerical feature vectors
- **Input:** Cuisine preference, city, budget, delivery preference
- **Output:** Top-N personalized restaurant recommendations

### Task 3 – Cuisine Classification
- **Objective:** Classify restaurants by their primary cuisine type.
- **Algorithms:** Logistic Regression, Decision Tree, **Random Forest** ✅ (Best), Gradient Boosting
- **Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Key Finding:** Rating and Price Range are top discriminating features.

### Task 4 – Location-Based Analysis
- **Objective:** Geographical analysis of restaurant distribution.
- **Analysis:** Global map, India regional map, city-level stats, price range heatmaps
- **Key Finding:** New Delhi dominates; higher price range correlates with better ratings.

---

## 🛠️ Libraries Used
```
pandas, numpy, matplotlib, seaborn, scikit-learn
```

## 🚀 How to Run
1. Clone the repository
2. Place `Dataset.csv` in the `dataset/` folder
3. Open any `.ipynb` file in Jupyter Notebook or VS Code
4. Run all cells

---

## 📬 Connect
**Cognifyz Technologies**  
🌐 [www.cognifyz.com](https://www.cognifyz.com)  
📧 contact@cognifyz.com  
🏷️ #cognifyz #cognifyzTech #cognifyzTechnologies
