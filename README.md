# User-Segmentation
Customer Segmentation Based on Delivery Mode Usage

## Context & Background

This project analyzes user behavior in a **multi-modal delivery platform**.

Think of an app where users can request different types of delivery services depending on their needs — for example:
- Bike or motorbike for small and fast deliveries
- Standard vans for medium-sized packages
- Heavy vans or trucks for large or bulky items
- Special vehicle types with specific configurations (e.g. floor, roof, box)

Users are free to choose different delivery modes over time, and their choices reflect **their real-world needs, habits, and constraints**.

The goal of this project is to understand those patterns and group users into meaningful behavioral segments.

>  Note:  
> This analysis is based on **anonymized and generalized data** from a real-world logistics platform.  
> All company-specific names, identifiers, and sensitive operational details have been removed or abstracted for public sharing.

---

## Objective

The main objective is to **segment non-business customers** based on how they distribute their orders across different delivery categories.

Instead of focusing on how *much* users order, we focus on **how they choose delivery modes**:
- Are they bike-only users?
- Do they mostly rely on vans?
- Do they switch between multiple categories?
- Are there niche users with very specific vehicle needs?

This type of segmentation is useful for:
- Product personalization
- Pricing and promotion strategies
- Operational planning
- Understanding customer intent beyond raw volume

---

##  Data Scope & Filtering

- **Time period:**  
  1 year.

- **Included:**  
  All completed delivery orders placed via the app during the analysis period.

- **Excluded:**
  - Business customers.

- **Focus of this report:**  
  **Non-business (individual) customers**, who represent the majority of the user base and exhibit more diverse behavioral patterns.

---

##  Analytical Approach

### 1. Feature Engineering (Behavior over Volume)

To capture *behavioral preference* rather than order intensity:

- Orders were aggregated **per customer × delivery category**
- For each customer, we computed: category_share = orders_in_category / total_orders


This means every customer is represented by a **distribution vector** whose values sum to 1.

---

### 2. Normalization

- Category share features were standardized using **StandardScaler**
- This prevents categories with higher natural variance from dominating the clustering

---

### 3. Clustering

- **K-Means clustering** was applied to group users with similar usage distributions
- The optimal number of clusters was selected based on:
- Inertia
- Silhouette score
- Cluster centers were then interpreted to define **clear behavioral identities**

---

## 📊 Cluster Summary

| Cluster | Dominant Usage Pattern | % of Users | % of Orders | Interpretation |
|-------|------------------------|-----------|-------------|----------------|
| 1 | BWB-dominant (~93%) | ~36% | ~52% | High-impact users with strong preference for one delivery mode |
| 2 | Van-dominant (~89%) | ~44% | ~30% | Largest user group, consistent medium-sized delivery needs |
| 3 | Bike + BWB mix | ~5% | ~7% | Hybrid users switching between lightweight and standard deliveries |
| 4 | Heavy Van-dominant | ~5% | ~3% | Users with consistently large or heavy items |
| 5 | PP-dominant | ~3.5% | ~5% | Distinct niche with specific delivery requirements |
| 6 | Van with floor | ~1.4% | <1% | Very specialized vehicle configuration users |
| 7 | VHM-dominant | <1% | <0.5% | Rare edge-case behavior |
| 8 | Truck-dominant | <1% | <0.5% | Heavy logistics, low-frequency |
| 9 | Carbox-dominant | ~1% | ~1% | Small, car-based delivery group |
| 10 | Bike-only (~97%) | ~2.5% | ~1% | Lightweight, fast-delivery users |

---

## 🔍 Key Insights

- **Behavioral dominance matters more than headcount**  
Some clusters generate a disproportionately large share of orders relative to their size.

- **Most users are specialists, not generalists**  
The majority strongly prefer a single delivery category rather than mixing many.

- **Long tail of niche behaviors exists**  
Small clusters represent operationally important but rare delivery needs.

---

## 🛠 Tech Stack & Methods

- Python (Pandas, NumPy)
- Scikit-learn (StandardScaler, KMeans)
- Exploratory data analysis
- Behavioral feature engineering

---

## 🚀 Why This Project Matters

This project demonstrates how:
- Raw transactional data can be transformed into **behavioral signals**
- Normalization enables fair clustering
- Clustering results become actionable when interpreted correctly

It is designed as a **portfolio-grade example** of customer segmentation in a real-world product context.

---

## 📌 Disclaimer

This repository is for **educational and demonstrational purposes only**.  
All data structures, metrics, and results have been anonymized and abstracted to avoid disclosure of proprietary or sensitive information.

  

