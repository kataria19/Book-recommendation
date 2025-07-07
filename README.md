# Book Recommendation System

A collaborative filtering-based book recommendation system using K-Nearest Neighbors (KNN) algorithm, built with Python and machine learning libraries.

## 🛠️ Technologies Used
- **Python 3**
- Pandas (Data Manipulation)
- NumPy (Numerical Operations)
- Scikit-Learn (KNN Model)
- Matplotlib (Visualizations)
- SciPy (Sparse Matrices)

## 📂 Dataset
The Book-Crossing dataset containing:
- 278,858 users
- 271,379 books
- 1,149,780 ratings

Files used:
- `BX_Books.csv` - Book metadata
- `BX-Users.csv` - User information
- `BX-Book-Ratings.csv` - Rating data

## 🔍 Data Preprocessing
1. Filtered active users (>200 ratings) and popular ratings (>100 occurrences)
2. Merged book and rating data
3. Calculated popularity thresholds (50+ ratings)
4. Focused on USA/Canada users for relevance

## 🤖 Model Implementation
- **K-Nearest Neighbors (KNN)** with cosine similarity
- Created user-book rating matrix using sparse representation
- Recommends 5 most similar books based on collaborative filtering

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/book-recommendation-system.git
   cd book-recommendation-system
