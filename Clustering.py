import pandas as pd
import json
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
try:
    with open(r"C:\Users\ayman\Desktop\4IR4\Machine learning\News_Category_Dataset_v3.json", 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    df_clean = df.drop_duplicates(subset=['headline', 'short_description'])
    df_clean = df_clean.dropna(subset=['headline', 'short_description'])
    df_clean = df_clean[df_clean['headline'].str.len() > 10]
    df_clean = df_clean[df_clean['short_description'].str.len() > 20]
    df_clean["text"] = df_clean["headline"] + " " + df_clean["short_description"]
    X = df_clean["text"]
    y = df_clean["category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=50000)),
        ('clf', LogisticRegression(max_iter=2000, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy :", accuracy)
    print("\n=== EXEMPLES DE PRÉDICTIONS ===")
    for i in range(10):
        print("\nTexte :", X_test.iloc[i][:150], "...")
        print("Vraie catégorie :", y_test.iloc[i])
        print("Prédiction :", y_pred[i])
        print("-" * 50)

except Exception as e:
    print(f"Erreur : {e}")
