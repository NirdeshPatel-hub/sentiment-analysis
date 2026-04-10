"""
train.py — Training pipeline for Sentiment Analysis model
Author: Nirdesh Patel

Dataset: Amazon Product Reviews (publicly available on Kaggle)
  https://www.kaggle.com/datasets/bittlingmayer/amazonreviews
  OR use the built-in demo dataset for quick testing.

Usage:
  python train.py                   # uses built-in demo data
  python train.py --csv reviews.csv # uses your own CSV file
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from model import SentimentModel
from utils import rating_to_label

# ──────────────────────────────────────────────
#  DEMO DATA (used when no CSV is provided)
# ──────────────────────────────────────────────
DEMO_REVIEWS = [
    ("Absolutely love this product! Works perfectly and exceeded expectations.", "Positive"),
    ("Outstanding quality, very durable and worth every penny.", "Positive"),
    ("Brilliant! Fast shipping, great packaging, highly recommend.", "Positive"),
    ("Amazing product, does exactly what it says. Five stars!", "Positive"),
    ("Best purchase I've made this year. Incredible value for money.", "Positive"),
    ("Some features are okay but nothing special. Average product.", "Neutral"),
    ("It works as described, neither great nor terrible.", "Neutral"),
    ("Decent quality for the price. Could be better, could be worse.", "Neutral"),
    ("Mixed feelings — some aspects good, some need improvement.", "Neutral"),
    ("Does the job but I expected more for this price point.", "Neutral"),
    ("Terrible quality! Broke after just two days of use.", "Negative"),
    ("Complete waste of money. Do NOT buy this product.", "Negative"),
    ("Awful experience. Product arrived defective and customer service was useless.", "Negative"),
    ("Worst purchase of my life. Returned immediately for a refund.", "Negative"),
    ("Horrible! Stopped working after one week. Avoid at all costs.", "Negative"),
] * 200   # scale up demo data to 3,000 samples


def load_data(csv_path: str = None) -> pd.DataFrame:
    """Load dataset from CSV or fall back to demo data."""
    if csv_path:
        print(f"📂 Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        # Auto-detect text & label columns
        text_col = next((c for c in df.columns if 'review' in c.lower() or 'text' in c.lower()), df.columns[0])
        label_col = next((c for c in df.columns if 'sentiment' in c.lower() or 'label' in c.lower() or 'rating' in c.lower()), df.columns[-1])
        df = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'})
        # Convert numeric ratings to labels if needed
        if df['label'].dtype in [np.float64, np.int64]:
            df['label'] = df['label'].apply(rating_to_label)
        df = df.dropna()
        print(f"✅ Loaded {len(df):,} reviews.")
    else:
        print("ℹ️  No CSV provided — using built-in demo dataset (3,000 samples).")
        df = pd.DataFrame(DEMO_REVIEWS, columns=['text', 'label'])
    return df


def plot_confusion_matrix(y_true, y_pred, classes):
    """Save confusion matrix as PNG."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Sentiment Model — Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.close()
    print("📊 Confusion matrix saved to confusion_matrix.png")


def plot_class_distribution(df):
    """Save class distribution bar chart."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df['label'].value_counts()
    colors = ['#28a745', '#ffc107', '#dc3545']
    counts.plot(kind='bar', color=colors, ax=ax, edgecolor='white', linewidth=0.5)
    ax.set_title('Class Distribution', fontsize=13, fontweight='bold')
    ax.set_xlabel('Sentiment', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_xticklabels(counts.index, rotation=0)
    for i, v in enumerate(counts):
        ax.text(i, v + 5, f'{v:,}', ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150)
    plt.close()
    print("📊 Class distribution chart saved to class_distribution.png")


def main():
    parser = argparse.ArgumentParser(description='Train Sentiment Analysis Model')
    parser.add_argument('--csv', type=str, default=None, help='Path to reviews CSV file')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split ratio (default: 0.2)')
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  💬 SENTIMENT ANALYSIS — TRAINING PIPELINE")
    print("="*55 + "\n")

    # 1. Load data
    df = load_data(args.csv)
    plot_class_distribution(df)
    print(f"\n📋 Class distribution:\n{df['label'].value_counts().to_string()}\n")

    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'],
        test_size=args.test_size,
        random_state=42,
        stratify=df['label']
    )
    print(f"📦 Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 3. Train model
    model = SentimentModel()
    model.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'='*55}")
    print(f"  🎯 TEST ACCURACY: {acc*100:.2f}%")
    print(f"{'='*55}")
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=model.classes))

    # 5. Cross-validation
    print("🔁 Running 5-fold cross-validation...")
    cv_scores = cross_val_score(model.pipeline, df['text'], df['label'], cv=5, scoring='accuracy')
    print(f"   CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # 6. Save artefacts
    plot_confusion_matrix(y_test, y_pred, model.classes)
    model.save("sentiment_model.pkl")

    print(f"\n✅ Training complete! Model saved as sentiment_model.pkl")
    print("   Run 'streamlit run app.py' to launch the web app.\n")


if __name__ == "__main__":
    main()
