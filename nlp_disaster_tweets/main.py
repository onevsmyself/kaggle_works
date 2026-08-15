import re
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import recall_score, precision_score


def cnt_best_threshold(model: LogisticRegression, x_val_tfidf: np.ndarray,
y_val: np.ndarray) -> tuple[float, float]:
    # выбор метрики для выбора оптимального порога  
    SCORERS = {
        "f1": f1_score,
        "recall": recall_score,
        "precision": precision_score,
    }
    metric = SCORERS["f1"]

    probas_val = model.predict_proba(x_val_tfidf)[:, 1]
    best_threshold, best_score = 0, 0

    for t in np.linspace(0.1, 0.99, 90):
        pred = (probas_val >= t).astype(int)
        score = metric(y_val, pred)

        if score > best_score:
            best_score = score
            best_threshold = t
    return best_threshold, best_score

def prerocess_df(df: pd.DataFrame) -> tuple[pd.Series, pd.Series | None]:
    if 'target' in df.columns:
        y = df["target"]
        x = df.drop(["target", "location"], axis=1)
    else:
        y = None
        x = df.drop(["location"], axis=1)

    x['keyword'] = x['keyword'].str.replace('%20', ' ', regex=False)
    # простая отчистка от всего лишнего, кроме букв и цифр
    x['text'] = x['text'].str.lower()
    x['text'] = x['text'].str.replace(r'https?://t.co/\S+', '', regex=True)
    x['text'] = x['text'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
    x['text'] = x['text'].str.replace(r'\s+', ' ', regex=True)
    x['text'] = x['text'].str.strip()

    x['combined'] = x['keyword'].fillna('') + ' ' + x['text']
    return x['combined'], y


def clean_text(text: str) -> str:
    text = text.lower()
    # заменяем спец. символы на пробелы
    text = text.replace("&amp;", " and ")
    text = text.replace("&lt;", " ")
    text = text.replace("&gt;", " ")

    # почемаем ссылки, упоминания
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"@\w+", " user ", text)

    # помечаем теги, знаки препин.
    text = re.sub(r"#(\w+)", r" \1 hashtag ", text)
    text = text.replace("!", " excl ")
    text = text.replace("?", " qmark ")

    # чистим от мусорных символов
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def keyword_to_tokens(keyword: str) -> str:
    keyword = keyword.replace("%20", " ").strip().lower()
    if not keyword:
        return "nokeyword"
    # отдельно помечаем ключевые слова, чтобы они не смешивались с обычными
    return " ".join("kw_" + token for token in keyword.split())


def preprocess_df_adv(df: pd.DataFrame) -> pd.DataFrame:
    if "target" in df.columns:
        y = df["target"]
        x = df.drop(["target", "location"], axis=1)
    else:
        y = None
        x = df.drop(["location"], axis=1)

    keywords = x["keyword"].fillna("").map(keyword_to_tokens)
    texts = x["text"].map(clean_text)

    x["combined"] = keywords + " " + texts
    return x["combined"], y

def main():
    df = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    test_ids = df_test["id"]

    preprocess = preprocess_df_adv
    x, y = preprocess(df)
    x_test, _ = preprocess(df_test)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        max_df=0.7,
        min_df=2,
        dtype=np.float32,
        norm='l2',
        use_idf=True,
        sublinear_tf=True,
    )

    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_val_tfidf = vectorizer.transform(x_val)

    model = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000,
        solver='saga',
        l1_ratio=0,
        C=1.0,
        verbose=1,
    )

    model.fit(x_train_tfidf, y_train)

    threshold, score = cnt_best_threshold(model, x_val_tfidf, y_val)
    print(f"Best threshold: {threshold:.4f}, best score: {score:.4f}")

    y_val_pred = (model.predict_proba(x_val_tfidf)[:, 1] >= threshold).astype(int)
    
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"Validation F1: {f1_score(y_val, y_val_pred):.4f}")
    print(f"Validation Recall: {recall_score(y_val, y_val_pred):.4f}")
    print(f"Validation Precision: {precision_score(y_val, y_val_pred):.4f}")
    print(f"\nClassification Report: \n{classification_report(y_val, y_val_pred)}")

    x_full_tfidf = vectorizer.fit_transform(x)
    x_test_tfidf = vectorizer.transform(x_test)
    model.fit(x_full_tfidf, y)

    y_pred = (model.predict_proba(x_test_tfidf)[:, 1] >= threshold).astype(int)

    submission = pd.DataFrame({
        "id": test_ids,
        "target": y_pred,
    })
    submission.to_csv("submission_new_feat.csv", index=False)
    print("good!")


if __name__ == "__main__":
    main()
