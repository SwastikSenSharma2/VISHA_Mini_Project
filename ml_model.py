import random
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline

DATA_FILE = Path("SBE-profiles.xlsx")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ==============================================================================
# PART 1: REAL DATASET ML MODEL (SBE-profiles.xlsx -> Toxin Class & Retriever)
# ==============================================================================
def train_sbe_profiles_model():
    if not DATA_FILE.exists():
        print(f"[SKIP] {DATA_FILE} not found in {Path.cwd()}")
        return None, None

    df = pd.read_excel(DATA_FILE, sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").fillna("")

    df["primary_toxin"] = df["Specific target toxin class"].apply(
        lambda v: "Unspecified" if not str(v).strip() or str(v).strip().upper() == "N/A"
        else str(v).split(";")[0].strip()
    )

    # Keep labelled toxin classes with >= 4 rows for stratified evaluation
    labelled = df[df["primary_toxin"] != "Unspecified"].copy()
    counts = labelled["primary_toxin"].value_counts()
    valid_classes = counts[counts >= 4].index.tolist()
    train_df = labelled[labelled["primary_toxin"].isin(valid_classes)].copy()

    X = (
        train_df["Target"].astype(str) + " " +
        train_df["Product"].astype(str) + " " +
        train_df["Key features & challenges"].astype(str)
    )
    y = train_df["primary_toxin"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3500, ngram_range=(1, 2), stop_words="english")),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    print("\n=== MODEL 1: SBE-profiles.xlsx Text -> Toxin Class ===")
    print(f"Total rows in Excel: {len(df)} | Supervised training rows: {len(train_df)}")
    print(f"Classes: {valid_classes}")
    print(f"Holdout Accuracy: {accuracy_score(y_test, preds):.2%}")
    print(f"Holdout Macro F1: {f1_score(y_test, preds, average='macro', zero_division=0):.2%}")
    print(classification_report(y_test, preds, zero_division=0))

    # Refit on all labelled rows & save
    clf.fit(X, y)
    joblib.dump(clf, MODEL_DIR / "sbe_toxin_classifier.joblib")

    # Build TF-IDF Retriever across ALL rows (Available + Investigational)
    corpus = (
        df["Name"].astype(str) + " | " +
        df["Type"].astype(str) + " | " +
        df["Relevant species"].astype(str) + " | " +
        df["Specific target toxin class"].astype(str) + " | " +
        df["Key features & challenges"].astype(str)
    )
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    mat = vec.fit_transform(corpus)
    joblib.dump({"vectorizer": vec, "matrix": mat, "records": df.to_dict("records")}, MODEL_DIR / "sbe_retriever.joblib")

    return clf, (vec, mat, df)


# ==============================================================================
# PART 2: SYNTHETIC CLINICAL SYMPTOM -> TOXIN ML MODEL (VISHA Sec. X & XIV)
# ==============================================================================
FEATURES = [
    "ptosis", "ophthalmoplegia", "bulbar_palsy", "resp_distress", "descending_paralysis",
    "wbct20_abnormal", "spontaneous_bleeding", "bite_bleeding", "shock_aki",
    "progressive_swelling", "blistering_necrosis", "severe_local_pain",
    "myalgia", "myoglobinuria",
]

SYNDROME_PROFILES = {
    "3FTxs (Alpha-Neurotoxins)": [0.90, 0.85, 0.78, 0.55, 0.70, 0.05, 0.02, 0.05, 0.10, 0.25, 0.10, 0.20, 0.08, 0.02],
    "SVMPs (Metalloproteinases)": [0.05, 0.03, 0.02, 0.05, 0.02, 0.92, 0.85, 0.88, 0.55, 0.85, 0.72, 0.75, 0.15, 0.10],
    "SVSPs (Serine Proteases)": [0.04, 0.02, 0.02, 0.04, 0.02, 0.95, 0.78, 0.82, 0.40, 0.30, 0.12, 0.25, 0.05, 0.04],
    "PLA2s (Phospholipases A2)": [0.35, 0.20, 0.15, 0.30, 0.20, 0.25, 0.10, 0.15, 0.35, 0.45, 0.25, 0.40, 0.92, 0.85],
    "Hyaluronidase / Cytotoxins": [0.02, 0.01, 0.01, 0.02, 0.01, 0.08, 0.05, 0.10, 0.05, 0.92, 0.88, 0.90, 0.20, 0.05],
    "Dry Bite / Indeterminate": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.03, 0.01, 0.08, 0.02, 0.01],
}


def train_symptom_toxin_model():
    rng = random.Random(42)
    rows, labels = [], []
    for label, probs in SYNDROME_PROFILES.items():
        for _ in range(30):
            rows.append([1 if rng.random() < p else 0 for p in probs])
            labels.append(label)

    syn_df = pd.DataFrame(rows, columns=FEATURES)
    syn_df["target_toxin_class"] = labels
    syn_df.to_csv("visha_synthetic_clinical_dataset.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        syn_df[FEATURES], syn_df["target_toxin_class"], test_size=0.25, random_state=42, stratify=labels
    )

    nb = BernoulliNB(alpha=1.0)
    nb.fit(X_train, y_train)
    preds = nb.predict(X_test)

    print("\n=== MODEL 2: Synthetic Clinical Symptoms -> Toxin Class (180 Cases) ===")
    print(f"Holdout Accuracy: {accuracy_score(y_test, preds):.2%}")
    print(f"Holdout Macro F1: {f1_score(y_test, preds, average='macro', zero_division=0):.2%}")
    print(classification_report(y_test, preds, zero_division=0))

    nb.fit(syn_df[FEATURES], syn_df["target_toxin_class"])
    joblib.dump(nb, MODEL_DIR / "symptom_toxin_nb.joblib")
    return nb


# ==============================================================================
# PART 3: EXAMPLE END-TO-END INFERENCE DEMO
# ==============================================================================
if __name__ == "__main__":
    sbe_clf, retriever_bundle = train_sbe_profiles_model()
    symptom_nb = train_symptom_toxin_model()

    # Example symptom input (ptosiss + bulbar palsy present)
    sample_symptoms = pd.DataFrame([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], columns=FEATURES)
    probs = symptom_nb.predict_proba(sample_symptoms)[0]
    ranked = sorted(zip(symptom_nb.classes_, probs), key=lambda x: x[1], reverse=True)
    print("\n=== Demo Symptom Inference ===")
    for cls, p in ranked[:3]:
        print(f"  {cls}: {p:.1%}")

    if retriever_bundle is not None:
        vec, mat, df = retriever_bundle
        q_vec = vec.transform(["India Big Four polyvalent antivenom neurotoxic"])
        sims = cosine_similarity(q_vec, mat).flatten()
        print("\n=== Top Retrieved SBE-Profiles Entries ===")
        for idx in sims.argsort()[::-1][:3]:
            print(f"  [{df.iloc[idx]['Type']}] {df.iloc[idx]['Name']} (sim={sims[idx]:.3f})")