import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import uniform, randint

# ============================
# 1. Load and prepare data
# ============================
df = pd.read_csv("data.csv")  # <-- Replace with your dataset
target = "y"                  # <-- Update if your target column has a different name

df[target] = df[target].map({"yes": 1, "no": 0})  # Binary encoding

X = df.drop(columns=[target])
y = df[target]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ============================
# 2. Preprocessor
# ============================
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# ============================
# 3. Models & Search Spaces
# ============================
models_params = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {"clf__C": uniform(0.1, 10)}
    },
    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {
            "clf__n_estimators": randint(50, 300),
            "clf__max_depth": randint(3, 20),
            "clf__min_samples_split": randint(2, 10)
        }
    },
    "XGBoost": {
        "model": XGBClassifier(eval_metric='logloss', use_label_encoder=False),
        "params": {
            "clf__n_estimators": randint(50, 300),
            "clf__max_depth": randint(3, 10),
            "clf__learning_rate": uniform(0.01, 0.3),
            "clf__subsample": uniform(0.5, 0.5),
            "clf__colsample_bytree": uniform(0.5, 0.5)
        }
    }
}

# ============================
# 4. Train, tune & compare
# ============================
best_score = 0
best_model = None
best_name = ""

for name, mp in models_params.items():
    print(f"\n🔍 Training {name}...")
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', mp["model"])
    ])

    search = RandomizedSearchCV(
        pipeline, mp["params"],
        n_iter=20,
        scoring='roc_auc',
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)

    print(f"{name} Best ROC AUC: {search.best_score_:.4f} | Params: {search.best_params_}")

    if search.best_score_ > best_score:
        best_score = search.best_score_
        best_model = search.best_estimator_
        best_name = name

# ============================
# 5. Final test evaluation
# ============================
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n✅ Best model: {best_name} with ROC AUC: {best_score:.4f}")
print(f"📊 Test ROC AUC: {test_auc:.4f}")

# ============================
# 6. Save model
# ============================
joblib.dump(best_model, "model.pkl")
print("📦 Model saved to model.pkl")






