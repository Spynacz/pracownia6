# main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

files = {
    "static": "csv/feature_vectors_static.csv",
    "syscalls": "csv/feature_vectors_syscalls_frequency_5_Cat.csv",
    "binders": "csv/feature_vectors_syscallsbinders_frequency_5_Cat.csv",
    "syscall_list": "csv/syscall_unique.csv",
}

INPUT_CSV = files["binders"]
df = pd.read_csv(INPUT_CSV)
print("Wczytano:", INPUT_CSV)
print("Rozmiar:", df.shape)

class_mapping = {1: "Adware", 2: "Banking", 3: "SMS", 4: "Riskware", 5: "Benign"}

df["Class"] = df["Class"].map(class_mapping)

print("Pierwsze 5 wierszy:\n", df.head())
print("Typy danych:\n", df.dtypes)
target_name = "Class" if "Class" in df.columns else df.columns[-1]
print(
    "Klasy:", df[target_name].unique(), "\nLiczności:\n", df[target_name].value_counts()
)

fig, ax = plt.subplots(figsize=(12, 6))

sns.countplot(x="Class", data=df)
plt.xlabel("Klasa")
plt.ylabel("Liczność")
ax.set_title("Rozkład liczności klass w zbiorze", fontsize=15)
plt.savefig("class_distribution.png")

corr_matrix = df.drop(columns=[target_name]).corr(numeric_only=True)
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=False, fmt=".2f", linewidths=0.5, cmap="coolwarm")
plt.title("Macierz korelacji")
plt.savefig("corr_matrix.png", dpi=1200)


feature_cols = [c for c in df.columns if c != target_name]
X = df[feature_cols].values
y = df[target_name].values

# ANOVA (f_classif)
F, pvals = f_classif(X, y)
anova_sorted = sorted(zip(feature_cols, F, pvals), key=lambda x: -x[1])

chi_vals, chi_pvals = chi2(X, y)
chi2_sorted = sorted(zip(feature_cols, chi_vals, chi_pvals), key=lambda x: -x[1])

print("Top 10 cech wg ANOVA:")
for feat, stat, pval in anova_sorted[:10]:
    print(f"{feat}: stat={stat:.2f}, p={pval:.1e}")

print("\nTop 10 cech wg chi2:")
for feat, stat, pval in chi2_sorted[:10]:
    print(f"{feat}: stat={stat:.2f}, p={pval:.1e}")


def plot_features_boxplots(df, feature_names, target_col, save_plots=True):
    for feature in feature_names:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x=target_col, y=feature)
        plt.title(f"Rozkład cechy '{feature}' w klasach malware")
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_plots:
            filename = f"boxplot_{feature}.png"
            plt.savefig(filename)
        else:
            plt.show()
        plt.close()


top_anova_features = [feat for feat, _, _ in anova_sorted[:10]]
top_chi2_features = [feat for feat, _, _ in chi2_sorted[:10]]

# plot_features_boxplots(df, top_anova_features, target_name)
# plot_features_boxplots(df, top_chi2_features, target_name)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, solver="lbfgs"),
    "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True),
}

dataset_versions = {
    "raw": (X_train_std, X_test_std),
}

results = {}
for setname, (Xtr, Xte) in dataset_versions.items():
    results[setname] = {}
    print(f"\n*** Wyniki dla zbioru: {setname}")
    for mname, model in models.items():
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        acc = accuracy_score(y_test, y_pred)
        print(f"{mname}: acc={acc:.4f}")
        results[setname][mname] = {
            "accuracy": acc,
            "classification_report": classification_report(
                y_test, y_pred, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=sorted(df[target_name].unique()),
            yticklabels=sorted(df[target_name].unique()),
        )
        plt.title(f"Macierz pomyłek - {mname}")
        plt.xlabel("Przewidywana klasa")
        plt.ylabel("Prawdziwa klasa")
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_{mname}_multiclass.png")
        plt.close()

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_std, y_train)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
topn = 15
plt.figure(figsize=(7, 5))
plt.title("Top 15 ważnych cech (RandomForest)")
plt.barh(range(topn), importances[indices][:topn][::-1], color="teal")
plt.yticks(range(topn), [feature_cols[i] for i in indices[:topn]][::-1])
plt.xlabel("Ważność cechy")
plt.tight_layout()
plt.savefig("feature_importance_RF.png")
plt.close()

df["BinaryClass"] = df["Class"].apply(
    lambda x: "Benign" if x == "Benign" else "Malware"
)

print("\nRozkład klas binarnych:")
print(df["BinaryClass"].value_counts())

X = df[feature_cols].values
y_bin = df["BinaryClass"].values

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X, y_bin, stratify=y_bin, test_size=0.2, random_state=42
)

X_train_b_std = scaler.fit_transform(X_train_b)
X_test_b_std = scaler.transform(X_test_b)

rf_bin = RandomForestClassifier(n_estimators=100, random_state=42)
rf_bin.fit(X_train_b_std, y_train_b)
y_pred_b = rf_bin.predict(X_test_b_std)

print("\n=== Klasyfikacja binarna ===")
print("Accuracy:", accuracy_score(y_test_b, y_pred_b))
print(classification_report(y_test_b, y_pred_b, zero_division=0))

cm_bin = confusion_matrix(y_test_b, y_pred_b)
sns.heatmap(
    cm_bin,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Benign", "Malware"],
    yticklabels=["Benign", "Malware"],
)
plt.title("Macierz pomyłek - klasyfikacja binarna (RandomForest)")
plt.xlabel("Przewidywana klasa")
plt.ylabel("Prawdziwa klasa")
plt.tight_layout()
plt.savefig("confusion_matrix_RF_binary.png")
plt.close()

for setname in results:
    for mname in results[setname]:
        with open(f"{setname}_{mname}_report.txt", "w") as f:
            f.write(results[setname][mname]["classification_report"])

print("\nDone.")
