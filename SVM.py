import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score

# Load dataset (Ensure CSV file is correctly formatted)
df = pd.read_csv("breast-cancer.csv")  # Replace with actual dataset file

# Drop 'id' column if present
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)

# Convert categorical target ('diagnosis') to numerical (if needed)
if df['diagnosis'].dtype == 'object':  # If it's a string, encode it
    label_encoder = LabelEncoder()
    df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])  # Maps 'B' → 0, 'M' → 1

# Extract features and target
X = df.iloc[:, 1:].values  # Features (skip diagnosis column)
y = df.iloc[:, 0].values   # Target (diagnosis)

# Ensure target variable contains at least two unique classes
unique_classes = np.unique(y)
if len(unique_classes) < 2:
    raise ValueError(f"Error: Only one class found in target variable {unique_classes}. Check dataset labels.")

# Apply feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data ensuring both classes are represented
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train SVM with Linear and RBF Kernel
linear_svm = SVC(kernel='linear', C=1.0)
linear_svm.fit(X_train, y_train)

rbf_svm = SVC(kernel='rbf', C=1.0, gamma='scale')
rbf_svm.fit(X_train, y_train)

# Make predictions
y_pred_linear = linear_svm.predict(X_test)
y_pred_rbf = rbf_svm.predict(X_test)

# Print accuracy scores
print("Linear Kernel Accuracy:", accuracy_score(y_test, y_pred_linear))
print("RBF Kernel Accuracy:", accuracy_score(y_test, y_pred_rbf))

# Hyperparameter tuning for RBF kernel using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.1, 1, 10]
}

grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)

# Cross-validation evaluation
cv_scores = cross_val_score(SVC(kernel='rbf', C=grid_search.best_params_['C'],
                                gamma=grid_search.best_params_['gamma']),
                            X_train, y_train, cv=5)
print("Cross-Validation Accuracy:", np.mean(cv_scores))

# Function to visualize decision boundary
def plot_decision_boundary(model, X, y, title):
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.title(title)
    plt.xlabel("Radius Mean")
    plt.ylabel("Texture Mean")
    plt.show()

# Plot decision boundaries (if dataset has exactly 2 features)
if X.shape[1] == 2:
    plot_decision_boundary(linear_svm, X, y, "Linear Kernel Decision Boundary")
    plot_decision_boundary(rbf_svm, X, y, "RBF Kernel Decision Boundary")