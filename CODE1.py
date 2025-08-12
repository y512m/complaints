#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import re 


# # Data Preparation

# ## 1. Load the data

# In[3]:


df = pd.read_csv("Complaints - Approved Version.csv")


# ## 2. Display data summary

# In[4]:


df.head


# In[5]:


df.info()


# In[6]:


df.describe()


# ## 3. Select target column

# In[7]:


target_col = 'COMPLAINT_TYPE'


# ## 4. Handle missing values

# In[8]:


# Fill missing text values with empty string ""
text_cols = ['CASE_DESC', 'RESOLUTION_DESCRIPTION', 'RESOLUTION']
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].fillna('')

# Fill missing numerical values with median
num_cols = df.select_dtypes(include=np.number).columns.tolist()
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())


# ## 5. Create unified text column by combining text columns

# In[9]:


df['combined_text'] = df[text_cols].agg(' '.join, axis=1)


# ## 6. Clean the text

# In[10]:


def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Remove non-alphabetic characters (keep only a-z and spaces)
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces, keep single spaces
    text = text.strip()  # Remove leading/trailing spaces
    return text

df['cleaned_text'] = df['combined_text'].apply(clean_text)

# Display the first few rows of combined and cleaned text columns
print(df[['combined_text', 'cleaned_text']].head())


# # Modeling

# In[11]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


# ## 1. Convert text to numerical features using TfidfVectorizer

# In[12]:


vectorizer = TfidfVectorizer(max_features=10000)  # limit features to 10k (can adjust)
X = vectorizer.fit_transform(df['cleaned_text'])


# ## 2. Define target variable

# In[13]:


y = df[target_col]


# ## 3. Split data into train and test sets (80% train, 20% test), stratify by target

# In[14]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, stratify=y, random_state=42)


# ## 4. Define models

# In[15]:


models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
    'Passive Aggressive': PassiveAggressiveClassifier(max_iter=1000),
    'SVM': SVC(kernel='linear'),
}


# ## 5. Train, predict, and store results
# 

# In[16]:


results = []

for name, model in models.items():
    print(f'Training {name}...')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {acc:.4f}')
    print(classification_report(y_test, y_pred))

    results.append({
        'Model': name,
        'Accuracy': acc,
    })

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results)
print(results_df)


# In[17]:


print(classification_report(y_test, y_pred, zero_division=0))


# ## SMOTE
# I used **SMOTE (Synthetic Minority Over-sampling Technique)** instead of the basic training loop on the original data because our dataset suffers from **class imbalance**. Some classes have much fewer samples than others, which can cause models to be biased towards the majority classes and perform poorly on minority classes.
# 
# The original training approach trains models on the imbalanced data as is, which often leads to low recall or precision for underrepresented classes.
# 
# SMOTE helps by **creating synthetic samples for the minority classes**, effectively balancing the training data without simply duplicating existing samples. This improves the model's ability to learn patterns for all classes more fairly.
# 
# By applying SMOTE before training, the models are exposed to a more balanced dataset, which usually results in better performance on minority classes and reduces warnings about undefined metrics.
# 
# 

# In[19]:


from imblearn.over_sampling import SMOTE

# After splitting:
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Then train models on X_train_res, y_train_res instead of X_train, y_train
for name, model in models.items():
    print(f'Training {name}...')
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {acc:.4f}')
    print(classification_report(y_test, y_pred, zero_division=0))

    results.append({
        'Model': name,
        'Accuracy': acc,
    })


# # Evaluation

# In[21]:


import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Assume:
# models: dict of trained models on X_train_res, y_train_res (after SMOTE)
# X_test, y_test: original test data (string labels)
# vectorizer: fitted TfidfVectorizer

# Encode y_test to numeric labels for ROC curve and other uses
le = LabelEncoder()
y_test_encoded = le.fit_transform(y_test)  # e.g. 'Commercial'->0, 'Technical'->1

results = []

for name, model in models.items():
    print(f"Evaluating {name}...")
    y_pred = model.predict(X_test)

    # Calculate metrics (using original string labels)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred, labels=le.classes_)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Save the model
    joblib.dump(model, f"{name.replace(' ', '_').lower()}_model.pkl")

    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1
    })

# Accuracy comparison bar chart
results_df = pd.DataFrame(results)
plt.figure(figsize=(10,6))
sns.barplot(data=results_df, x='Model', y='Accuracy')
plt.title('Accuracy Comparison Between Models')
plt.ylim(0,1)
plt.show()

# ROC Curve (only for binary classification)
if len(le.classes_) == 2:
    plt.figure(figsize=(8,6))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:,1]  # probability for positive class (encoded 1)
        else:
            # Some models like SVM may not have predict_proba by default
            y_prob = model.decision_function(X_test)
            # Scale to [0,1]
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
        fpr, tpr, _ = roc_curve(y_test_encoded, y_prob, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

# Save the vectorizer
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')


# In[22]:


print(df[target_col].value_counts(normalize=True))


# In[ ]:




