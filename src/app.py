import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pickle import dump

# Step 1: Loading the dataset
url = 'https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv'
data = pd.read_csv(url)

# Step 2: Study of variables and their content
# Assuming we are only interested in the 'review' and 'polarity' columns
X = data['review'].str.strip().str.lower()  # Preprocess text data
y = data['polarity']

# Step 2 continued: Divide dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform text into a word count matrix
vec_model = CountVectorizer(stop_words='english')
X_train = vec_model.fit_transform(X_train).toarray()
X_test = vec_model.transform(X_test).toarray()

# Step 3: Build Naive Bayes models
models = {
    'GaussianNB': GaussianNB(),
    'MultinomialNB': MultinomialNB(),
    'BernoulliNB': BernoulliNB()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")

# Step 4: Optimize the best Naive Bayes model with Random Forest
best_model_name = 'MultinomialNB'  # Replace with the best performing model
best_model = models[best_model_name]

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")


# Step 6: Explore other alternatives
# Initialize models
gnb_model = GaussianNB()
mnb_model = MultinomialNB()
bnb_model = BernoulliNB()
rf_model = RandomForestClassifier(random_state=42)

# Train Gaussian Naive Bayes
gnb_model.fit(X_train, y_train)
gnb_preds = gnb_model.predict(X_test)
gnb_accuracy = accuracy_score(y_test, gnb_preds)
print("GaussianNB Accuracy:", gnb_accuracy)

# Train Multinomial Naive Bayes
mnb_model.fit(X_train, y_train)
mnb_preds = mnb_model.predict(X_test)
mnb_accuracy = accuracy_score(y_test, mnb_preds)
print("MultinomialNB Accuracy:", mnb_accuracy)

# Train Bernoulli Naive Bayes
bnb_model.fit(X_train, y_train)
bnb_preds = bnb_model.predict(X_test)
bnb_accuracy = accuracy_score(y_test, bnb_preds)
print("BernoulliNB Accuracy:", bnb_accuracy)

# Train Random Forest
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)
print("Random Forest Accuracy:", rf_accuracy)

# Save the best model based on accuracy (assuming MultinomialNB in this case)
if mnb_accuracy >= max(gnb_accuracy, bnb_accuracy, rf_accuracy):
    dump(mnb_model, 'best_model.joblib')
    print("Saved MultinomialNB model as the best performer.")
elif gnb_accuracy >= max(mnb_accuracy, bnb_accuracy, rf_accuracy):
    dump(gnb_model, 'best_model.joblib')
    print("Saved GaussianNB model as the best performer.")
elif bnb_accuracy >= max(gnb_accuracy, mnb_accuracy, rf_accuracy):
    dump(bnb_model, 'best_model.joblib')
    print("Saved BernoulliNB model as the best performer.")
else:
    dump(rf_model, 'best_model.joblib')
    print("Saved Random Forest model as the best performer.")