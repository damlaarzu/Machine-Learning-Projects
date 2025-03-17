import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Load Dataset
# The dataset contains SMS messages labeled as 'ham' (not spam) or 'spam'.
dataset_path = 'SMSSpamCollection'
df = pd.read_csv(dataset_path, sep='\t', names=['label', 'message'], header=None)

# Preprocessing
# Define a function to preprocess text by converting it to lovercase.
def preprocess_text(text):
    """Cleans and preprocesses text."""
    return text.lower()

# Apply the preprocessing function to the 'message' column.
df['message'] = df['message'].apply(preprocess_text)

# Convert labels to binary format for classification (ham -> 0, spam -> 1).
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# TF-IDF Vectorization
# Transform text messages into TF-IDF feture vectors.
vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')  # Using unigrams and removing stop words.
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Dataset Splitting
# Create a testing dataset that includes all spam messages and 100 random ham messages.
spam_messages = df[df['label'] == 1]  # All spam messages.
ham_messages = df[df['label'] == 0]  # All ham messages.

# Randomly sample 100 ham messages for the testing dataset.
test_ham = ham_messages.sample(100, random_state=42)
test_set = pd.concat([spam_messages, test_ham])

# Remaining ham messages are used for training.
train_ham = ham_messages.drop(test_ham.index)
train_set = pd.concat([train_ham, spam_messages])

# Generate feature matrices and labels for treining and testing.
X_train = vectorizer.transform(train_set['message'])
y_train = train_set['label']
X_test = vectorizer.transform(test_set['message'])
y_test = test_set['label']

# Train Logistic Regression Model
# Fit a logistic regression model on the training dataset.
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Testing
# Predict labels on the testing dataset.
y_pred = model.predict(X_test)

# Evaluation
# Calculate performance metrics for the model.
TP = ((y_test == 1) & (y_pred == 1)).sum()  # True Positives: Spam correctly identified.
TN = ((y_test == 0) & (y_pred == 0)).sum()  # True Negatives: Ham correctly identified.
FP = ((y_test == 0) & (y_pred == 1)).sum()  # False Positives: Ham incorrectly marked as spam.
FN = ((y_test == 1) & (y_pred == 0)).sum()  # False Negatives: Spam incorrectly marked as ham.

# Calculate acuracy of the model.
accuracy = accuracy_score(y_test, y_pred)

# Print performance metrics.
print("Performance Metrics:")
print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"Accuracy: {accuracy:.4f}")
