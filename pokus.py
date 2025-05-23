# Import necessary libraries
import numpy as np
import gensim
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load your pre-trained word embeddings
# Replace 'path/to/your/embeddings.txt' with the actual path to your embeddings file
# If your embeddings are in binary format (e.g., Word2Vec), set binary=True

print("Loading word embeddings...")
embeddings_path = "C:/Users/janko/Documents/Skola/TUKE/3/ZS/BP/Prakticka cast/w2v_cc_300d_sk_3.4.1_3.0_1647457801748/"
embeddings_model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary=False)
print("Word embeddings loaded successfully!")

# Prepare your data
# words: list of words to classify
# labels: list of labels corresponding to each word (1 for neologism, 0 for non-neologism)

# Example: Load your words and labels from a CSV file or any source
# Ensure that your labels are integers (0 or 1)

# For demonstration, let's assume we have the following lists
words = ['innovate', 'selfie', 'algorithm', 'vlog', 'blockchain']  # Replace with your words
labels = [0, 1, 0, 1, 1]  # Corresponding labels

# Filter out words not present in the embeddings vocabulary
print("Filtering words not found in embeddings...")
filtered_words = []
filtered_labels = []
word_vectors = []

for word, label in zip(words, labels):
    if word in embeddings_model:
        filtered_words.append(word)
        filtered_labels.append(label)
        word_vectors.append(embeddings_model[word])
    else:
        print(f"Word '{word}' not found in embeddings, skipping.")

# Ensure we have data to work with
if not word_vectors:
    raise ValueError("None of the words were found in the embeddings model.")

# Convert lists to NumPy arrays
X = np.array(word_vectors)
y = np.array(filtered_labels)

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build the neural network model
print("Building the neural network model...")
model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
print("Compiling the model...")
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
print("Training the model...")
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=8,
    validation_split=0.1,
    verbose=1
)

# Evaluate the model on the test data
print("Evaluating the model...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Make predictions
print("Making predictions...")
predictions = model.predict(X_test)
predicted_labels = (predictions > 0.5).astype(int).flatten()

# Show classification report
print("\nClassification Report:")
print(classification_report(y_test, predicted_labels))

# Visualize Training Progress
print("Visualizing training progress...")

# Plot accuracy
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Show some prediction results
print("\nSample Predictions:")
for idx in range(len(X_test)):
    actual_word = filtered_words[idx]
    actual_label = y_test[idx]
    predicted_label = predicted_labels[idx]
    print(f"Word: {actual_word}, Actual: {actual_label}, Predicted: {predicted_label}")
