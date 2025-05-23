import numpy as np
import tensorflow as tf
import fasttext
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def extract_embeddings(words, model):
    embeddings = []
    for word in words:
        try:
            embedding = model.get_word_vector(word)
        except Exception:
            embedding = model.get_word_vector('<unk>')
        embeddings.append(embedding)
    return np.array(embeddings)

def build_classifier(input_dim):
    inputs = tf.keras.Input(shape=(input_dim,))
    
    x = tf.keras.layers.Dense(1024, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    shortcut = x
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, shortcut])

    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    shortcut = x
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, shortcut])

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

# model_path = 'Models/cc.sk.300.bin'
# model_path = 'Models/custom_model.bin'
model_path = './Models/wiki_sk.bin'
raw_data = pd.read_csv('Dátové_sady/candidates_dataset.csv')
labels = raw_data['Target']
words = raw_data['Word']

ft_model = fasttext.load_model(model_path)

X = extract_embeddings(words, ft_model)
y = np.array(labels)

pca = PCA(n_components=100)
X_pca = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.25, random_state=42, stratify=y)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

model = build_classifier(X_pca.shape[1])

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',    
    factor=0.2,              
    patience=5,            
    min_delta=0.001,
    min_lr=1e-6,
    verbose=1               
)

history = model.fit(
    X_train, 
    y_train, 
    validation_split=0.2,
    epochs=50,        
    batch_size=32,            
    callbacks=[reduce_lr],   
    verbose=1
)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

predictions = model.predict(X_test)
predicted_labels = (predictions > 0.5).astype(int)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

cm = confusion_matrix(y_test, predicted_labels)
tn, fp, fn, tp = cm.ravel()

tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
tnr = tn / (tn + fp) if (tn + fp) > 0 else 0 
balanced_acc = (tpr + tnr) / 2

precision = precision_score(y_test, predicted_labels)
recall = recall_score(y_test, predicted_labels)
f1 = f1_score(y_test, predicted_labels)

output_file = 'FastTextVysledky_FastText_vyladeny.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("=== Výsledky klasifikácie neologizmov ===\n\n")
    f.write("1. Matica zámien (Confusion Matrix):\n")
    f.write(f"{cm}\n\n")
    f.write("2. Vyvážená presnosť (Balanced Accuracy):\n")
    f.write(f"TPR (True Positive Rate): {tpr:.4f}\n")
    f.write(f"TNR (True Negative Rate): {tnr:.4f}\n")
    f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n\n")
    f.write("3. Precision, Recall, F1-skóre:\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1-skóre:  {f1:.4f}\n")

print(f"\nVýsledky boli uložené do súboru: {output_file}")

if 'lr' in history.history:
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['lr'])
    plt.title('Learning Rate During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)
    plt.show()     