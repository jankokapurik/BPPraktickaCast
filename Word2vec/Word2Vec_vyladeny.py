import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.decomposition import PCA

def extract_embeddings(words, w2v_model, subword_size=3):
    embeddings = []
    valid_words = []

    def get_subwords(word, size):
        return [word[i:i+size] for i in range(len(word) - size + 1)]
    count_skipped = 0
    for word in words:
        if word in w2v_model:
            embeddings.append(w2v_model[word])
            valid_words.append(word)
        else:
            subwords = get_subwords(word, subword_size)
            subword_vectors = [w2v_model[subword] for subword in subwords if subword in w2v_model]
            
            if subword_vectors:
                embeddings.append(np.mean(subword_vectors, axis=0))
                valid_words.append(word)
            else:
                count_skipped += 1
    print(f"Skipped {count_skipped} words with no subwords in vocabulary.")
    return np.array(embeddings), valid_words

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

dataset_path = './Datasets/Dataset_SubstringWords.txt' 
model_path = './Models/vec-sk-cbow-lemma'         

raw_data = pd.read_csv('Dátové_sady/candidates_dataset.csv')
labels = raw_data['Target']
words = raw_data['Word']

w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
print("Loaded Word2Vec model.")

X, valid_words = extract_embeddings(words, w2v_model, subword_size=3)

word_label_mapping = dict(zip(words, labels))
y = np.array([int(word_label_mapping[word]) for word in valid_words if word in word_label_mapping])

print(f"Dataset size: X shape: {X.shape}, y shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pca = PCA(n_components=100)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

model = build_classifier(input_dim=X_train.shape[1])

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
    batch_size=32, 
    callbacks=[reduce_lr],    
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy}")

y_pred_probs = model.predict(X_test)
predicted_labels = (y_pred_probs > 0.5).astype(int).flatten()

cm = confusion_matrix(y_test, predicted_labels)
tn, fp, fn, tp = cm.ravel()

tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
tnr = tn / (tn + fp) if (tn + fp) > 0 else 0 
balanced_acc = (tpr + tnr) / 2

precision = precision_score(y_test, predicted_labels)
recall = recall_score(y_test, predicted_labels)
f1 = f1_score(y_test, predicted_labels)

output_file = 'Word2vec/Vysledky_Word2Vec_vyladeny.txt'
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