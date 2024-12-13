#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import ast
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras import layers, Model, Input


# In[2]:


tf.config.set_visible_devices([], 'GPU')


# In[3]:


data=pd.read_csv('perfumes_sample.csv')
data.head()


# In[4]:


data.info()


# In[5]:


notes_data=pd.read_csv('Development/LLM_DATA/NOTES_COMPREHENSIVE.csv')
notes_data.head()


# In[6]:


def parse_notes(notes_str):
    try:
        return ast.literal_eval(notes_str)
    except:
        return []
data['notes'] = data['notes'].apply(parse_notes)


# In[7]:


data.head()


# In[8]:


data.info()


# In[9]:


note_enrichment = {}
for idx, row in notes_data.iterrows():
    name = str(row['Correct_Name']) if not pd.isnull(row['Correct_Name']) else str(row['Name'])
    scientific = row['Scientific_name'] if not pd.isnull(row['Scientific_name']) else ""
    other = row['Other_names'] if not pd.isnull(row['Other_names']) else ""
    group = row['Group'] if not pd.isnull(row['Group']) else ""
    odor = row['Odor_profile'] if not pd.isnull(row['Odor_profile']) else ""

    enriched_desc = f"{name}. Scientific: {scientific}. Other names: {other}. Group: {group}. Odor: {odor}."
    note_enrichment[name.lower()] = enriched_desc


# In[10]:


def enrich_notes(notes_list):
    enriched = []
    for note in notes_list:
        lower_note = note.lower()
        if lower_note in note_enrichment:
            enriched.append(note_enrichment[lower_note])
        else:
            # If no enrichment found, just use the note name
            enriched.append(note)
    return " ".join(enriched)


# In[11]:


data['enriched_text'] = data['notes'].apply(enrich_notes) + " " + data['description'].fillna("")


# In[12]:


data['enriched_text'].head()


# In[13]:


model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(data['enriched_text'].tolist(), convert_to_tensor=False)
embeddings = np.array(embeddings)


# In[14]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[15]:


range_n_clusters = [5, 10, 15, 20]  # Adjust this range as needed
best_score = -1
best_k = None

for n_clusters in range_n_clusters:
    kmeans_test = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans_test.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    print(f"For n_clusters = {n_clusters}, silhouette score is {score:.4f}")

    if score > best_score:
        best_score = score
        best_k = n_clusters

print(f"Best number of clusters: {best_k} with silhouette score {best_score:.4f}")


# In[16]:


kmeans = KMeans(n_clusters=best_k, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)


# In[17]:


pca = PCA(n_components=2, random_state=42)
reduced_embeddings = pca.fit_transform(embeddings)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Cluster Label')
plt.title('Fragrance Clusters Visualization (PCA Reduced)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


# In[18]:


data['cluster'] = cluster_labels


# In[19]:


pos_pairs = []
neg_pairs = []

cluster_map = {}
for i, c in enumerate(cluster_labels):
    cluster_map.setdefault(c, []).append(i)

for c, indices in cluster_map.items():
    if len(indices) > 1:
        # Generate pairs within the cluster
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                pos_pairs.append((indices[i], indices[j]))


# In[20]:


all_indices = list(range(len(data)))
for c, indices in cluster_map.items():
    other_indices = [idx for idx in all_indices if cluster_labels[idx] != c]
    neg_sample_size = min(len(indices), len(other_indices))
    neg_indices_sample = np.random.choice(other_indices, size=neg_sample_size, replace=False)
    pos_indices_sample = np.random.choice(indices, size=neg_sample_size, replace=True)

    for i1, i2 in zip(pos_indices_sample, neg_indices_sample):
        neg_pairs.append((i1, i2))


# In[21]:


pairs = [(p[0], p[1], 1) for p in pos_pairs] + [(n[0], n[1], 0) for n in neg_pairs]
np.random.shuffle(pairs)


# In[22]:


X1 = []
X2 = []
Y = []
for (i1, i2, label) in pairs:
    X1.append(embeddings[i1])
    X2.append(embeddings[i2])
    Y.append(label)

X1 = np.array(X1)
X2 = np.array(X2)
Y = np.array(Y)


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


print("X1 shape:", X1.shape, "X2 shape:", X2.shape, "Y shape:", Y.shape)


# In[25]:


subset_size = min(len(Y), 500000) 
X1_sub = X1[:subset_size]
X2_sub = X2[:subset_size]
Y_sub = Y[:subset_size]

X1_temp, X1_test, X2_temp, X2_test, Y_temp, Y_test=train_test_split(X1_sub, X2_sub, Y_sub, test_size=0.2, random_state=42)


X1_train, X1_val, X2_train, X2_val, Y_train, Y_val = train_test_split(X1_temp, X2_temp, Y_temp, test_size=0.1, random_state=42)


print("Training set class distribution:", np.bincount(Y_train))
print("Validation set class distribution:", np.bincount(Y_val))
print("Test set class distribution:", np.bincount(Y_test))


# In[26]:


embedding_dim = embeddings.shape[1]

# Define inputs
input_1 = Input(shape=(embedding_dim,))
input_2 = Input(shape=(embedding_dim,))


# In[27]:


dense = layers.Dense(128, activation='relu')
out_1 = dense(input_1)
out_2 = dense(input_2)


# In[28]:


l1_distance = tf.abs(out_1 - out_2)
similarity_score = layers.Dense(1, activation='sigmoid')(l1_distance)


# In[29]:


siamese_model = Model(inputs=[input_1, input_2], outputs=similarity_score)
siamese_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[30]:


siamese_model.summary(show_trainable=True)


# In[31]:


history = siamese_model.fit(
    [X1_train, X2_train],
    Y_train,
    batch_size=32,
    epochs=5,
    validation_data=([X1_val, X2_val], Y_val)
)


# In[32]:


val_preds_prob = siamese_model.predict([X1_val, X2_val])
val_preds = (val_preds_prob > 0.5).astype("int32").flatten()


# In[33]:


from sklearn.metrics import silhouette_score, confusion_matrix, classification_report
import seaborn as sns


# In[34]:


cm = confusion_matrix(Y_val, val_preds)
print("Confusion Matrix:")

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.tight_layout()
plt.show()


cr = classification_report(Y_val, val_preds, digits=4)
print("Classification Report:")
print(cr)


# In[35]:


train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(epochs, train_loss, 'o-', label='Train Loss')
plt.plot(epochs, val_loss, 'o--', label='Val Loss')
plt.title('Training & Validation Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, train_acc, 'o-', label='Train Accuracy')
plt.plot(epochs, val_acc, 'o--', label='Val Accuracy')
plt.title('Training & Validation Accuracy', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()


# In[36]:


def enrich_single_note(note):
    lower_note = note.lower()
    if lower_note in note_enrichment:
        return note_enrichment[lower_note]
    else:
        return note

def recommend_fragrances_by_notes(notes_list, top_k=5, gender=None):
    enriched_notes = [enrich_single_note(n) for n in notes_list]
    query_text = " ".join(enriched_notes)
    query_embedding = model.encode([query_text])[0]
    if gender is not None:
        candidate_df = data[data['gender'].str.lower() == gender.lower()]
    else:
        candidate_df = data
    candidate_embeddings = embeddings[candidate_df.index]
    sims = cosine_similarity(query_embedding.reshape(1, -1), candidate_embeddings)[0]

    top_indices = sims.argsort()[::-1][:top_k]
    recommended_indices = candidate_df.iloc[top_indices].index
    return data.loc[recommended_indices, 'title'].tolist()


# In[37]:


user_notes_input = input("Please enter notes separated by commas (e.g., Bergamot,Jasmine,Patchouli): ")
notes_query = [note.strip() for note in user_notes_input.split(',') if note.strip()]
user_gender_input = input("Please enter desired gender filter (Men/Women/Unisex) or press enter to skip: ").strip()

if user_gender_input.lower() not in ["men", "women", "unisex", ""]:
    print("Invalid gender input. Proceeding without gender filter.")
    user_gender_input = None
elif user_gender_input == "":
    user_gender_input = None
recommendations = recommend_fragrances_by_notes(notes_query, top_k=3, gender=user_gender_input)
print("Recommended Fragrances:", recommendations)


# In[52]:


from tensorflow.keras.utils import plot_model


# In[54]:


plot_model(siamese_model, show_shapes=True, show_layer_names=True, to_file='outer-model.png')


# In[51]:


import os
os.system("dot -V")


# In[ ]:




