import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

data_list = data_dict['data']  # List of sequences
labels = np.asarray(data_dict['labels'])

# Validate and process samples
valid_samples = []
for i, sample in enumerate(data_list):
    try:
        np_sample = np.array(sample, dtype=np.float32)
        valid_samples.append(np_sample)
    except ValueError:
        print(f"Error with sample at index {i}: {sample}")

# Pad sequences to ensure consistent dimensions
max_len = max(len(sample) for sample in valid_samples)  # Find maximum length
data = pad_sequences(valid_samples, maxlen=max_len, padding='post', dtype='float32')

# Encode labels (ensure they start from 0)
le = LabelEncoder()
labels = le.fit_transform(labels)

# Check class distribution
from collections import Counter
print("Class distribution:", Counter(labels))

# Train-test split
try:
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels
    )
except ValueError as e:
    print("Train-test split failed:", str(e))
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True
    )

# Train and evaluate the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f'{score * 100}% of samples were classified correctly!')

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)