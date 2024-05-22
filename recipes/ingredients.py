import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('../data/ingredients_list.csv')
data.rename(columns={'0': 'ingredients'}, inplace=True)

nlp = spacy.load('en_core_web_md')
# embed
embeddings = np.array([nlp(item).vector for item in data['ingredients']])

data[['recipes','id']] = pd.read_csv('../data/recipes_ingredients.csv')[['name','id']]
data.join(pd.DataFrame(embeddings, index=data.index))

# Use random forest to predict the recipes
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split the data
X_train, X_test, y_train, y_test = train_test_split(embeddings, data['id'], test_size=0.2, random_state=42)

# Train
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate
accuracy_score(y_test, y_pred)

# Visualize accuracy
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
plt.matshow(cm)
plt.colorbar()
plt.show()
