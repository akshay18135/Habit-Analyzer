import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv(r"C:\Users\akshx\OneDrive\Desktop\habit-analyzer\data\habit_data.csv")

# Encode categorical columns
le = LabelEncoder()

for col in ['Day', 'Time', 'Activity', 'Mood', 'Category']:
    data[col] = le.fit_transform(data[col])

# Features & target
X = data[['Day', 'Time', 'Activity', 'Mood', 'Duration']]
y = data['Category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)