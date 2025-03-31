import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step 1: Create Sample House Price Data
data = {
    "Size (sq ft)": [1500, 1800, 1200, 2200, 1700, 1600, 1400, 2000, 2500, 3000],
    "Bedrooms": [3, 4, 2, 5, 3, 3, 2, 4, 5, 6],
    "Price ($)": [250000, 320000, 190000, 410000, 280000, 260000, 220000, 350000, 450000, 550000]
}
df = pd.DataFrame(data)

# Step 2: Prepare Training Data
X = df[["Size (sq ft)", "Bedrooms"]]
y = df["Price ($)"]

# Step 3: Split Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Save the Model as `house_price_model.pkl`
model_path = r"C:\Users\ADMIN\Machine Learning\house_price_model.pkl"  # Adjust path if needed
with open(model_path, "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained and saved successfully as 'house_price_model.pkl'!")
