import streamlit as st
import time
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pickle

# Set page config
st.set_page_config(page_title="House Price Prediction Model", page_icon="🏠", layout="centered")

# Custom Title with Styling
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700&display=swap');
        .stApp { background-color:#E6F5E9; }
        .title {
            font-size: 42px;
            font-weight: bold;
            text-align: center;
            color: #558B2F;
            font-family: 'Poppins', sans-serif;
        }
    </style>
    <div class="title"> 🏡 House Price Model</div>
    """,
    unsafe_allow_html=True
)

# Display Image
st.image("house-4737447_1280.png", caption="Your Dream Home", width=700)

# Welcome Message
st.write("Welcome to the **House Price Prediction Model**! Enter details below to estimate house prices instantly.")

# Load ML Model
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# User Input Fields
# User Input Fields
st.subheader("Enter House Details")



#prediction
# Load the trained ML model
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Define predict_price function before calling it
def predict_price(size, bedrooms):
    input_data = [[size, bedrooms]]
    predicted_price = model.predict(input_data)[0]
    return predicted_price

# ✅ Add unique 'key' argument
size = st.number_input("Choose House Size (sq ft)", min_value=500, max_value=10000, step=50, value=2000, key="size_input")
bedrooms = st.number_input("Choose Number of Bedrooms", min_value=1, max_value=10, step=1, value=3, key="bedrooms_input")

# Predict button
if st.button("🚀 Predict House Price"):
    with st.spinner("🔄 Analyzing market trends... Please wait!"):
        time.sleep(1.5)  # Simulate processing time
        
        price = predict_price(size, bedrooms)
        price_inr = price * 83  # Convert USD to INR

        # Categorize price range
        if price_inr < 1650000:
            category = "🟢 Affordable Home"
        elif 1650000 <= price_inr < 30000000:
            category = "🟡 Mid-Range Property"
        else:
            category = "🔴 Premium Luxury Home"
        
        st.success(f"🏡 Estimated House Price: **₹{round(price_inr, 2)}** {category}")

        # Linear Price Trend Graph
        st.subheader("📊 House Price Trend")
        x = np.array([500, 2000, 5000, 10000])
        y = np.array([100000, 300000, 700000, 1500000]) * 83  # Convert prices to INR
        plt.figure(figsize=(6,3))
        plt.plot(x, y, marker='o', linestyle='-', color='b', label='Price Trend')
        plt.axvline(size, color='r', linestyle='--', label='Selected Size')
        plt.xlabel("Size (sq ft)")
        plt.ylabel("Estimated Price (₹)")
        plt.legend()
        st.pyplot(plt)

        # Interactive 3D House Price Visualization
        st.subheader("📊 Interactive 3D House Price Visualization")
        sizes = np.linspace(500, 10000, 20)
        bedrooms_range = np.linspace(1, 10, 20)
        prices = (50000 + sizes * 100 + bedrooms_range * 5000) * 83  # Convert prices to INR

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=sizes, y=bedrooms_range, z=prices,
            mode='markers',
            marker=dict(size=5, color=prices, colorscale='Viridis', opacity=0.8)
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title='Size (sq ft)',
                yaxis_title='Bedrooms',
                zaxis_title='Price (₹)'
            ),
            title="3D House Price Trend",
        )
        st.plotly_chart(fig)
import streamlit as st

# Define quiz questions (including easy ones)
quiz_questions = [
    {
        "question": "Supervised learning is a type of machine learning where:",
        "options": [
            "The model learns from labeled data",
            "The model explores data without labels",
            "The model predicts random values",
            "The model works only with images"
        ],
        "answer": "The model learns from labeled data"
    },
     {
        "question": "Which algorithm is an example of supervised learning?",
        "options": ["K-Means Clustering", "Linear Regression", "Apriori Algorithm", "DBSCAN"],
        "answer": "Linear Regression"
    },
    {
        "question": "In supervised learning, the model learns from?",
        "options": ["Unlabeled data", "Labeled data", "Random guesses", "Noise"],
        "answer": "Labeled data"
    },
    {
        "question": "Which metric is commonly used to evaluate regression models?",
        "options": ["Accuracy", "Mean Squared Error (MSE)", "Silhouette Score", "F1 Score"],
        "answer": "Mean Squared Error (MSE)"
    },
    {
        "question": "Which of these is an example of supervised learning?",
        "options": [
            "Sorting emails as spam or not spam",
            "Grouping customers into clusters",
            "Finding frequent shopping patterns",
            "Detecting outliers in a dataset"
        ],
        "answer": "Sorting emails as spam or not spam"
    },
    {
        "question": "Which algorithm is commonly used for classification?",
        "options": [
            "Linear Regression",
            "K-Nearest Neighbors (KNN)",
            "K-Means Clustering",
            "Apriori Algorithm"
        ],
        "answer": "K-Nearest Neighbors (KNN)"
    },
    {
        "question": "In supervised learning, what is the purpose of a 'label'?",
        "options": [
            "To name the dataset",
            "To give correct answers for training",
            "To group similar data points",
            "To increase dataset size"
        ],
        "answer": "To give correct answers for training"
    },
    {
        "question": "Which dataset is commonly used for supervised learning?",
        "options": [
            "MNIST (Handwritten Digits)",
            "Wikipedia Articles",
            "Random Number Sequences",
            "Unstructured Text Data"
        ],
        "answer": "MNIST (Handwritten Digits)"
    },
    {
        "question": "What is the main goal of supervised learning?",
        "options": [
            "To create random predictions",
            "To learn from labeled data and make predictions",
            "To group similar data without labels",
            "To find hidden patterns in data"
        ],
        "answer": "To learn from labeled data and make predictions"
    },
    {
        "question": "Which of these is a supervised learning algorithm?",
        "options": [
            "Decision Trees",
            "DBSCAN Clustering",
            "Principal Component Analysis (PCA)",
            "Reinforcement Learning"
        ],
        "answer": "Decision Trees"
    }
]

# Store user responses
user_answers = {}

st.subheader("🧠 Supervised Learning Quiz")

# Loop through questions
for idx, q in enumerate(quiz_questions):
    st.write(f"**Q{idx+1}: {q['question']}**")
    user_answers[idx] = st.radio("", q["options"], key=f"q{idx}")

# Submit button
if st.button("Check Answers"):
    score = 0
    st.subheader("📊 Quiz Results")

    for idx, q in enumerate(quiz_questions):
        if user_answers[idx] == q["answer"]:
            st.write(f"✅ Q{idx+1}: Correct!")
            score += 1
        else:
            st.write(f"❌ Q{idx+1}: Incorrect. The correct answer is **{q['answer']}**.")

    st.write(f"🎯 Your Final Score: **{score}/{len(quiz_questions)}**")

    # Trigger balloons effect if full score
    if score == len(quiz_questions):
        st.balloons()
        st.success("🎉 Perfect Score! Well done! 🎊")

# Footer
st.markdown(
    """
    <style>
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        .footer {
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            padding: 20px;
            animation: fadeIn 2s ease-in-out;
        }
    </style>
    
    <div class="footer">
        🚀 <span style="color: #ff5733;">Unlock the Future of Home Buying!</span>  
        This <span style="color: #2c3e50;">AI-Powered House Price Predictor</span>  
        showcases <span style="color: #27ae60;">Machine Learning</span> in action—helping you make  
        smarter decisions because <span style="color:#f39c12;">your dream home starts with the right prediction! 🏡📊</span>
    </div>
    """,
    unsafe_allow_html=True
)
