import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Bank Customer Segmentation", layout="wide")

# Title and introduction
st.title("Bank Customer Segmentation")
st.write("""Using a banks dataset, we can segment the customers into different groups based on their characteristics. 
Once we have a segmentation, we achieve the following benefits:
1. Have a set of based based actions, such as which products to offer
2. Use the Kmeans algorithm to predict which segment a NEW customer belongs to by manually inputting feature values.""")

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        df_raw = pd.read_csv("df_raw.csv")
        return df_raw
    except FileNotFoundError:
        st.error("Error: df_raw.csv not found. Please make sure the file is in the correct location.")
        return None

# Load raw data
df_raw = load_data()

# Show raw data
st.subheader("Raw Data Preview")
st.write("Here are the first few rows of the raw dataframe. We can see right away there are some irrelevant columns that we can drop such as RowNumber, CustomerID, and Surname.")
st.dataframe(df_raw.head())

def preprocess_data(df):
    # Drop initial columns
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
    
    # Convert to dummies first time
    df = pd.get_dummies(df, drop_first=True)
    
    # Drop specified columns
    df = df.drop(columns=["Geography_Spain", "Geography_Germany", "Satisfaction Score",
                         "Card Type_PLATINUM", "Card Type_SILVER", "Card Type_GOLD", 
                         "IsActiveMember"])
    
    # Convert to dummies second time and convert to int
    df = pd.get_dummies(df, drop_first=True).astype(int)
    
    # Scale continuous variables
    conts = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", 
             "HasCrCard", "EstimatedSalary", "Point Earned"]
    df_conts = df[conts]
    df_binary = df.drop(columns=conts)
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_conts)
    df_scaled = pd.DataFrame(df_scaled, columns=conts)
    
    # Combine scaled continuous and binary variables
    df_full = pd.concat([df_scaled, df_binary], axis=1)
    
    return df_full, scaler, conts, df_scaled

# Elbow plot
def plot_elbow(df_scaled):
    clusters = range(1, 10)
    inertia = []
    for cluster in clusters:
        model = KMeans(n_clusters=cluster, random_state=42)
        model.fit(df_scaled)
        inertia.append(model.inertia_)
    
    fig, ax = plt.subplots()
    plt.plot(clusters, inertia, marker="o", color="black")
    plt.title("Elbow")
    plt.grid()
    
    # Add axis labels and adjust scale
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.ticklabel_format(style='plain', axis='y')
    
    st.pyplot(fig)

# Heatmap
def plot_heatmap(df_cluster):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_cluster, cmap="coolwarm", annot=True, fmt=".2f")
    st.pyplot(fig)

# Segment predictor
def get_segment_recommendations(cluster):
    recommendations = {
        0: [
            "Focus on increasing engagement through rewards program education",
            "Offer savings accounts or deposit products to build balances",
            "Consider entry-level investment products",
            "Cross-sell based on their existing product mix"
        ],
        1: [
            "Premium/VIP banking services",
            "Wealth management and investment advisory services",
            "Exclusive credit card offerings with enhanced rewards",
            "Personalized relationship management"
        ],
        2: [
            "Loyalty rewards or recognition programs",
            "Debt consolidation products",
            "Financial advisory services to help build wealth",
            "Retirement planning products"
        ],
        3: [
            "Educational campaigns about rewards program benefits",
            "Premium products that align with their higher balance profiles",
            "Digital banking tools to increase engagement",
            "Targeted rewards program promotions"
        ]
    }
    return recommendations[cluster]

# Sidebar for user input
st.sidebar.header("Input Customer Features")

def get_user_input(conts):
    user_input = {}
    for feature in conts:
        user_input[feature] = st.sidebar.number_input(f"Enter {feature}", value=0.0)
    return pd.DataFrame([user_input])

# Main app logic
df_full, scaler, conts, df_scaled = preprocess_data(df_raw)

# Train KMeans model
model = KMeans(n_clusters=4, random_state=42)
model.fit(df_full)
df_full["cluster"] = model.predict(df_full)
df_cluster = df_full.groupby("cluster").agg("mean")

# Show elbow plot
st.subheader("Elbow Plot for Optimal Clusters")
st.write("Defining how many clusters to choose can be a bit more art than science. The elbow plot compares the number of segments and model inertia.")
plot_elbow(df_scaled)

# Show cluster averages
st.subheader("Cluster Feature Averages")
st.write("Average feature values for each cluster:")
st.dataframe(df_cluster)

# Show heatmap
st.subheader("Cluster Heatmap")
st.write("Heatmap visualization of cluster characteristics:")
plot_heatmap(df_cluster)

# Predict segment for new customer
st.sidebar.subheader("Predict Customer Segment")
user_input = get_user_input(conts)

if st.sidebar.button("Predict Segment"):
    # Scale the user input
    scaled_input = scaler.transform(user_input)
    
    # Predict cluster
    predicted_cluster = model.predict(scaled_input)[0]
    
    st.sidebar.write(f"Predicted Segment: Cluster {predicted_cluster}")
    
    # Show recommendations
    st.sidebar.subheader("Recommended Actions:")
    recommendations = get_segment_recommendations(predicted_cluster)
    for rec in recommendations:
        st.sidebar.write(f"• {rec}")