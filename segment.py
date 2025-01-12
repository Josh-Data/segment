import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Bank Customer Segmentation", layout="wide")

# Set page background color
st.markdown("""
    <style>
    .stApp {
        background-color: #2D3748;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

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
    
    # Get binary columns before dummies conversion
    binary_cols = ["HasCrCard", "Exited", "Complain", "IsActiveMember"]
    
    # Convert to dummies first time
    df = pd.get_dummies(df, drop_first=True)
    
    # Drop specified columns
    df = df.drop(columns=["Geography_Spain", "Geography_Germany", "Satisfaction Score",
                         "Card Type_PLATINUM", "Card Type_SILVER", "Card Type_GOLD"])
    
    # Convert to dummies second time and convert to int
    df = pd.get_dummies(df, drop_first=True).astype(int)
    
    # Define continuous and categorical columns
    conts = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary", "Point Earned"]
    cats = [col for col in df.columns if col not in conts + binary_cols]
    
    # Scale continuous variables
    df_conts = df[conts]
    df_binary = df[binary_cols + cats]
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_conts)
    df_scaled = pd.DataFrame(df_scaled, columns=conts)
    
    # Combine scaled continuous and binary variables
    df_full = pd.concat([df_scaled, df_binary], axis=1)
    
    return df_full, scaler, conts, binary_cols, cats, df_conts.mean()

st.markdown("""
# Key Features in the Dataset:

* **CreditScore**: A numerical representation of creditworthiness
* **Age**: Customer's age
* **Tenure**: Number of years the customer has been with the bank
* **Balance**: The account balance of the customer
* **NumOfProducts**: Number of products the customer has purchased
* **HasCrCard**: Indicates whether the customer has a credit card (1 = Yes, 0 = No)
* **EstimatedSalary**: Customer's estimated salary
* **Point Earned**: Loyalty points earned by the customer
* **Exited**: Indicates whether the customer left the bank (1 = Yes, 0 = No)
* **Complain**: Indicates whether the customer lodged complaints (1 = Yes, 0 = No)
* **Gender_Male**: Binary variable indicating gender (1 = Male, 0 = Female)
""")

# Calculate and show feature variances
st.subheader("Feature Variance Analysis")
st.write("After deleting un-needed columns, we can check the variance of each feature to get a better understanding of which columns will be truly helpful in segmentation.")

# Prepare data for variance analysis
df_variance_calc = df_raw.copy()
df_variance_calc = df_variance_calc.dropna()
df_variance_calc = df_variance_calc.drop(columns=["RowNumber", "CustomerId", "Surname", "Unnamed: 0"])
df_variance_calc = pd.get_dummies(df_variance_calc, drop_first=True)
variance_series = df_variance_calc.var().sort_values(ascending=False)
st.dataframe(variance_series)

st.write("""We can see above that some of the similar products such as card type all have similar variance so can be eliminated, 
as it won't help to differentiate between the segments.""")

# Show scaled data preview
st.subheader("Value Standardization via scaling")
st.write("""Now that we have the relevant features chosen, we need to standardize the values. This prevents large values such as 
the estimated salary column from overpowering the smaller numerical features like number of products. The scaled dataset will 
keep the same differential representations per feature column, but will allow the model to operate more effectively. 
The first five rows of the scaled data frame can be seen below:""")

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

# Sidebar for user input with non-scaled mean values
def get_user_input(conts, binary_cols, cats, feature_means):
    user_input = {}
    
    st.sidebar.subheader("Continuous Features")
    for feature in conts:
        mean_value = feature_means[feature]
        user_input[feature] = st.sidebar.number_input(
            f"Enter {feature}", 
            value=float(mean_value),
            help=f"Average value: {mean_value:.2f}"
        )
    
    st.sidebar.subheader("Binary Features (0 or 1)")
    for feature in binary_cols:
        user_input[feature] = st.sidebar.selectbox(
            f"Select {feature}",
            options=[0, 1],
            help="0 = No, 1 = Yes"
        )
    
    st.sidebar.subheader("Categorical Features")
    for feature in cats:
        user_input[feature] = st.sidebar.selectbox(
            f"Select {feature}",
            options=[0, 1],
            help="0 = No, 1 = Yes"
        )
    
    return pd.DataFrame([user_input])

# Main app logic
df_full, scaler, conts, binary_cols, cats, feature_means = preprocess_data(df_raw)

# Train KMeans model
model = KMeans(n_clusters=4, random_state=42)
model.fit(df_full)
df_full["cluster"] = model.predict(df_full)
df_cluster = df_full.groupby("cluster").agg("mean")

# Show elbow plot
st.subheader("Elbow Plot for Optimal Clusters")
st.write("Defining how many clusters to choose can be a bit more art than science. The elbow plot compares the number of segments and model inertia.")
st.write("Analyzing the plot requires looking for a pivot point where after a steep decrease, there are diminishing returns for loss of inertia. I see no obvious elbow here, although an argument could be made for 2. However, and this is more art than science, I think 4 is a reasonable choice as it provides a decent decrease in inertia and creates a decent amount of cluster segments that are easily discernable and provide unique characterisicts")
plot_elbow(df_full)

# Show cluster averages
st.subheader("Cluster Feature Averages")
st.write("After choosing 4 segments, we run the model and assign each customer a cluster. Then we can take the average feature values for each cluster, as seen below:")
st.dataframe(df_cluster)

# Show heatmap from saved image
st.subheader("Cluster Heatmap")
st.write("The cluster averages can be better seen via a correlation heat map")
try:
    st.image("heatmap.png", use_column_width=True)
except FileNotFoundError:
    st.error("Heatmap image not found. Please ensure 'heatmap.png' is in the same directory as the app.")

st.write(''' Now that we can see the clear differences between each segment, we can use this information to strategize about how to treat each customer segment, examples below:''')

# Customer Segment Analysis
st.markdown("""
## Cluster 0 - New Multi-Product Users

### Balance Growth Initiative
* Implement tiered interest rates based on balance milestones
* Create automated savings programs with bonus incentives
* Offer balance transfer promotions with competitive rates
* Develop "save to earn" programs linking spending to savings

### Early Retention Strategy
* Schedule 3-month and 6-month relationship review meetings
* Provide complimentary financial planning sessions
* Create early-stage customer feedback loops
* Implement predictive churn analytics for early intervention

### Digital Engagement Enhancement
* Push mobile banking app adoption with rewards
* Gamify financial literacy through app-based learning
* Create personalized digital onboarding journeys
* Implement smart notification systems for product usage

## Cluster 1 - High-Value Engaged Customers

### Premium Service Enhancement
* Launch exclusive VIP banking services
* Provide dedicated relationship managers
* Create invitation-only events and networking opportunities
* Offer premium card upgrades with enhanced benefits

### Product Diversification
* Develop tailored investment products
* Create bundled services with preferential pricing
* Introduce exclusive insurance products
* Offer specialized lending solutions

### Value Multiplication
* Design multi-product loyalty bonuses
* Create referral programs with premium rewards
* Implement family banking packages
* Develop cross-border banking solutions

## Cluster 2 - Loyal Low-Balance Customers

### Balance Growth Programs
* Create loyalty-based balance incentives
* Implement automated micro-savings features
* Develop round-up savings programs
* Offer special term deposit rates for tenure milestones

### Product Optimization
* Conduct product utilization analysis
* Streamline product portfolio for cost efficiency
* Create bundle discounts based on tenure
* Implement usage-based fee reductions

### Relationship Deepening
* Schedule annual financial health checks
* Provide tenure-based rewards and recognition
* Create family member referral programs
* Develop community banking initiatives

## Cluster 3 - High-Balance Low-Engagement

### Engagement Activation
* Launch personalized rewards programs
* Create high-value transaction incentives
* Implement dynamic point multiplication schemes
* Develop exclusive lifestyle partnerships

### Digital Adoption
* Provide one-on-one digital banking tutorials
* Create digital banking incentives
* Implement smart banking features
* Develop digital wealth management tools

### Product Education
* Create personalized product discovery journeys
* Implement AI-driven product recommendations
* Develop educational webinars and content
* Offer product usage incentives

## Implementation Timeline

### Immediate (1-3 months)
* Launch digital engagement initiatives
* Implement basic rewards programs
* Start personalized communication campaigns

### Medium-term (3-6 months)
* Develop and roll out new products
* Implement technology solutions
* Train staff on new initiatives

### Long-term (6-12 months)
* Launch premium services
* Implement advanced analytics
* Develop comprehensive loyalty programs
""")

# Predict segment for new customer
st.sidebar.subheader("Predict Customer Segment")
user_input = get_user_input(conts, binary_cols, cats, feature_means)

if st.sidebar.button("Predict Segment"):
    # Scale only the continuous features
    cont_input = user_input[conts]
    scaled_cont_input = scaler.transform(cont_input)
    
    # Replace the continuous features with their scaled versions
    prediction_input = user_input.copy()
    prediction_input[conts] = scaled_cont_input
    
    # Predict cluster
    predicted_cluster = model.predict(prediction_input)[0]
    
    st.sidebar.write(f"Predicted Segment: Cluster {predicted_cluster}")
    
    # Show recommendations
    st.sidebar.subheader("Recommended Actions:")
    recommendations = get_segment_recommendations(predicted_cluster)
    for rec in recommendations:
        st.sidebar.write(f"• {rec}")