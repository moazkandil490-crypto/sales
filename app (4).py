import gradio as gr
import joblib
import numpy as np
import pandas as pd

# Load the saved model and scaler
scaler = joblib.load('scaler.joblib')
cluster_model = joblib.load('cluster_model.joblib')

# Define cluster profiles based on our previous EDA
cluster_descriptions = {
    0: "Cluster 0: Low-value segment (transactions characterized by smaller quantities, lower unit prices, and overall low sales values).",
    1: "Cluster 1: High-value segment (transactions characterized by larger quantities, premium unit prices, and high sales values).",
    2: "Cluster 2: Average-value segment (transactions characterized by lower quantities but premium unit prices, producing mid-tier sales values)."
}

def predict_cluster(quantity, price, sales):
    # 1. Structure input into a DataFrame with the same column names as fitted
    input_data = pd.DataFrame([[quantity, price, sales]], columns=['QUANTITYORDERED', 'PRICEEACH', 'SALES'])
    
    # 2. Scale features using the loaded scaler
    input_scaled = scaler.transform(input_data)
    
    # 3. Predict cluster assignment (using our hierarchical clustering model)
    # NOTE: Since AgglomerativeClustering does not have a predict method, we typically assign new points 
    # to the nearest cluster representative (centroid) in scaled feature space.
    centroids = [
        [-0.114, -1.218, -0.669],  # Approximated scaled centroids from our clustered dataset
        [ 0.449,  0.742,  1.326],
        [-0.743,  0.528, -0.217]
    ]
    
    distances = [np.linalg.norm(input_scaled[0] - c) for c in centroids]
    predicted_cluster = np.argmin(distances)
    
    return f"Assigned Cluster: {predicted_cluster}\n\n{cluster_descriptions[predicted_cluster]}"

# Build the Gradio UI
interface = gr.Interface(
    fn=predict_cluster,
    inputs=[
        gr.Number(label="Quantity Ordered (e.g., 30)", value=30),
        gr.Number(label="Price Each (e.g., 95.00)", value=95.00),
        gr.Number(label="Sales (e.g., 2800.00)", value=2800.00)
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="Customer Sales Transaction Clustering",
    description="Input transaction values to assign the record to one of the 3 hierarchical clusters."
)

if __name__ == "__main__":
    interface.launch()
