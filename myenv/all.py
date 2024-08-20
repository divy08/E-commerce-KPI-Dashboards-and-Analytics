import streamlit as st
import pickle
import joblib
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from fpdf import FPDF


# Load models
def load_models():
    models = {}
    with open(r'C:\Users\Lenovo\Desktop\CDAC DBDA\final\myenv\models\rf_model.pkl', 'rb') as file:
        models['sentiment_model'] = pickle.load(file)
    with open(r'C:\Users\Lenovo\Desktop\CDAC DBDA\final\myenv\models\vectorizer.pkl', 'rb') as file:
        models['vectorizer'] = pickle.load(file)
    models['time_series_model'] = joblib.load(r'C:\Users\Lenovo\Desktop\CDAC DBDA\final\myenv\models\forecasting_model.pkl')
    with open(r'C:\Users\Lenovo\Desktop\CDAC DBDA\final\myenv\models\recommendation_model.pkl', 'rb') as file:
        models['recommendation_model'] = pickle.load(file)
    with open(r'C:\Users\Lenovo\Desktop\CDAC DBDA\final\myenv\models\id_mapping.pkl', 'rb') as file:
        models['id_mapping'] = pickle.load(file)
    with open(r'C:\Users\Lenovo\Desktop\CDAC DBDA\final\myenv\models\decisionTree (1).pkl', 'rb') as file:
        models['churn_model'] = pickle.load(file)
    return models

# Load models
models = load_models()

# Function to display Sentiment Analysis
def sentiment_analysis():
    st.title("Sentiment Analysis App")
    user_input = st.text_area("Enter the text you want to analyze:")
    if st.button("Analyze"):
        if user_input:
            input_vectorized = models['vectorizer'].transform([user_input])
            prediction = models['sentiment_model'].predict(input_vectorized)[0]
            prediction_proba = models['sentiment_model'].predict_proba(input_vectorized)[0]
            st.write(f"Prediction: {'Positive' if prediction == 1 else 'Negative'}")
            st.write(f"Confidence: {prediction_proba[prediction]:.2f}")
        else:
            st.write("Please enter some text to analyze.")

# Function to display Time Series Forecasting
def time_series_forecasting():
    st.title("Time Series Forecasting App")
    st.sidebar.header('Input Parameters')
    periods = st.sidebar.slider("Number of Months to forecast", min_value=1, max_value=24, value=10, step=1)
    start_date = st.sidebar.date_input("Start Date", datetime.date.today())

    if st.button('Forecast'):
        with st.spinner('Generating forecast...'):
            future_dates = pd.date_range(start=start_date, periods=periods, freq='M')
            future_df = pd.DataFrame({'ds': future_dates})
            forecast = models['time_series_model'].predict(future_df)

            st.subheader('Forecasted Values')
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecast': forecast['yhat'],
                'Lower Bound': forecast['yhat_lower'],
                'Upper Bound': forecast['yhat_upper']
            })
            st.write(forecast_df)

            if hasattr(models['time_series_model'], 'history'):
                st.subheader('Historical Data')
                historical_df = models['time_series_model'].history
                st.write(historical_df)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=historical_df['ds'], y=historical_df['y'], mode='lines', name='Historical Data'))
                fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines', name='Forecast', line=dict(dash='dash')))
                fig.update_layout(title='Historical Data and Forecast', xaxis_title='Date', yaxis_title='Value')
                st.plotly_chart(fig)

            st.subheader('Forecast Plot')
            fig = px.line(forecast_df, x='Date', y='Forecast', title='Forecasted Values')
            fig.update_traces(mode='lines+markers')
            fig.update_layout(xaxis_title='Date', yaxis_title='Value')
            st.plotly_chart(fig)

            st.subheader('Confidence Intervals')
            fig_conf = go.Figure()
            fig_conf.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines', name='Forecast'))
            fig_conf.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Lower Bound'], mode='lines', name='Lower Bound', line=dict(dash='dash')))
            fig_conf.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Upper Bound'], mode='lines', name='Upper Bound', line=dict(dash='dash')))
            fig_conf.update_layout(title='Forecast with Confidence Intervals', xaxis_title='Date', yaxis_title='Value')
            st.plotly_chart(fig_conf)

            st.subheader('Forecast Summary Statistics')
            st.write(f"Minimum Forecast Value: {forecast_df['Forecast'].min()}")
            st.write(f"Maximum Forecast Value: {forecast_df['Forecast'].max()}")
            st.write(f"Average Forecast Value: {forecast_df['Forecast'].mean()}")

            csv = forecast_df.to_csv(index=False)
            st.download_button(label="Download CSV", data=csv, file_name='forecasted_data.csv', mime='text/csv')

# Function to display Recommendation System
def recommendation_system():
    st.title("Recommendation System")

    file_path = r'C:\Users\Lenovo\Desktop\CDAC DBDA\archive\joined_dataset.csv'
    df = pd.read_csv(file_path)
    df['product_name'] = df.apply(lambda row: convert_product_name(row), axis=1)
    df['customer_id'] = df['customer_id'].map(models['id_mapping'])

    selected_customer_id = st.selectbox("Select Customer ID", df['customer_id'].unique())

    if st.button("Get Recommendations"):
        recommendations = get_recommendations(models['recommendation_model'], df, selected_customer_id)
        st.write("Top 5 Recommended Products:")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")

def convert_product_name(row):
    category = row['product_category_name_english']
    product_id = str(row['product_id'])
    category_abbreviation = ''.join(word[:].upper() for word in category.split())
    index = int(''.join(filter(str.isdigit, product_id)))
    product_name = f'{category_abbreviation}_{index}'
    return product_name

def get_recommendations(algo, df, selected_customer_id):
    u_pid = df[df['customer_id'] == selected_customer_id]['product_name'].unique()
    product_ids = df['product_name'].unique()
    pids_to_predict = np.setdiff1d(product_ids, u_pid)
    testset = [[selected_customer_id, product_name, 0.] for product_name in pids_to_predict]
    predictions = algo.test(testset)
    pred_ratings = np.array([pred.est for pred in predictions])
    sorted_indices = np.argsort(pred_ratings)[::-1]
    top_recs = pids_to_predict[sorted_indices][:5]
    return top_recs

# Function to display Churn Prediction
def churn_prediction():
    df = pd.read_csv(r'C:\Users\Lenovo\Desktop\CDAC DBDA\archive\df.csv')

# Title
    st.title("Olist Customer Segmentation Analysis")


# EDA - Number of Customers in Each Segment
    st.write("### Number of Customers in Each Segment")

# Create a bar chart to show the number of customers in each segment
    fig, ax = plt.subplots(figsize=(10, 6))
    segment_counts = df['segment'].value_counts()
    sns.barplot(x=segment_counts.index, y=segment_counts.values, ax=ax)
    ax.set_title('Number of Customers in Each Segment')
    ax.set_xlabel('Segment')
    ax.set_ylabel('Number of Customers')
    # Save the plot to a file
    save_path = r'C:\Users\Lenovo\Desktop\CDAC DBDA\final\myenv\images\segment_plot.png'
    plt.savefig(save_path)

# Display the bar chart
    st.pyplot(fig)
    segment_counts = df['segment'].value_counts()
    segment_percentages = (segment_counts / segment_counts.sum()) * 100

# Display segment percentages
    st.write("### Percentage of Customers in Each Segment")
    st.table(segment_percentages)
    st.title('Churn Prediction')

    freight_value = st.number_input('Enter freight value:', min_value=0.0, step=1.0)
    price = st.number_input('Enter price:', min_value=0.0, step=1.0)
    monetary_value = st.number_input('Enter monetary value:', min_value=0.0, step=1.0)
    review_score = st.number_input('Enter review score: Range 1-5', min_value=1.0, max_value=5.0, step=1.0)
    payment_installments = st.number_input('Enter payment installments:', min_value=0.0, step=1.0)
    customer_state = st.number_input('Enter customer state: Centralwest:0, Northeastern:1, Northern:2, Southeastern:3, Southern:4', min_value=0, max_value=4, step=1)
    frequency = st.number_input('Enter frequency:', min_value=0.0, step=1.0)
    # Recommendations based on segments
    recommendations = {
        "Regular": "1] Encourage regular customers with loyalty programs and personalized offers.\n"
                   "2] Bundle products or offer multi-buy discounts to increase average order value.\n"
                   "3] Promote add-on products or accessories to increase basket size.",
        "Engaged": "1] Maintain engagement with regular updates and exclusive offers.\n"
                   "2] Offer loyalty rewards or exclusive discounts.\n"
                   "3] Provide early access to new products or special events.",
        "Inactive": "1] Re-engage inactive customers with special promotions and reminders.\n"
                    "2] Encourage repeat purchases through personalized follow-up emails with discount codes.\n"
                    "3] Suggest complementary products based on their recent purchase.\n"
                    "4] Highlight customer benefits for returning and making more purchases.\n"
                    "5] Offer incentives for returning, such as a limited-time discount or free shipping.\n"
                    "6] Survey to understand why they stopped purchasing and address their concerns."
    }
# Create PDF
    class PDF(FPDF):
      def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Customer Segmentation Insights', 0, 1, 'C')

      def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

      def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

      def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

      def add_recommendation(self, segment):
        self.chapter_title(f"Recommendations for {segment} Segment")
        self.chapter_body(recommendations[segment])

    pdf = PDF()
    pdf.add_page()

# Add plot
    pdf.image(r'C:\Users\Lenovo\Desktop\CDAC DBDA\final\myenv\images\segment_plot.png', x = 10, y = 20, w = 180)

# Add insights
    pdf.ln(100)
    for segment in df['segment'].unique():
        pdf.add_recommendation(segment)

# Save the PDF
# Save the PDF to a file
    pdf_save_path = r'C:\Users\Lenovo\Desktop\CDAC DBDA\final\myenv\images\Customer_Segmentation_Insights.pdf'
    pdf.output(pdf_save_path)
    # Provide download option for the PDF
    with open(pdf_save_path, "rb") as pdf_file:
        st.download_button(
            label="Download Customer Segmentation Insights PDF",
            data=pdf_file,
            file_name="Customer_Segmentation_Insights.pdf",
            mime="application/pdf"
        )

    print("PDF generated successfully!")
    input_data = {
        'freight_value': [freight_value],
        'price': [price],
        'monetary_value': [monetary_value],
        'payment_installments': [payment_installments],
        'review_score': [review_score],
        'customer_state': [customer_state],
        'customer_tenure': [frequency]
    }
    input_df = pd.DataFrame(input_data)

    st.subheader('Input Data:')
    st.write(input_df)

    if st.button('Predict Churn'):
        prediction = models['churn_model'].predict(input_df)
        prediction_proba = models['churn_model'].predict_proba(input_df)

        if prediction[0] == 1:
            st.subheader('Churn Prediction:')
            st.write('The customer is predicted to churn.')
        else:
            st.subheader('Churn Prediction:')
            st.write('The customer is predicted to not churn.')
        st.write(f"Prediction Probability: {prediction_proba[0][prediction[0]]:.2f}")
     # Load your dataset from the CSV file



# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", 
                                ["Sentiment Analysis", "Time Series Forecasting", 
                                 "Recommendation System", "Churn Prediction"])
# Render the selected app mode
if app_mode == "Sentiment Analysis":
    sentiment_analysis()
elif app_mode == "Time Series Forecasting":
    time_series_forecasting()
elif app_mode == "Recommendation System":
    recommendation_system()
elif app_mode == "Churn Prediction":
    churn_prediction()