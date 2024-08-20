import streamlit as st
import joblib
import pandas as pd
import datetime
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Load the pickled time series model
model = joblib.load(r'C:\Users\Lenovo\Desktop\CDAC DBDA\final\myenv\models\forecasting_model.pkl')

# Title of the app
st.title("Time Series Forecasting App")

# Instructions
st.write("""
    This app allows you to make future predictions using a time series model.
    Enter the required parameters and the app will provide the forecasted values.
    Adjust the forecast period and start date using the sidebar inputs.
""")

# Sidebar input for user data
st.sidebar.header('Input Parameters')

def user_input_parameters():
    # Example: Number of periods to forecast
    periods = st.sidebar.slider("Number of Months to forecast", min_value=1, max_value=24, value=10, step=1)

    # Start date input
    start_date = st.sidebar.date_input("Start Date", datetime.date.today())

    return periods, start_date

periods, start_date = user_input_parameters()

# Button to trigger forecasting
if st.button('Forecast'):
    with st.spinner('Generating forecast...'):
        # Generating a date range for the forecast
        future_dates = pd.date_range(start=start_date, periods=periods, freq='M')

        # Prepare future dataframe for Prophet
        future_df = pd.DataFrame({'ds': future_dates})

        # Making predictions
        forecast = model.predict(future_df)

        # Display the results
        st.subheader('Forecasted Values')
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecast': forecast['yhat'],
            'Lower Bound': forecast['yhat_lower'],
            'Upper Bound': forecast['yhat_upper']
        })
        st.write(forecast_df)

        # Historical data visualization (if available)
        if hasattr(model, 'history'):
            st.subheader('Historical Data')
            historical_df = model.history
            st.write(historical_df)

            # Plot historical data and forecast
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=historical_df['ds'], y=historical_df['y'], mode='lines', name='Historical Data'))
            fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines', name='Forecast', line=dict(dash='dash')))
            fig.update_layout(title='Historical Data and Forecast', xaxis_title='Date', yaxis_title='Value')
            st.plotly_chart(fig)

        # Plotting the forecast with Plotly
        st.subheader('Forecast Plot')
        fig = px.line(forecast_df, x='Date', y='Forecast', title='Forecasted Values')
        fig.update_traces(mode='lines+markers')
        fig.update_layout(xaxis_title='Date', yaxis_title='Value')
        st.plotly_chart(fig)

        # Confidence intervals
        st.subheader('Confidence Intervals')
        fig_conf = go.Figure()
        fig_conf.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines', name='Forecast'))
        fig_conf.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Lower Bound'], mode='lines', name='Lower Bound', line=dict(dash='dash')))
        fig_conf.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Upper Bound'], mode='lines', name='Upper Bound', line=dict(dash='dash')))
        fig_conf.update_layout(title='Forecast with Confidence Intervals', xaxis_title='Date', yaxis_title='Value')
        st.plotly_chart(fig_conf)

        # Displaying summary statistics
        st.subheader('Forecast Summary Statistics')
        st.write(f"Minimum Forecast Value: {forecast_df['Forecast'].min()}")
        st.write(f"Maximum Forecast Value: {forecast_df['Forecast'].max()}")
        st.write(f"Average Forecast Value: {forecast_df['Forecast'].mean()}")

        # Download option for forecasted data
        st.subheader('Download Forecast Data')
        csv = forecast_df.to_csv(index=False)
        st.download_button(label="Download CSV", data=csv, file_name='forecasted_data.csv', mime='text/csv')
