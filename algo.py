import datetime
import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objs as go
import requests
from bs4 import BeautifulSoup
from streamlit_extras.stylable_container import stylable_container
import base64


@st.cache_data
def get_img_as_base64(file):
    with open(file,'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("ddd.jpg")

page_bg_img = f'''
<style>
[data-testid='stAppViewContainer']{{
background-image: url("data:image/png;base64,{img}");
background-size: cover;
}}
[data-testid='stHeader']{{
    background: rgba(0,0,0,0)
}}
[data-testid="stToolbar"]{{
    right: 2rem;
}}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown(
        """
        <style>
            body {
                background-color: #2d3748;
            }
            h1, h2, h3, h4, h5, h6 {
                color: orange;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to fetch stock data
def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Function to preprocess data
def preprocess_data(data):
    data['Date'] = data.index
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    features = ['Year', 'Month', 'Day']
    target = 'Close'
    X = data[features]
    y = data[target]
    return X, y

# Function to train model
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Function to predict stock prices
def predict_stock_prices(model, current_date):
    future_dates = pd.date_range(start=current_date, periods=30)  # One month
    future_features = pd.DataFrame({
        'Year': future_dates.year,
        'Month': future_dates.month,
        'Day': future_dates.day
    })
    future_predictions = model.predict(future_features)
    return future_dates, future_predictions

# Function to fetch S&P 500 tickers
def fetch_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"class": "wikitable"})
    rows = table.find_all("tr")[1:]  # Skip header row
    sp500_tickers = []
    for row in rows:
        ticker = row.find_all("td")[0].text.strip()
        sp500_tickers.append(ticker)
    return sp500_tickers

# Main function
def main():
    st.title('Stock Price Prediction')
        
    # Sidebar for selecting favorite stocks
    st.sidebar.title('Select Favorite Stocks')
    selected_favorites = st.sidebar.multiselect('Select stocks:', fetch_sp500_tickers())

    # Display selected stocks' graphs separately
    for selected_stock in selected_favorites:
        st.markdown(f'### Stock: {selected_stock}')
        current_date = datetime.datetime.now().date()
        start_date = current_date - datetime.timedelta(days=5*365)  # Last 5 years
        end_date = current_date

        # Fetch stock data for 5 years
        stock_data = fetch_stock_data(selected_stock, start_date, end_date)

        # Filter data for the last year
        stock_data_last_year = stock_data.tail(365)

        # Preprocess data
        X, y = preprocess_data(stock_data)

        # Train model
        model = train_model(X, y)

        # Predict stock prices for one month
        future_dates, future_predictions = predict_stock_prices(model, current_date)

        # Visualize predictions using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data_last_year.index, y=stock_data_last_year['Close'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Predicted', line=dict(dash='dash')))
        fig.update_layout(title=f'Actual vs. Predicted Stock Prices (Last Year + 1 Month) - {selected_stock}', xaxis_title='Date', yaxis_title='Stock Price')
        
        # Add CSS to create a highlighted box around the graph
        with stylable_container(
            key='column4',
            css_styles=""" 
        {
            
            border: 0.5px solid #009688;
            border-radius: 15px;
            padding: 0px;
            margin-bottom: 30px;
            background-color: #f0f0f0;
            animation: spin 10s linear infinite;
            box-shadow: 8px 8px 20px rgba(0.1, 0.1, 0.1, 0.5);
        }
        """,
        ):
        
        # Render graph inside the highlighted box
            st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
            st.plotly_chart(fig)
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
