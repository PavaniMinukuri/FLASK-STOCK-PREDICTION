import os
from flask import Flask, request, render_template, jsonify
import yfinance as yf
import logging
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Use environment variable for security
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_secret_key')

def extracting_data(symbol):
    """Fetch stock data using the provided symbol and clean it."""
    end_date = datetime.now().strftime('%Y-%m-%d')
    df = yf.download(symbol, start="2000-01-01", end=end_date)

    if df.empty:
        return df  # Return empty DataFrame if symbol is invalid

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df = df.asfreq('D').fillna(method='ffill')
    return df

def calculate_rsi(data, window=14):
    """Calculate the RSI for a given stock data."""
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def determine_rsi_condition(rsi_value):
    """Determine RSI condition: Overbought, Oversold, or Neutral."""
    if rsi_value >= 70:
        return "Overbought"
    elif rsi_value <= 30:
        return "Oversold"
    else:
        return "Neutral"

def sarimax_model(df):
    """Train a SARIMAX model and generate forecasts for the next 30 days."""
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    exog_train = train[['High', 'Low', 'Volume']]
    exog_test = test[['High', 'Low', 'Volume']]

    sarima_model = SARIMAX(train['Close'],
                           order=(2, 1, 1),  
                           seasonal_order=(1, 1, 1, 12),  
                           exog=exog_train)
    sarima_results = sarima_model.fit()

    forecast_period = 30
    future_forecast = sarima_results.get_forecast(steps=forecast_period, exog=exog_test.tail(forecast_period))
    future_forecast_mean = future_forecast.predicted_mean
    future_forecast_ci = future_forecast.conf_int()

    future_dates = pd.date_range(start=test.index[-1] + pd.Timedelta(days=1), periods=forecast_period, freq='D')

    future_forecast_mean.index = future_dates
    future_forecast_ci.index = future_dates

    combined_data = pd.concat([df[['Close']], future_forecast_mean.rename('Close')])
    combined_data['RSI'] = calculate_rsi(combined_data['Close'])

    future_forecast_values = pd.DataFrame({
        'Date': future_forecast_mean.index,
        'Predicted Value': future_forecast_mean.values,
        'Lower Confidence Interval': future_forecast_ci.iloc[:, 0].values,
        'Upper Confidence Interval': future_forecast_ci.iloc[:, 1].values,
        'RSI': combined_data['RSI'].iloc[-forecast_period:].values
    })

    future_forecast_values['Condition'] = future_forecast_values['RSI'].apply(determine_rsi_condition)

    # Plotting the forecasted values and RSI conditions
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot Predicted Prices as Line Plot
    axs[0].plot(future_forecast_values['Date'], future_forecast_values['Predicted Value'], label='Predicted Price', color='red')
    axs[0].fill_between(future_forecast_values['Date'], future_forecast_values['Lower Confidence Interval'], future_forecast_values['Upper Confidence Interval'], color='gray', alpha=0.2)
    axs[0].set_title('SARIMAX Forecast with Confidence Intervals')
    axs[0].set_xlabel("Date")
    axs[0].set_ylabel("Price")
    axs[0].legend()

    # Plot RSI as Line Plot
    axs[1].plot(future_forecast_values['Date'], future_forecast_values['RSI'], label='RSI', color='blue')
    axs[1].axhline(70, linestyle='--', color='red', label='Overbought (70)')
    axs[1].axhline(30, linestyle='--', color='green', label='Oversold (30)')
    axs[1].fill_between(future_forecast_values['Date'], 0, 30, color='green', alpha=0.2)
    axs[1].fill_between(future_forecast_values['Date'], 70, 100, color='red', alpha=0.2)
    axs[1].set_title(f"RSI for Forecasted Values")
    axs[1].set_xlabel("Date")
    axs[1].set_ylabel("RSI")
    axs[1].legend()

    plt.tight_layout()

    # Convert plot to PNG image in base64 format
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plot_url = base64.b64encode(img_buffer.read()).decode('utf-8')

    # Close the plot to prevent it from being displayed in memory
    plt.close(fig)

    return future_forecast_values.to_html(classes='forecast-table', index=False), plot_url, future_forecast_values['Date'].tolist()

@app.route("/", methods=['GET', 'POST'])
def index():
    forecast_results = []

    if request.method == 'POST':
        symbol = request.form['symbol'].strip().upper()

        if symbol:
            data = extracting_data(symbol)
            if data.empty:
                return render_template('index.html', error=f"No data found for {symbol}.")
            
            forecast_values, candlestick_chart, forecast_dates = sarimax_model(data)

            forecast_results.append({
                'symbol': symbol,
                'forecast_values': forecast_values,
                'chart': candlestick_chart
            })

            return render_template('index.html', forecast_results=forecast_results, symbol=symbol)

    return render_template('index.html', forecast_results=forecast_results)


@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json()
    print("Request received:", req)  # Debugging

    intent_name = req["queryResult"]["intent"]["displayName"]
    stock_symbol = req["queryResult"]["parameters"].get("stock_symbol", "").upper()
    user_dates = req["queryResult"]["parameters"].get("dates", [])  # List of dates provided by the user
    response_text = ""

    if intent_name == "Default Welcome Intent":
        response_text = "Hello! Which company's stock would you like to check?"

    elif intent_name == "Stock Selection Intent":
        if stock_symbol:
            response_text = f"Got it! Fetching stock forecast for {stock_symbol}."
        else:
            response_text = "Please provide a stock symbol."

    elif intent_name == "GetInvestmentAdvice":
        if not stock_symbol:
            response_text = "Please provide a valid stock symbol."
        else:
            data = extracting_data(stock_symbol)
            if data.empty:
                response_text = f"Sorry, no data found for {stock_symbol}."
            else:
                # Generate the 30-day forecast
                forecast_values, _, forecast_dates = sarimax_model(data)

                # Store forecast data for future use
                forecast_data = dict(zip(forecast_dates, forecast_values))

                if user_dates:
                    # Check if user-entered dates are within the forecast range
                    for date_str in user_dates:
                        try:
                            user_date = datetime.strptime(date_str, "%Y-%m-%d")
                            if user_date in forecast_data:
                                forecast_info = forecast_data[user_date]
                                rsi_value = calculate_rsi(data['Close']).iloc[-1]
                                rsi_condition = determine_rsi_condition(rsi_value)

                                response_text += f"\n\nFor {date_str}, the forecasted price is: {forecast_info}. The RSI condition is: {rsi_condition}.\n"

                                # Provide advice based on RSI condition
                                if rsi_condition == "Overbought":
                                    response_text += f"{stock_symbol} is overbought on {date_str}. You might want to wait before investing.\n"
                                elif rsi_condition == "Oversold":
                                    response_text += f"{stock_symbol} is oversold on {date_str}. This could be a buying opportunity.\n"
                                else:
                                    response_text += f"{stock_symbol} is in a neutral RSI zone on {date_str}.\n"
                            else:
                                response_text += f"\n{date_str} is not within the forecasted range of the next 30 days.\n"
                        except ValueError:
                            response_text += f"\n{date_str} is not in a valid date format. Please provide a date in 'yyyy-mm-dd' format.\n"

                # Ask the user if they want to check another stock symbol or exit
                response_text += "\nWould you like to check another stock symbol or exit?"

    elif intent_name == "GetForecastForDates":  # New Intent for entering the forecast
        if not stock_symbol:
            response_text = "Please provide a stock symbol."
        else:
            # Inform the user that the next 30 days forecast is available
            response_text = f"I've got the next 30 days forecast for {stock_symbol}. Please enter the dates in 'yyyy-mm-dd' format that you would like to know the forecast for. You can enter multiple dates, separated by commas."

    return jsonify({"fulfillmentMessages": [{"text": {"text": [response_text]}}]})



if __name__ == "__main__":
    # Run the app with HTTPS enabled
    app.run(debug=True)
