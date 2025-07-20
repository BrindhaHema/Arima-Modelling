
import yfinance as yf  # Importing the yfinance library, used to fetch historical market data from Yahoo Finance.
import pandas as pd  # Importing the pandas library, used for data manipulation and analysis.
import numpy as np  # Importing the numpy library, used for numerical computing.
import matplotlib.pyplot as plt  # Importing the matplotlib library, used for data visualization.
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # Importing plot_acf and plot_pacf functions
from statsmodels.tsa.stattools import adfuller, kpss  # Importing functions for stationarity tests.
from statsmodels.stats.stattools import durbin_watson  # Importing the Durbin-Watson test for autocorrelation.
from statsmodels.tsa.arima.model import ARIMA  # Importing the ARIMA model for time series analysis.
from pmdarima import auto_arima  # Importing the auto_arima function for automatic ARIMA model selection.
from datetime import datetime, timedelta  # Importing datetime-related modules for handling dates.
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error  # Importing scalers for feature scaling in machine learning.

## (If any package is not installed install using - "pip install pmdarima")

# Step 1: Extract historical price data from Yahoo Finance
def get_stock_data(ticker, start_date, end_date):
    """
    Fetches historical stock data from Yahoo Finance.

    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date for data retrieval.
        end_date (str): End date for data retrieval.

    Returns:
        pandas.DataFrame: Adjusted close prices.
    """
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)  # Fetching historical stock data.
        return stock_data['Close']  # Returning adjusted close prices.
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

ticker = "IDBI.NS"  # Stock ticker symbol.
start_date = "2023-01-01"  # Start date for data retrieval.
end_date = "2025-05-16"  # End date for data retrieval.
stock_prices = get_stock_data(ticker, start_date, end_date)  # Calling function to get stock data.
#print(stock_prices)

if stock_prices is None:
    print("Exiting due to data download error.")
    exit()

# Step 2: Compute the returns
# stock_returns = stock_prices.pct_change().dropna()    ## if you want to compute just the price difference
stock_returns = np.log(stock_prices / stock_prices.shift(1)).dropna()  # Calculating log returns.

# Step 3: Test for presence of autocorrelation
def test_autocorrelation(data):
    """
    Performs Durbin-Watson test for autocorrelation.

    Args:
        data (pandas.Series): Time series data.

    Returns:
        float: Durbin-Watson test statistic.
    """
    dw_test = durbin_watson(data)  # Performing the Durbin-Watson test for autocorrelation.
    return dw_test  # Returning the Durbin-Watson test statistic.

# Apply differencing to induce autocorrelation
differenced_data = stock_returns.diff().dropna()  # Differencing the data to induce autocorrelation.
dw_test_differenced = test_autocorrelation(differenced_data)  # Performing Durbin-Watson test on differenced data.
print("Durbin-Watson Test Statistic (after differencing):", dw_test_differenced)  # Printing test statistic.

# Step 4: Generate ACF and PACF Plots on Price Data
def plot_acf_pacf_price(data):
    """
    Generates and displays ACF and PACF plots.

    Args:
        data (pandas.Series): Time series data.
    """
    # ACF and PACF plots
    fig, ax = plt.subplots(2, figsize=(12, 8))  # Creating subplots for ACF and PACF.
    plot_acf(data, ax=ax[0], lags=20)  # Generating ACF plot for price data.
    plot_pacf(data, ax=ax[1], lags=20)  # Generating PACF plot for price data.
    ax[0].set_title('ACF - Price Data')  # Setting title for ACF plot.
    ax[1].set_title('PACF - Price Data')  # Setting title for ACF plot.
    plt.show()  # Displaying the plots.

# Generate ACF and PACF plots on the price data
plot_acf_pacf_price(stock_prices)  # Calling function to generate ACF and PACF plots.

def plot_acf_pacf_price_s(data, title):
    # ACF and PACF plots
    fig, ax = plt.subplots(3, figsize=(12, 8))  # Creating subplots for ACF and PACF.
    ax[0].plot(data); ax[0].set_title(title)
    plot_acf(data, ax=ax[1], lags=20)  # Generating ACF plot for price data.
    plot_pacf(data, ax=ax[2], lags=20)  # Generating PACF plot for price data.
    ax[1].set_title('ACF - Price Data')  # Setting title for ACF plot.
    ax[2].set_title('PACF - Price Data')  # Setting title for PACF plot.
    plt.show()  # Displaying the plots.

# Generate ACF and PACF plots on the price data
plot_acf_pacf_price_s(stock_prices,'Original Series' )  # Calling function to generate ACF and PACF plots.
plot_acf_pacf_price_s(stock_prices.diff().dropna(),'1st Order Differencing')
plot_acf_pacf_price_s(stock_prices.diff().diff().dropna(),'2nd Order Differencing')

# Step 5: Test for stationarity
def test_stationarity(data):
    """
    Performs Augmented Dickey-Fuller (ADF) and KPSS tests for stationarity.  Also plots the time series.

    Args:
        data (pandas.Series): Time series data.
    """
    plt.figure(figsize=(12, 6))  # Setting the figure size for the plot.
    plt.plot(data)  # Plotting the data.
    plt.title('Stock Returns')  # Setting title for the plot.
    plt.xlabel('Date')  # Setting label for the x-axis.
    plt.ylabel('Returns')  # Setting label for the y-axis.
    plt.show()  # Displaying the plot.
    
    adf_test = adfuller(data)  # Performing the Augmented Dickey-Fuller test for stationarity.
    print("ADF Test Statistic:", adf_test[0])  # Printing ADF test statistic.
    print("p-value:", adf_test[1])  # Printing p-value for ADF test.
    
    kpss_test = kpss(data)  # Performing the KPSS test for stationarity.
    print("KPSS Test Statistic:", kpss_test[0])  # Printing KPSS test statistic.
    print("p-value:", kpss_test[1])  # Printing p-value for KPSS test.
    
# Test for stationarity on the differenced data
test_stationarity(differenced_data)  # Calling function to test stationarity.

# Step 6: Split data into training and testing sets
test_size = 0.2  # Define the proportion of the data for the test set
X = differenced_data
train_size = int(len(X) * (1 - test_size))
train, test = X[:train_size], X[train_size:]
print(f"Training set length: {len(train)}, Testing set length: {len(test)}")

# Step 7: Determine the best ARIMA model order
try:
    auto_arima_model = auto_arima(train, seasonal=False,
                                  trace=True, error_action='ignore', suppress_warnings=True,
                                  information_criterion='aic')  # Performing automatic ARIMA model selection.
    print("Best ARIMA Model Order:", auto_arima_model.order)  # Printing the best ARIMA model order.
except Exception as e:
    print(f"Error in auto_arima: {e}")
    print("Exiting due to error in model selection.")
    exit()

# Step 8: Estimate the parameters and print the ARIMA model summary
try:
    arima_model = ARIMA(train, order=auto_arima_model.order)  # Creating ARIMA model object.
    arima_fit = arima_model.fit()  # Fitting the ARIMA model to the data.
    print(arima_fit.summary())  # Printing the summary of the fitted ARIMA model.
except Exception as e:
    print(f"Error in ARIMA fitting: {e}")
    print("Exiting due to error in model fitting.")
    exit()




# Step 9: Make predictions on the test set
try:
    predictions = arima_fit.predict(start=len(train), end=len(X) - 1)  # Generate predictions for the test set
    print("Predictions:", predictions)
    #print("length of predictions" , len(predictions))
except Exception as e:
    print(f"Error during prediction: {e}")
    predictions = None # set predictions to None so we can handle the error in the next step
    

days_to_forecast = 30
try:
    forecast = arima_fit.forecast(steps=days_to_forecast)
    print("forecast", forecast)
except Exception as e:
    print(f"Error during forecast: {e}")
    forecast = None

# Step 10: Make Forecast of Prices for 30 days 

latest_price = stock_prices[ticker].iloc[-1] # index of last training price.


cum_ret_forecast = forecast.cumsum()
print(f"Cumulative returns (length: {len(cum_ret_forecast)}):") # Debug

priceforecast = latest_price * np.exp(cum_ret_forecast)
    
    
#print("price_forecast", priceforecast)
    
# Create a date range for the forecast period
forecast_dates = pd.date_range(start=stock_prices.index[-1], periods=days_to_forecast)
temp_df = pd.DataFrame({'Forecast': priceforecast, 'Date': forecast_dates})

    # Reset the index so that we can use the index as a column
temp_df = temp_df.reset_index()

    # Create a new DataFrame
price_forecast = pd.DataFrame({'Date': forecast_dates, 'Forecast': priceforecast})
print("price_forecast", price_forecast)



# Step 11: Compute MSE, RMSE, MAPE
def calculate_mean_error(y_true, y_pred):
    """
    Calculates the Mean Error (ME).

    Args:
        y_true (array-like): The true values.
        y_pred (array-like): The predicted values.

    Returns:
        float: The Mean Error.
    """
    return np.mean(y_pred - y_true)

def calculate_mean_percentage_error(y_true, y_pred):
    """
    Calculates the Mean Percentage Error (MPE).

    Args:
        y_true (array-like): The true values.
        y_pred (array-like): The predicted values.

    Returns:
        float: The Mean Percentage Error.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) / y_true) * 100
if predictions is not None and test is not None and len(predictions) == len(test): # added this check
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, predictions)
    mpe = calculate_mean_percentage_error(test, predictions)
    mape = mean_absolute_percentage_error(test, predictions)  # Corrected this line

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MPE: {mpe:.4f}")
    print(f"MAPE: {mape:.4f}")
else:
    print("Skipping error metrics calculation due to missing or misaligned predictions.")
    mse = None
    rmse = None
    mape = None
    mae = None
    mpe = None
'''
# Print the key values
print(f"\nLatest Price: {latest_price}")
if forecast is not None:
    print("\nForecasted Returns for the next 30 days:")
    print(forecast)
else:
    print("\nForecasted Returns for the next 30 days: Not Available")
if price_forecast is not None:
    print("\nForecasted Prices for the next 30 days:")
    print(price_forecast)
else:
    print("\nForecasted Prices for the next 30 days: Not Available")
if mse is not None and rmse is not None and mape is not None and mae is not None and mpe is not None:
    print(f"\nMSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MPE: {mpe:.4f}, MAPE: {mape:.4f}")
else:
    print("\nError Metrics: Not Available")

# Plotting the results
if predictions is not None and len(predictions) > 0: # added this check
    plt.figure(figsize=(12, 6))
    plt.plot(stock_prices.index[:len(train)], train, label='Historical Prices')
    print ("train_size",train_size,len(predictions))
    plt.plot(stock_prices.index[-(len(predictions)):], predictions, color='r', label='Predicted Prices')
    if price_forecast is not None:
        # Create a date range for the forecast period, starting from the end of the test set
        
        plt.xlim(pd.to_datetime('2023-01-01'))
        plt.plot(price_forecast['Forecast'], color='g', label='30-Day Forecast')
    plt.title('Stock Price Prediction with ARIMA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
else:
    print("Skipping plotting due to missing predictions.")
'''