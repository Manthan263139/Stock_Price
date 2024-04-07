import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import seaborn as sns
from textblob import TextBlob 





import yfinance as yf
import pandas as pd




with open('style.css') as f:
     st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)

st.title('Stock Trend Prediction')
user_input = st.text_input("Enter the stock Ticker", 'AAPL')

# Allow the user to select starting and ending dates
start_date = st.date_input("Select the starting date", pd.to_datetime('2010-01-01'))
end_date = st.date_input("Select the ending date", pd.to_datetime('2019-12-31'))

# Convert selected dates to strings in the required format
start = start_date.strftime('%Y-%m-%d')
end = end_date.strftime('%Y-%m-%d')

df = yf.download(user_input, start=start, end=end)

st.subheader(f'Data from {start} to {end}')
st.write(df.describe())
# In this modified code, we use st.date_input to create two date selection widgets for the starting and ending dates. The selected dates are then converted to the desired format ('%Y-%m-%d') using strftime and used as the start and end parameters when downloading the stock data with yf.download. This allows users to specify a custom date range for the stock data.






















#visualizations
st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize =(12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize =(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize =(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)


 #graph of original price vs predicted price


data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)
#x_train=[]
#y_train=[]
#for i in range(100,data_training_array.shape[0]):
    #x_train.append(data_training_array[i-100:i])
    #y_train.append(data_training_array[i,0])
#x_train, y_train=np.array(x_train),np.array(y_train)   


# Load my model
model=load_model('keras_model.h5')

# Testing  Part
past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_testing,ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])


x_test,y_test=np.array(x_test),np.array(y_test)
y_predicted=model.predict(x_test)
scaler=scaler.scale_
scale_factor=1/scaler[0]
y_predicted=y_predicted * scale_factor
y_test=y_test * scale_factor


# Final Graph
st.subheader('Predictions vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)




# news_headlines = [
#     "Stocks surge as company reports strong earnings",
#     "Investors worried about economic downturn",
#     "New product launch boosts stock price",
#     "Stock market experiences a sharp decline",
#     # Add more headlines as needed
# ]

# # Analyze the sentiment of each headline using TextBlob
# sentiments = [TextBlob(headline).sentiment.polarity for headline in news_headlines]

# # Plot original price, predicted price, and sentiment on the same graph
# st.subheader('Original Price, Predicted Price, and Sentiment')
# fig3, ax1 = plt.subplots(figsize=(12, 6))

# # Plot the original price and predicted price
# ax1.plot(y_test, 'b', label='Original Price')
# ax1.plot(y_predicted, 'r', label='Predicted Price')
# ax1.set_xlabel('Time')
# ax1.set_ylabel('Price')
# ax1.legend(loc='upper left')

# # Create a second y-axis for sentiment
# ax2 = ax1.twinx()
# ax2.plot(sentiments, 'g', label='Sentiment')
# ax2.set_ylabel('Sentiment')
# ax2.legend(loc='upper right')

# st.pyplot(fig3)
# # In this modified code, we analyze the sentiment of each news headline using TextBlob and then plot the original price, predicted price, and sentiment on the same graph. You'll need to adapt the code to fetch and analyze real-world textual data related to the stock you're interested in.




# sentiment_scores = [-0.2, 0.5, -0.8, 0.2, 0.7]  # Example sentiment scores

# # Map sentiment scores to emotions
# emotions = []
# for score in sentiment_scores:
#     if score >= 0.7:
#         emotions.append("Happy")
#     elif score >= 0.2:
#         emotions.append("Positive")
#     elif score >= -0.2:
#         emotions.append("Neutral")
#     elif score >= -0.7:
#         emotions.append("Negative")
#     else:
#         emotions.append("Sad")

# # Plot original price, predicted price, and emotions
# st.subheader('Original Price, Predicted Price, and Emotions')
# fig5, ax1 = plt.subplots(figsize=(18, 9))

# # Plot the original price and predicted price
# ax1.plot(y_test, 'b', label='Original Price')
# ax1.plot(y_predicted, 'r', label='Predicted Price')
# ax1.set_xlabel('Time')
# ax1.set_ylabel('Price')
# ax1.legend(loc='upper left')

# # Create a second y-axis for emotions
# ax2 = ax1.twinx()
# ax2.set_yticks([])  # Remove y-axis ticks for emotions
# for i, emotion in enumerate(emotions):
#     ax2.annotate(emotion, (i, 0.5), fontsize=12, ha='center', va='center', color='black')

# st.pyplot(fig5)
# # In this version of the code, we use text labels ("Happy," "Positive," "Neutral," "Negative," and "Sad") instead of emojis, which should be more widely supported. This should work on most systems without font or emoji rendering issues.




# sentiment_scores = [-0.2, 0.5, -0.8, 0.2, 0.7]  # Example sentiment scores

# # Map sentiment scores to emotions
# emotions = []
# for score in sentiment_scores:
#     if score >= 0.7:
#         emotions.append("Happy")
#     elif score >= 0.2:
#         emotions.append("Positive")
#     elif score >= -0.2:
#         emotions.append("Neutral")
#     elif score >= -0.7:
#         emotions.append("Negative")
#     else:
#         emotions.append("Sad")



# Calculate Mean Absolute Error (MAE)
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_predicted)

# Display MAE
st.subheader(f'Mean Absolute Error (MAE): {mae:.2f}')

# # Plot original price, predicted price, and emotions
# st.subheader('Original Price, Predicted Price, and Emotions')
# fig5, ax1 = plt.subplots(figsize=(18, 9))

# # Plot the original price and predicted price
# ax1.plot(y_test, 'b', label='Original Price')
# ax1.plot(y_predicted, 'r', label='Predicted Price')
# ax1.set_xlabel('Time')
# ax1.set_ylabel('Price')
# ax1.legend(loc='upper left')

# # Create a second y-axis for emotions
# ax2 = ax1.twinx()
# ax2.set_yticks([])  # Remove y-axis ticks for emotions
# for i, emotion in enumerate(emotions):
#     ax2.annotate(emotion, (i, 0.5), fontsize=12, ha='center', va='center', color='black')

# st.pyplot(fig5)
# Plot original price, predicted price, and emotions

