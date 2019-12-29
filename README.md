# stockmarketLSTM
This project is for MGMT 590- Big Data technologies at Krannert School of Management, developed by Team : Roli Gupta, Zaid Ahmed, Saumya Bharati, Gajender Saharan and Samir Husain. This is a data science model and implementation of a stock market prediction for Walt Disney Co.


Workflow process
Group 2 - Stock market prediction

The video for our project can be found here – 
https://youtu.be/v7VBTBaZpc8
Data pipeline
1.	Download data from Kaggle NYSE - https://www.kaggle.com/dgawlik/nyse
2.	Push data to Hive and create a hive table
  a.	Create a Hive script to splice data for just Disney stock prices
  b.	Push this data to Tableau and download in CSV for modelling
3.	Use this data and append recent stock data extracted using Yahoo finance API
4.	Run the LSTM model (code filename: group2_stockprediction_code.ipynb)and send the predictions for the next 8 days to a google sheet
5.	This google sheet then feeds into the Tableau dashboard
6.	The modelling python script is hosted in a GCP VM instance running Debian/GNU 9
7.	Using crontab, we have scheduled the python script to run every day at 1am
  a.	During every run, the script runs LSTM modelling on training data that includes opening and closing prices from January 1, 1962 to the day before and predicts the open and close prices for the next 7 days.
 
