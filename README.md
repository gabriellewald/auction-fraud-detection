<img src='images/robots-greeting.png' width="700">

# Auction Fraud Detection
## Is it Human or Robot?
### Predicting fraud through classification machine learning models and feature engineering.


**Problem Statement:**

On an auction website, human bidders are becoming increasingly frustrated with their inability to win auctions vs. their software-controlled counterparts. As a result, usage from the site's core customer base is plummeting. In order to rebuild customer happiness, the site owners need to eliminate computer generated bidding from their auctions.

**Objective:**

The goal is to identify online auction bids that are placed by "robots", helping the site owners easily flag these users for removal from their site to prevent unfair auction activity.

## 1. Data Acquisition
[Data Wrangling Report](https://github.com/gabriellewald/auction-fraud-detection/blob/main/notebooks/1_data_wrangling.ipynb)

**Data Source:**
This project idea is part of an Engineering competition created by Facebook and Kaggle in 2015. The provided dataset has been said to be one of the richest data of its kind and a world class machine learning problem with great potential for feature engineering. The data was retrieved from the Kaggle website in csv format.

There are two datasets. The bidder dataset includes a list of bidder information, including their unique id, payment account, address, and outcome. The bid dataset includes 7.6 million bids on different auctions - Bids are all made by mobile devices. And, the online auction platform has a fixed increment of dollar amount for each bid, so it doesn't include an amount for each bid.

Challenges included: obfuscated variables to protect privacy and variables that represent unique identifiers. There is a time variable and categorical variables that represent merchandise, country and cellphone devices.

The datasets were merged on bidder_id to make it easier to access information. Although the dataset was fairly clean, there were 29 missing data points for several variables that were mapped to human bids. Since the interest is in identifying robot bids, the rows corresponding to these missing values were dropped. All columns were kept. Regarding the obfuscated fields, they are an issue for interpretability, but are also going to be useful when performing EDA and feature engineering. The columns were rearranged for convinience and the new DataFrame was saved to csv.

## 2. Exploratory Data Analysis
[Exploratory Data Analysis Report](https://github.com/gabriellewald/auction-fraud-detection/blob/main/notebooks/2_exploratory_data_analysis.ipynb)

**Initial Hypothesis:**

1. Total number of bids: Robots might have significantly higher number of bids compared to humans.
2. Number of bids per auction: Robots might have higher number of bids each auction.
3. Distinct IPs: Robots might bid from more diverse IP addresses.
4. Merchandise: Robots might bid more often on certain merchandises.

<p align="center">
<img src='images/robot-human-proportion.png' width="400"> 
<img src='images/proportion-fraud-legitimate.png' width="360">
</p>

The data is highly unbalanced from the bidders and bids perspective. Humans represent 94.88% of the data, while only 5.12% are robots. Legitimate bids represent 86.52% of the data, while 13.43% are fraudulent. There are 12,740 auctions represented in this dataset, and over 3 million bids in the train dataset. Bids come from 5,729 device models in 199 countries, 663,873 URLs and 1,030,950 IP addresses. There are 10 distinct merchandise categories.

