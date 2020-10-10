## %
from pytrends.request import TrendReq
from addon import *
import pandas as pd
import time
import pycountry

## %
text_file = open("./data/trends/countries.txt", "r")
lines = text_file.read().split(',')

print(len(lines))

pytrends = TrendReq(hl='en-US', tz=360)

## %
kw_list = ["covid symptoms"]
df = pd.DataFrame()

pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-22 2020-10-09', geo="US", gprop='')
country_df = pytrends.interest_over_time()
country_df["Country_Name"] = "US"
country_df["Country_Code"] = "US"
df = pd.concat([df, country_df])
df.to_csv("./data/trends/countries_trend_USA.csv", index=False)