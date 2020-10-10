## %
from pytrends.request import TrendReq
from addon import *
import pandas as pd
import time
import pycountry

## %
text_file = open("./data/trends/countries.txt", "r")
lines = text_file.read().split(',')

## %
countries = {}
for country in pycountry.countries:
    countries[country.name] = country.alpha_2

print(len(lines))

pytrends = TrendReq(hl='en-US', tz=360)

## %
kw_list = ["how do i know if i have covid"]
df = pd.DataFrame()


for c in lines:
    try:
        pytrends.build_payload(kw_list, cat=0, timeframe='2020-03-22 2020-10-09', geo=countries[c], gprop='')
        country_df = pytrends.interest_over_time()
        country_df["Country_Name"] = c
        country_df["Country_Code"] = countries[c]
        df = pd.concat([df, country_df])
        df.to_csv("./data/trends/countries_trend_2.csv", index=False)
        time.sleep(.1)
        print(c)
    # print(pytrends.interest_by_region(resolution="COUNTRY"))
    except:
        print("fail", c)
        time.sleep(5)

## %
# df.to_csv("./data/trends/countries_trend.csv", index=False)