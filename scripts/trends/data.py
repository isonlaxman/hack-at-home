from pytrends.request import TrendReq
from addon import *

pytrends = TrendReq(hl='en-US', tz=360)

kw_list = ["covid symptoms"]
pytrends.build_payload(kw_list, cat=0, timeframe='2020-02-01 2020-10-09', geo='', gprop='')
# print(interest_by_city(pytrends))
print(pytrends.interest_over_time())
print(pytrends.interest_by_region(resolution="COUNTRY"))
