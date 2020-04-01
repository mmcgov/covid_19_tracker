# -*- coding: utf-8 -*-
import scrapy
from collections import defaultdict
from scrapy.selector import HtmlXPathSelector
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
import pandas as pd
import openpyxl
import sys 
import requests
from lxml import html
from datetime import datetime
import time
import os
from datetime import date
from scrapy.spiders import BaseSpider
from scrapy.http import FormRequest
from scrapy.utils.response import open_in_browser
from twisted.internet import reactor, defer
from scrapy.crawler import CrawlerRunner
from scrapy.crawler import Crawler
from scrapy.utils.log import configure_logging
from scrapy.utils.project import get_project_settings
import ssl
from functools import wraps
import pandas as pd
import os
import requests
import traceback
import numpy as np
from subprocess import call


class CountriesItem(scrapy.Item):
     country = scrapy.Field()
     total_cases = scrapy.Field()
     new_cases = scrapy.Field()
     total_deaths = scrapy.Field()
     new_deaths = scrapy.Field()
     total_recovered = scrapy.Field()
     active_cases = scrapy.Field()
     serious_critical = scrapy.Field()
     total_deaths_per_million_pop = scrapy.Field()


class covid_19_data_collector(scrapy.Spider):

    def __init__(self):
        name = 'coronavirus_tracker'
        allowed_domains = ['www.worldometers.info']
        start_urls = ['https://www.worldometers.info/coronavirus/#countries']
    
    def start_requests(self):
            yield scrapy.Request('https://www.worldometers.info/coronavirus/#countries', callback=self.parse)

    def parse(self, response):
        global raw_res
        raw_res = list()
        row = 0 
        for _ in range(0,len(response.xpath('//*[@id="main_table_countries_today"]/tbody/tr'))-1):  
            row += 1 
            for column in range(1,12): 
                try: 
                    raw_res.append(response.xpath(f'//*[@id="main_table_countries_today"]/tbody[1]/tr[{row}]/td[{column}]//text()').extract_first()) 
                except AttributeError: 
                    raw_res.append(response.xpath(f'//*[@id="main_table_countries_today"]/tbody[1]/tr[{row}]/td[{column}]/text()').extract_first()) 

 
    def closed(self, reason):
        columns = ['country',
                   'total_cases',
                   'new_cases',
                   'total_deaths',
                   'new_deaths',
                   'total_recovered',
                   'active_cases',
                   'serious_critical',
                   'total_cases_per_million_pop',
                   'total_deaths_per_million_pop',
                   'first_case_date']
        today = datetime.now().strftime('%d/%m/%Y')
        vals = zip(raw_res[0::11], raw_res[1::11], raw_res[2::11], raw_res[3::11], raw_res[4::11],
           raw_res[5::11], raw_res[6::11], raw_res[7::11], raw_res[8::11], raw_res[9::11], raw_res[10::11])

        results = pd.DataFrame(vals, columns= columns)
        results['first_case_date'] = results['first_case_date'].apply(lambda x: x.replace('\n','').strip())
        results.insert(0, 'date', today)
        master_results = pd.read_csv('../../../data/covid_19_case_data.csv')
        master_results = master_results.append(results)
        master_results.to_csv('../../../data/covid_19_case_data.csv', index=False)

        data = pd.read_csv('../../../data/covid_19_case_data.csv')
        data['date'] = data['date'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y'))
        data = data.sort_values(['country', 'date']).reset_index(drop=True)
        data['total_cases'] = data['total_cases'].apply(lambda x: int(x.replace(',','')))
        data['new_cases'] = data['total_cases'] - data['total_cases'].shift(1).fillna(0)
        data['new_cases'] = np.where(data['new_cases']<0, 0, data['new_cases'].astype(int))
        data['total_deaths'] = data['total_deaths'].apply(lambda x: x.replace(',','').strip())
        data = data.replace(r'^\s*$', np.nan, regex=True).fillna(0)
        data['total_deaths'] = data['total_deaths'].apply(lambda x: int(x))
        data['new_deaths'] = data['total_deaths'] - data['total_deaths'].shift(1).fillna(0)
        data['new_deaths'] = np.where(data['new_deaths']<0, 0, data['new_deaths'].astype(int))
        data['mask'] = data['total_recovered'].apply(lambda x: isinstance(x, int))
        data['total_recovered'] = np.where(data['mask']==True,
                                   data['total_recovered'].astype(str),
                                   data['total_recovered'])
        data['total_recovered'] = data['total_recovered'].apply(lambda x: x.replace(',','').strip())
        data = data.replace(r'^\s*$', np.nan, regex=True).fillna(0)
        data['total_recovered'] = data['total_recovered'].apply(lambda x: int(x))
        data['active_cases'] = data['active_cases'].apply(lambda x: x.replace(',','').strip())
        data['active_cases'] = data['active_cases'].apply(lambda x: int(x))
        data['mask'] = data['total_recovered'].apply(lambda x: isinstance(x, int))
        data['serious_critical'] = np.where(data['mask']==True,
                                   data['serious_critical'].astype(str),
                                   data['serious_critical'])
        data['serious_critical'] = data['serious_critical'].apply(lambda x: x.replace(',','').strip())
        data['serious_critical'] = data['serious_critical'].apply(lambda x: int(x))
        data['mask'] = data['total_deaths_per_million_pop'].apply(lambda x: isinstance(x, float))
        data['total_cases_per_million_pop'] = np.where(data['mask']==True,
                                   data['total_cases_per_million_pop'].astype(str),
                                   data['total_cases_per_million_pop'])
        data['total_cases_per_million_pop'] = data['total_cases_per_million_pop'].apply(lambda x: x.replace(',','').strip())
        data['total_cases_per_million_pop'] = data['total_cases_per_million_pop'].astype(float)
        data['mask'] = data['total_deaths_per_million_pop'].apply(lambda x: isinstance(x, float))
        data['total_deaths_per_million_pop'] = np.where(data['mask']==True,
                                   data['total_deaths_per_million_pop'].astype(str),
                                   data['total_deaths_per_million_pop'])
        data['total_deaths_per_million_pop'] = data['total_deaths_per_million_pop'].apply(lambda x: x.replace(',','').strip())
        data['total_deaths_per_million_pop'] = data['total_deaths_per_million_pop'].astype(float)        
        data['case_growth_rate'] = (data['new_cases']/data['total_cases'].shift(1))*100
        data['case_growth_rate'] = np.where(data['case_growth_rate']<0,0,data['case_growth_rate'])
        data['death_growth_rate'] = (data['new_deaths']/data['total_deaths'].shift(1))*100
        data['death_growth_rate'] = np.where(data['death_growth_rate']<0,0,data['death_growth_rate'])
        data.fillna(0, inplace=True)
        data = data.drop('mask', axis=1)
        pops_df = pd.read_csv('../../../data/world_pops.csv')
        pops_dict = dict(zip(pops_df['country'],pops_df['pop_mills'].astype(float)))
        data['pop_millions'] = data['country'].map(pops_dict)
        data.fillna(0, inplace=True)
        dates_df = pd.read_csv('../../../data/first_case.csv')
        dates_dict = dict(zip(dates_df['country'],dates_df['date_first_case']))
        data['first_case_date'] = data['country'].map(dates_dict)
        data.to_csv('../../../data/graph_data.csv', index=False)





configure_logging()
runner = CrawlerRunner()


@defer.inlineCallbacks
def crawl():
    yield runner.crawl(covid_19_data_collector)
    reactor.stop()


def main():
   crawl()
   reactor.run() # the script will block here until the last crawl call is finished
  # rc=call('../../../website/website.sh')


if __name__ == '__main__':
    main()
