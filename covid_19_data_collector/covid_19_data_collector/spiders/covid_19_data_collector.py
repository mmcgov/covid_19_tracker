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
        row = 7 
        for _ in range(14,len(response.xpath('//*[@id="main_table_countries_today"]/tbody/tr'))-1):  
            row += 1 
            for column in range(1,13): 
                try: 
                    raw_res.append(response.xpath(f'//*[@id="main_table_countries_today"]/tbody[1]/tr[{row}]/td[{column}]//text()').extract_first()) 
                except AttributeError: 
                    raw_res.append(response.xpath(f'//*[@id="main_table_countries_today"]/tbody[1]/tr[{row}]/td[{column}]/text()').extract_first()) 
         
 
    def closed(self, reason):
        #appending new daily data to master data
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
                   'total_tests',
                   'total_tests_per_million_pop']
        today = datetime.now().strftime('%d/%m/%Y')
        vals = zip(raw_res[0::12], raw_res[1::12], raw_res[2::12], raw_res[3::12], raw_res[4::12],
           raw_res[5::12], raw_res[6::12], raw_res[7::12], raw_res[8::12], raw_res[9::12], raw_res[10::12], raw_res[11::12])

        results = pd.DataFrame(vals, columns= columns)
        results.insert(0, 'date', today)
        master_results = pd.read_csv('../../../data/covid_19_case_data.csv')
        master_results = master_results.append(results)
        master_results.to_csv('../../../data/covid_19_case_data.csv', index=False)

        # Post formatting for graph data
        cols = list(master_results.columns[2:])
        int_cols = ['total_cases', 'new_cases', 'total_deaths', 
                    'new_deaths', 'total_recovered', 'active_cases', 'serious_critical', 'total_tests']
        master_results[cols] = master_results[cols].astype(str)
        master_results['date'] = master_results['date'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y'))
        master_results = master_results.sort_values(['country', 'date']).reset_index(drop=True)
        master_results[cols] = master_results[cols].replace({r'^\s*$':0, 'None':0, 'nan':0, ',':'', 'N/A':0}, regex=True)
        master_results[cols] = master_results[cols].astype(float)
        master_results['case_growth_rate'] = (master_results['new_cases']\
                                              /master_results['total_cases'].shift(1))*100
        master_results['case_growth_rate'] = np.where(master_results['case_growth_rate']<0,0,
                                                      master_results['case_growth_rate'])
        master_results['death_growth_rate'] = (master_results['new_deaths']\
                                               /master_results['total_deaths'].shift(1))*100
        master_results['death_growth_rate'] = np.where(master_results['death_growth_rate']<0,0,
                                                       master_results['death_growth_rate'])
        pops_df = pd.read_csv('../../../data/world_pops.csv')
        pops_dict = dict(zip(pops_df['country'],pops_df['pop_mills'].astype(float)))
        master_results['pop_millions'] = master_results['country'].map(pops_dict)
        dates_df = pd.read_csv('../data/first_case.csv')
        dates_dict = dict(zip(dates_df['country'],dates_df['date_first_case']))
        master_results['first_case_date'] = master_results['country'].map(dates_dict)
        master_results.fillna(0, inplace=True)
        for col in int_cols:
            master_results[col] = master_results[col].astype(int)
        master_results.to_csv('../../../data/graph_data.csv', index=False)

configure_logging()
runner = CrawlerRunner()


@defer.inlineCallbacks
def crawl():
    yield runner.crawl(covid_19_data_collector)
    reactor.stop()


def main():
   crawl()
   reactor.run() # the script will block here until the last crawl call is finished
   rc=call('../website/website.sh')


if __name__ == '__main__':
    main()
