# -*- coding: utf-8 -*-
import scrapy
import pandas as pd
from datetime import datetime
from twisted.internet import reactor, defer
from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging
from subprocess import call
import numpy as np
import os


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
        yield scrapy.Request('https://www.worldometers.info/coronavirus/#countries',
                             callback=self.parse)

    def parse(self, response):
        global raw_res
        raw_res = list()
        self.c_table = '//*[@id="main_table_countries_today"]/tbody/tr'
        self.row = 8
        for _ in range(14, len(response.xpath(self.c_table)) - 1):
            self.row += 1
            for self.column in range(2, 16):
                self.row_data = f'//*[@id="main_table_countries_today"]/tbody[1]\
                                  /tr[{self.row}]/td[{self.column}]//text()'
                self.row_data_alt = f'//*[@id="main_table_countries_today"]/tbody[1]\
                                     /tr[{self.row}]/td[{self.column}]/text()'
                try:
                    raw_res.append(
                        response.xpath(f'//*[@id="main_table_countries_today"]\
                            /tbody[1]/tr[{self.row}]/td[{self.column}]//text()'
                                       ).extract_first())
                except AttributeError:
                    raw_res.append(
                        response.xpath(f'//*[@id="main_table_countries_today"]\
                            /tbody[1]/tr[{self.row}]/td[{self.column}]/text()'
                                       ).extract_first())

    def closed(self, reason):
        # appending new daily data to master data
        self.case_data = '../../../data/covid_19_case_data.csv'
        self.graph_data = '../../../data/graph_data.csv'
        self.first_case_data = '../../../data/first_case.csv'
        self.world_pop_data = '../../../data/world_pops.csv'
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
                   'total_tests_per_million_pop',
                   'population']
        today = datetime.now().strftime('%d/%m/%Y')
        vals = zip(raw_res[0::14], raw_res[1::14], raw_res[2::14],
                   raw_res[3::14], raw_res[4::14], raw_res[5::14],
                   raw_res[6::14], raw_res[8::14], raw_res[9::14], 
                   raw_res[10::14], raw_res[11::14], raw_res[12::14], 
                   raw_res[13::14])
        results = pd.DataFrame(vals, columns=columns)
        results.insert(0, 'date', today)
    
        if os.path.isfile(self.case_data):
            master_results = pd.read_csv(self.case_data, 
                                         index_col=False)
        else:
            master_results = pd.DataFrame()
        master_results = master_results.append(results)
        master_results.to_csv(self.case_data, index=False)

        # Post formatting for graph data
        cols = list(master_results.columns[2:])
        master_results[cols] = master_results[cols].astype(str)
        master_results['date'] = master_results['date'].apply(
                                 lambda x: datetime.strptime(x, '%d/%m/%Y'))
        master_results = master_results.sort_values(
                         ['country', 'date']).reset_index(drop=True)
        master_results[cols] = master_results[cols].replace(
                               {r'^\s*$': 0, 'None': 0, 'nan': 0,
                                ',': '', 'N/A': 0}, regex=True)
        master_results[cols] = master_results[cols].astype(float)
        master_results['case_growth_rate'] = (master_results['new_cases']
                                              / master_results['total_cases']
                                              .shift(1)) * 100
        master_results['case_growth_rate'] = np.where(
                                    master_results['case_growth_rate'] < 0, 0,
                                    master_results['case_growth_rate'])
        master_results['death_growth_rate'] = (master_results['new_deaths']
                                               / master_results['total_deaths']
                                               .shift(1)) * 100
        master_results['death_growth_rate'] = np.where(
                                    master_results['death_growth_rate'] < 0, 0,
                                    master_results['death_growth_rate'])
        pops_df = pd.read_csv(self.world_pop_data)
        pops_dict = dict(zip(pops_df['country'],
                             pops_df['pop_mills'].astype(float)))
        master_results['pop_millions'] = master_results['country'].map(
                                         pops_dict)
        dates_df = pd.read_csv(self.first_case_data)
        dates_dict = dict(zip(dates_df['country'],
                              dates_df['date_first_case']))
        master_results['first_case_date'] = master_results['country'].map(
                                            dates_dict)
        master_results.fillna(0, inplace=True)
        for col in cols:
            master_results[col] = master_results[col].astype(int)
        master_results.to_csv((self.graph_data), index=False)


configure_logging()
runner = CrawlerRunner()


@defer.inlineCallbacks
def crawl():
    yield runner.crawl(covid_19_data_collector)
    reactor.stop()


def main():
    crawl()
    reactor.run()
    # This will refresh dash site. Comment out if not using site
    # rc = call('../../../website/website.sh')


if __name__ == '__main__':
    main()
