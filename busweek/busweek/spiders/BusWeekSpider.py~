# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 10:51:09 2016

@author: brian
"""
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

from busweek.items import BusweekItem

class BusWeekSpider(CrawlSpider):
    name = "busweek"
    allowed_domains = ["businessweek.com"]
    start_urls = ["http://www.businessweek.com/archive/news.html"]
    
    rules = (
                Rule(LinkExtractor(allow=('www\.businessweek\.com\/archive\/2014-\d\d\/news\.html')),
                     callback='parse_day_tabs'),)

#    def parse_item(self, response):
#        for href in response.xpath('//ul[@class = "link_list months"]/li/ul/li/a/@href'):
#            url = href.extract()
#            yield scrapy.Request(url, callback=self.parse_month_links)

    def parse_day_tabs(self, response):
        for href in response.xpath('//ul[@class="weeks"]/li/a/@href'):
            url = href.extract()
            yield scrapy.Request(url, callback=self.parse_month_links)
            
    #in the year-month first tab. need to cycle through each tab
    def parse_month_links(self, response):
        for href in response.xpath('//ul[@class = "archive"]/li/h1/a/@href'):
            url = href.extract()
            yield scrapy.Request(url, callback=self.parse_articles)
            
    def parse_articles(self, response):
        item = BusweekItem();
        item['date'] = response.xpath('//meta[@name = "pub_date"]/@content').extract()
        item['body'] = response.xpath('//div[@id = "article_body"]/p/text()').extract()
        item['keywords'] = response.xpath('//meta[@name = "keywords"]/@content').extract()
        item['title'] = response.xpath('//title/text()').extract()
        yield item