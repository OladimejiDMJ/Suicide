#!/usr/bin/env python
# coding: utf-8

from selenium import webdriver
driver=webdriver.Chrome(r'C:\Users\OLANREWAJU\Desktop\chromedriver_win32\chromedriver_win32kk\chromedriver')
driver.get('https://www.facebook.com')
username=input("Enter your facebook login username/phone number: ")
password=input("Enter your facebook password: ")
driver.find_element_by_name("email").send_keys(username)
driver.find_element_by_name("pass").send_keys(password)
driver.find_element_by_id("loginbutton").click()

driver.find_element_by_xpath('//*[@id="navItem_1572366616371383"]/a/div').click()


driver.find_element_by_xpath('//*[@id="pagelet_bookmark_seeall"]/div/div/div[1]/div/div[1]/div/a[2]').click()


for i in range(3):
    driver.execute_script('window.scrollTo(0,document.body.scrollHeight)')

friends=driver.find_elements_by_xpath('//li/div/div/div[2]/div/div[2]/div/a')
for i in friends:
    friend=i.text
    print(friend)
