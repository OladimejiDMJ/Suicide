#!/usr/bin/env python
# coding: utf-8

# In[1]:


from selenium import webdriver
driver=webdriver.Chrome(r'C:\Users\OLANREWAJU\Desktop\chromedriver_win32\chromedriver_win32kk\chromedriver')
driver.get('https://www.facebook.com')
username=input("Enter your facebook login username/phone number: ")
password=input("Enter your facebook password: ")
driver.find_element_by_name("email").send_keys(username)
driver.find_element_by_name("pass").send_keys(password)
driver.find_element_by_id("loginbutton").click()


# In[2]:


driver.find_element_by_xpath('//*[@id="navItem_1572366616371383"]/a/div').click()


# In[3]:


driver.find_element_by_xpath('//*[@id="pagelet_bookmark_seeall"]/div/div/div[1]/div/div[1]/div/a[2]').click()


# In[4]:


for i in range(3):
    driver.execute_script('window.scrollTo(0,document.body.scrollHeight)')


# In[13]:


#driver.find_element_by_xpath('//*[@id="pagelet_timeline_app_collection_100000684131931:2356318349:2"]/ul/li[3]/div/div/div[2]/div/div[2]/div/a').click()


# In[37]:


#post=driver.find_element_by_xpath('//*[@id="js_12j"]/p')


# In[38]:


#print(post.text)


# In[5]:


friends=driver.find_elements_by_xpath('//li/div/div/div[2]/div/div[2]/div/a')


# In[15]:


for i in friends:
    friend=i.text
    print(friend)


# In[19]:


f= open("facebook_friends.txt","a+")
f.write(friend)


# In[ ]:




