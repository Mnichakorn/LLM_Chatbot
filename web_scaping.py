import time
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
from tqdm import tqdm

# Gender/ Age
def gender_age(soup):
    divs = soup.find_all('div', class_='flex flex-row justify-between')
    p = []
    for div in divs:
        p_tag = div.find('p', class_='text-sm text-gray-500')
        p.append(p_tag)
    p[0].get_text(strip=True).split('|')
    gender = p[0].get_text(strip=True).split('|')[0]
    age = re.findall(r'\d+', p[0].get_text(strip=True).split('|')[1])[0]
    return gender, age

def tag_fuc(soup):
    ul = soup.find('ul', class_='flex flex-row gap-2')
    tag_list = []
    if ul:
        for li in ul.find_all('li'):
            tag_list.append(li.get_text(strip=True))
    return tag_list

def doc_name_func(soup):
    divs = soup.find_all('div', class_='flex flex-col justify-between')
    for div in divs:
        p_tag = div.find_all('p', class_='font-bold')
    return p_tag[0].get_text()

df = pd.DataFrame(data={'Date':[],'Gender':[],'Age':[],'Topic':[],'Tag':[],'Question':[],'Doctor_name':[],'Answer':[]})

main_url = 'https://www.agnoshealth.com/forums'
for i in tqdm(range(1,183)):
    req = requests.get(f'{main_url}/search?page={i}')
    main_soup = BeautifulSoup(req.content, "html.parser")
    for link in main_soup.find_all('a', attrs={'class':'undefined'}):
        sub_url = f'{main_url}{link['href']}'
        print(sub_url)
        req = requests.get(sub_url)
        soup = BeautifulSoup(req.content, "html.parser")
        try: 
            qa = soup.select_one('main div section div span').get_text(strip=True)
        except: 
            qa=''
        try:
            gender, age = gender_age(soup)
        except: 
            gender = ''; age = ''
        try: 
            topic = soup.find('p', class_='font-bold').get_text()
        except:
            topic = ''
        try:
            date = soup.find('time').get_text()
        except:
            date = ''
        try:
            tag = tag_fuc(soup)
        except:
            tag = ''
        try:
            doc_name = doc_name_func(soup)
        except:
            doc_name = ''
        try:
            ans = soup.find('p', class_='mt-4').get_text(strip=True)
        except:
            ans = ''
        df1 = pd.DataFrame(data={'Date':date,'Gender':gender,'Age':age,'Topic':topic,'Tag':[tag],'Question':qa,'Doctor_name':doc_name,'Answer':ans})
        df = pd.concat([df, df1], ignore_index=True)
        df.to_csv('assets/health_forum_raw.csv', index=False)

df = df[df['Answer'].notna()]
df = df.drop(columns = ['Doctor_name', 'Date'])
df['Question'] = df['Question'].str.replace('สวัสดีค่ะ','').str.replace('สวัสดีครับ','').str.replace('สวัสดีฮะ','').str.replace('\n','')
df['Answer'] = df['Answer'].str.replace('สวัสดีค่ะ','').str.replace('สวัสดีครับ','').str.replace('สวัสดีฮะ','').str.replace('\n','')
df.to_csv('health_forum.csv', index=False)