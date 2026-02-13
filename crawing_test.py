import requests
from bs4 import BeautifulSoup
 
headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36'}
data = requests.get('스크래핑할 주소',headers=headers)
 
soup = BeautifulSoup(data.text, 'html.parser')
 
# 이 이후로 필요한 부분 추출