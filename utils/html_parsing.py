from bs4 import BeautifulSoup
from urllib.request import urlopen

url = "https://kpburgula.github.io/"
html = urlopen(url)
soupified = BeautifulSoup(html, 'html.parser')

# print(soupified.prettify())
print('Title',soupified.title)
question = soupified.find("div", {"class": "col-md-4"})
question = question.find("div", {"class": "skills-box"})
question = question.find("p")
print(question)