from bs4 import BeautifulSoup
import pandas as pd
import requests
import time

def main():
    mises = 'https://mises.org.br'
    my_data = []
    df = pd.DataFrame(columns=['title', 'text', 'author', 'pub_date'])
    for i in range(25):
        # print(i)
        url = url = '/Articles_Thumbs.aspx?page='+str(i)+'&type=3&text='
        current_url = mises+url

        time.sleep(0.01)
        data = requests.get(current_url)

        html = BeautifulSoup(data.text, 'html.parser')
        articles = html.find_all('h4', class_='media-heading mis-subtitle1')

        articles_links = []
        for article in articles:
            link = article.find('a', class_='no-link thumbsArticle').get('href')
            articles_links.append(link)
        
        for link in articles_links:
            article_url = mises+link
            # print(article_url)
            time.sleep(0.01)
            article_page = requests.get(article_url).text
            article_html = BeautifulSoup(article_page, 'html.parser')

            title = article_html.find('div', class_='mis-title1 mis-fg-almostblack').text
            author = article_html.find('a', class_='no-link mis-fg-alpha mis-author-name').text
            pub_date = article_html.find('div', class_='mis-fg-lightgray mis-text mis-article-date').text
            text = article_html.find('div', class_='col-md-22 col-sm-28 col-xs-36 mis-article-center')
            paragraphs = text.find_all('p')

            full_text = ''
            for paragraph in paragraphs:
                full_text += paragraph.text

            my_data.append({'title': title.strip(), 'text': " ".join(full_text.split()), 'author': author.strip(), 'pub_date': pub_date.strip()})

    df = pd.DataFrame.from_records(my_data)
    df.to_csv('mises_brasil_dataset.csv', index=False)
if __name__ == '__main__':
    main()