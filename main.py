from bs4 import BeautifulSoup
from urllib.parse import urljoin
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import get_store, get_embedding
from datetime import datetime, timedelta
from dotenv import load_dotenv
import shelve
import requests
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    temperature=0.1, model_name="gpt-3.5-turbo-16k", openai_api_key=OPENAI_API_KEY
)
chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

store = get_store()
embeddings = get_embedding()


def generative_ai_call(template, data):
    prompt = PromptTemplate.from_template(template)
    llm_chain = prompt | llm
    return llm_chain.invoke(data)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=0, length_function=len, separators=["\n"]
)


def generate_prompt_context(articles):
    lines = [f"{index}. {article['text']}" for index, article in articles.items()]
    return "\n".join(lines)


def split_context(context):
    return text_splitter.create_documents([context])


def get_time_from_string(value):
    return datetime.strptime(value, "%H:%M")


def get_last_upload_time():
    with shelve.open("mycache") as db:
        return db.get("time", "00:00")


def update_last_upload_time():
    with shelve.open("mycache", writeback=True) as db:
        db["time"] = datetime.now().strftime("%H:%M")


def retrieve_ukr_pravda_news():
    last_update_time = get_time_from_string(get_last_upload_time())

    url = "https://www.pravda.com.ua/news/"

    try:
        page = requests.get(url)
        page.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to retrieve the page: {e}")
        exit(1)

    soup = BeautifulSoup(page.text, "html.parser")

    article_headers = soup.find_all(
        "div", class_=["article_news_list article_news_bold", "article_news_list"]
    )

    articles = []
    links = []

    for i, article in enumerate(article_headers):
        article_time = article.find("div", class_="article_time")
        if article_time:
            article_time_value = get_time_from_string(article_time.text.strip())
            if article_time_value < last_update_time:
                break
        a_tag = article.find("a")
        if a_tag:
            articles.append(a_tag.text)
            full_link = urljoin(url, a_tag["href"])
            links.append(full_link)

    metadatas = [{"source": "ukr_pravda", "link": link} for link in links]
    store.add_texts(articles, metadatas)
    print(f"Uploaded {len(articles)} new news from Ukrainska pravda")


def get_bbc_news_time(text_container):
    time_element = text_container.find("time")
    result_time = None
    if time_element:
        time_number = time_element.text.split(" ")[0]
        try:
            time_number = int(time_number)
            print(f"time_number element: {time_number}")
        except ValueError:
            print("Error converting time number to integer")
            return None

        if "хвилин" in time_element.text:
            result_time = datetime.now() - timedelta(minutes=time_number)
        elif "годин" in time_element.text:
            result_time = datetime.now() - timedelta(hours=time_number)
        else:
            print("No time unit found in the text")

    if result_time:
        return result_time.strftime("%H:%M")
    else:
        return None


def retrieve_bbc_ukraine_news():
    url = "https://www.bbc.com/ukrainian"
    last_update_time = get_time_from_string(get_last_upload_time())

    try:
        page = requests.get(url)
        page.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to retrieve the page: {e}")
        exit(1)

    soup = BeautifulSoup(page.text, "html.parser")
    articles = []
    links = []
    promos = soup.select("li[class*=bbc-]")
    promos_list = list(promos)
    news_promos = promos_list[:-4]
    if news_promos:
        for promo in news_promos:
            promo_text_container = promo.find("div", class_="promo-text")
            if promo_text_container:
                news_time = get_bbc_news_time(promo_text_container)
                if not news_time:
                    break
                if get_time_from_string(news_time) < last_update_time:
                    continue
                a = promo_text_container.find("a")
                if a:
                    articles.append(a.text)
                    links.append(a["href"])

    metadatas = [{"source": "bbc_ukraine", "link": link} for link in links]
    store.add_texts(articles, metadatas)
    print(f"Uploaded {len(articles)} new news from BBC Ukraine")


def generative_ai_call(template, data):
    llm_chain = template | llm
    return llm_chain.invoke(data)


def summirize_news_text(text):
    template = PromptTemplate.from_template(
        "Write a concise summary of the following news in the {language} language , with a maximum length of 300 characters. News: {news}"
    )
    response = generative_ai_call(template, {"language": "Ukrainian", "news": text})
    return response.content


def load_full_page(document):

    try:
        page = requests.get(document.metadata["link"])
        page.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to retrieve the page: {e}")
        exit(1)

    soup = BeautifulSoup(page.text, "html.parser")
    news_container = soup.find("div", class_=["block_post"])
    if news_container:
        image = news_container.find("img", class_="post_photo_news_img")
        if image:
            st.image(image["src"])

        news_text = news_container.find("div", class_=["post_text", "post__text"])
        if news_text:
            st.write(summirize_news_text(news_text.text))


def display_news(request):
    query_vector = embeddings.embed_query(request)

    documents = store.similarity_search_by_vector(query_vector, k=5)
    print(f"Documents: \n{documents}")
    news_articles = "".join(doc.page_content + "\n" for doc in documents)
    print(news_articles)
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant, your task is to find news articles that are related to the user's request and return them as a semicolon separated string. The articles should not be changed or reformatted in any way. If no related articles are found, return an empty string. News articles: {news_articles}. Example: news_articles[1], news_articles[3]",
            ),
            ("human", "{request}"),
        ]
    )
    response = generative_ai_call(
        template, {"request": request, "news_articles": news_articles}
    )
    print(f"Response: {response}")
    if response.content:
        article_results = [el.strip() for el in response.content.split(";")]
        print(f"Articles results: {article_results}")
        if article_results:
            for document in documents:
                for article in article_results:
                    if document.page_content.strip() == article.strip():
                        with st.expander(document.page_content):
                            st.write(document.metadata["link"])
                            load_full_page(document)


def main():
    request = st.text_input(label="Write search request")
    if request:
        retrieve_bbc_ukraine_news()
        retrieve_ukr_pravda_news()
        update_last_upload_time()
        display_news(request)


if __name__ == "__main__":
    main()
