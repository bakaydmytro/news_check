from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import BigQueryVectorSearch
from langchain_community.vectorstores.utils import DistanceStrategy
from google.cloud import bigquery
from dotenv import load_dotenv

import logging
import sys
import os

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
DATASET = os.getenv("DATASET")
TABLE = os.getenv("TABLE")
REGION = os.getenv("REGION")


class Config:
    def __init__(self):
        self.embedding, self.client = self.setup_embedding_and_client()
        self.store = self.setup_store(self.embedding)

    def setup_embedding_and_client(self):
        embedding = VertexAIEmbeddings(
            model_name="textembedding-gecko@latest", project=PROJECT_ID
        )
        client = bigquery.Client(project=PROJECT_ID, location=REGION)
        client.create_dataset(dataset=DATASET, exists_ok=True)
        return embedding, client

    def setup_store(self, embedding):
        print("Setuping Big Query Store")
        return BigQueryVectorSearch(
            project_id=PROJECT_ID,
            dataset_name=DATASET,
            table_name=TABLE,
            location=REGION,
            embedding=embedding,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        )


config = Config()


def get_store():
    return config.store


def get_embedding():
    return config.embedding
