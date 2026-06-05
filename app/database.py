from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from app.config import settings

embeddings = OpenAIEmbeddings(
    model=settings.embedding_model,
    api_key=settings.openai_api_key,
)

COLLECTION_NAME = "documents"


def get_vector_store() -> PGVector:
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=settings.database_url,
        use_jsonb=True,
    )
