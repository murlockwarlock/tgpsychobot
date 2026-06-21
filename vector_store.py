import chromadb
import asyncio
import os
import sys
import importlib.util
# Workaround for broken torchvision in the environment
_orig_find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, pkg=None: None if name == 'torchvision' else _orig_find_spec(name, pkg)

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

app_port = os.environ.get('APP_PORT', 'default')
if "pytest" in sys.modules or os.environ.get("TESTING") == "true":
    client = chromadb.EphemeralClient()
else:
    client = chromadb.PersistentClient(path=f"/root/telegram_bots/newbots/chroma_db_{app_port}")

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

collection = client.get_or_create_collection(name="knowledge_base")
collection_cases = client.get_or_create_collection(name="case_studies")


async def update_vector_index(document_id: int, text_content: str):
    if not text_content:
        print(f"Предупреждение: для документа ID {document_id} нет контента.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text_content)

    if not chunks:
        print(f"Предупреждение: не удалось создать чанки для документа ID {document_id}.")
        return

    embeddings = await asyncio.to_thread(model.encode, chunks)

    embeddings_list = [e.tolist() for e in embeddings]

    ids = [f"doc{document_id}_chunk{i}" for i, _ in enumerate(chunks)]
    metadatas = [{"document_id": document_id} for _ in chunks]

    await asyncio.to_thread(
        collection.add,
        embeddings=embeddings_list,
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )
    print(f"✅ Документ ID {document_id} успешно проиндексирован. Добавлено чанков: {len(chunks)}")


async def search_relevant_chunks(query: str, n_results: int = 5, document_ids: list[int] | None = None) -> list[str]:
    query_embedding = await asyncio.to_thread(model.encode, query)

    where_clause = {}
    if document_ids is not None:
        if not document_ids:
            return []
        if len(document_ids) == 1:
            where_clause = {"document_id": document_ids[0]}
        else:
            where_clause = {"document_id": {"$in": document_ids}}

    query_params = {
        'query_embeddings': [query_embedding.tolist()],
        'n_results': n_results
    }

    if where_clause:
        query_params['where'] = where_clause

    results = await asyncio.to_thread(
        collection.query,
        **query_params
    )

    if not results or not results.get('documents') or not results['documents'][0]:
        return []

    return results['documents'][0]


def delete_document_vectors(document_id: int):
    try:
        collection.delete(where={"document_id": document_id})
        print(f"✅ Векторы для документа ID {document_id} успешно удалены.")
    except Exception as e:
        print(f"⚠️ Ошибка при удалении векторов документа {document_id}: {e}")


async def update_case_study_index(case_id: int, text_content: str):
    if not text_content:
        return

    embedding = await asyncio.to_thread(model.encode, text_content)

    await asyncio.to_thread(
        collection_cases.add,
        embeddings=[embedding.tolist()],
        documents=[text_content],
        metadatas=[{"case_id": case_id}],
        ids=[f"case_{case_id}"]
    )


async def search_relevant_case(query: str, n_results: int = 1) -> str | None:
    query_embedding = await asyncio.to_thread(model.encode, query)

    results = await asyncio.to_thread(
        collection_cases.query,
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )

    if not results or not results.get('documents') or not results['documents'][0]:
        return None

    return results['documents'][0][0]


def delete_case_study_vectors(case_id: int):
    try:
        collection_cases.delete(where={"case_id": case_id})
        print(f"✅ Векторы для кейса ID {case_id} успешно удалены.")
    except Exception as e:
        print(f"⚠️ Ошибка при удалении векторов кейса {case_id}: {e}")