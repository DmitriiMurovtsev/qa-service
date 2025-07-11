import logging
from typing import Dict, List
from fastapi import FastAPI, HTTPException, Request
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
from uuid import uuid4

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# логируем в stdout
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Настройка
MODEL_NAME = 'cointegrated/LaBSE-en-ru'
COLLECTION_NAME = 'qa_collection'
VECTOR_SIZE = 768  # зависит от модели

# Инициализация
app = FastAPI()
model = SentenceTransformer(MODEL_NAME)
client = QdrantClient(host="qdrant", port=6333)

# Создаём коллекцию при запуске, если не существует
if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )

@app.post("/add")
async def add(request: Request) -> Dict[str, str]:
    """
    Добавляет вопрос-ответ в коллекцию

    Parameters:
        question: str - вопрос
        answer: str - ответ

    Returns:
        Dict[str, str] - статус и сообщение

    Raises:
        Exception - ошибка при добавлении
    """
    data = await request.json()
    question = data.get("question")
    answer = data.get("answer")
    try:
        combined = f"Вопрос: {question} Ответ: {answer}"
        embedding = model.encode(combined)

        point = PointStruct(
            id=str(uuid4()),
            vector=embedding,
            payload={"question": question, "answer": answer}
        )

        client.upsert(collection_name=COLLECTION_NAME, points=[point])
        return {"status": "ok", "message": "Question-answer added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(request: Request) -> List[Dict[str, str]]:
    """ 
    Ищет похожие вопросы в коллекции.
    
    Parameters:
        query: str - поисковый запрос
        top: int = 3 - количество результатов

    Returns:
        List[Dict[str, str]] - список найденных вопросов и ответов

    Raises:
        Exception - ошибка при поиске
    """
    data = await request.json()
    query = data.get("query")
    top = data.get("top")
    try:
        embedding = model.encode(query)
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding,
            limit=top,
            score_threshold=0.31,
        )
        logger.info(f'Вопрос: {query}\nРезультат поиска: {results}')
        return [r.payload for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/all")
async def all() -> List[Dict[str, str]]:
    """
    Возвращает все вопросы из коллекции.

    Returns:
        List[Dict[str, str]] - список всех вопросов и ответов

    Raises:
        Exception - ошибка при получении всех вопросов
    """
    try:
        return [r.payload for r in client.scroll(collection_name=COLLECTION_NAME)[0]]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete")
async def delete(request: Request) -> bool:
    """
    Удаляет вопрос-ответ из коллекции.
    Parameters:
        question: str - вопрос
        answer: str - ответ
    Returns:
        bool - успешно ли удалено
    Raises:
        Exception - ошибка при удалении
    """
    data = await request.json()
    question = data.get("question")
    answer = data.get("answer")
    try:
        result = client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="question",
                        match=MatchValue(value=question)
                    ),
                    FieldCondition(
                        key="answer",
                        match=MatchValue(value=answer)
                    ),
                ]
            )
        )
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    