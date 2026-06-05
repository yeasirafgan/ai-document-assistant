from typing import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.database import get_vector_store


class RAGState(TypedDict):
    question: str
    documents: list[Document]
    answer: str


def retrieve(state: RAGState) -> dict:
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(state["question"], k=settings.retrieval_k)
    return {"documents": docs}


def generate(state: RAGState) -> dict:
    llm = ChatAnthropic(
        model=settings.chat_model,
        api_key=settings.anthropic_api_key,
        max_tokens=1024,
    )

    context = "\n\n".join([
        f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in state["documents"]
    ])

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
At the end of your answer, mention which page(s) you used as sources.
If the context does not contain enough information, say so honestly.

Context:
{context}

Question: {state["question"]}

Answer:"""

    response = llm.invoke(prompt)
    return {"answer": response.content}


def build_rag_graph():
    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()


def ask_question(question: str) -> dict:
    graph = build_rag_graph()
    result = graph.invoke({
        "question": question,
        "documents": [],
        "answer": "",
    })

    seen = set()
    sources = []
    for doc in result["documents"]:
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "?")
        key = (page, source)
        if key not in seen:
            seen.add(key)
            sources.append({"page": page, "source": source})

    return {"answer": result["answer"], "sources": sources}


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is RAG?"
    print(f"Question: {question}")
    result = ask_question(question)
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources:")
    for s in result["sources"]:
        print(f"  Page {s['page']} — {s['source']}")
