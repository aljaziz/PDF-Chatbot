from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


class State(TypedDict):
    question: str
    chat_history: List[tuple]
    documents: List
    answer: str


def create_langgraph_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatGroq(model="gemma2-9b-it")

    def retrieve(state: State):
        docs = retriever.get_relevant_documents(state["question"])
        return {**state, "documents": docs}

    def generate(state: State):
        context = "\n\n".join(doc.page_content for doc in state["documents"])
        chat_history_msgs = [
            SystemMessage(
                content="You are an academic assistant who answers questions about research papers."
            ),
            SystemMessage(content=f"Here is the context from the paper:\n{context}"),
        ]
        for q, a in state["chat_history"]:
            chat_history_msgs.append(HumanMessage(content=q))
            chat_history_msgs.append(AIMessage(content=a))
        chat_history_msgs.append(HumanMessage(content=state["question"]))

        answer = llm.invoke(chat_history_msgs)
        return {**state, "answer": answer.content}

    def update_chat_history(state: State):
        updated_history = state["chat_history"] + [(state["question"], state["answer"])]
        return {**state, "chat_history": updated_history}

    graph = StateGraph(State)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("finalize", update_chat_history)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
