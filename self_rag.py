from langchain_databricks.vectorstores import DatabricksVectorSearch
from langchain_databricks import ChatDatabricks
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from typing_extensions import TypedDict
from typing import List

from config import *

### Retriver

# Construct the RetrievalQA chain for Vector Store search
def get_retriever(persist_dir=None, k=4):
    # vsc = VectorSearchClient(disable_notice=True)
    # vs_index = vsc.get_index(vs_endpoint_name, vs_index_table_fullname)
    vectorstore = DatabricksVectorSearch(
        endpoint=VS_ENDPOINT_NAME,
        index_name=VS_INDEX_TABLE_FULLNAME,
        columns=["pmid"],
    )
    
    return vectorstore.as_retriever(
        search_type='similarity_score_threshold',
        search_kwargs={"k": k, 'query_type': 'HYBRID', "fetch_k": 2, "lambda_mult": 0.5, "score_threshold": 0.5},
    )

retriever = get_retriever(k=5)

### Self-RAG

MODEL = 'databricks-meta-llama-3-1-70b-instruct'

llm = ChatDatabricks(endpoint=MODEL, max_tokens = 1000)

score_tools = [
    {
        "type": "function",
        "function": {
            "name": "_score",
            "description": "Gives a binary 'yes' or 'no' grading score of the input text",
            "parameters": {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "string",
                        "enum": ["yes", "no"],
                    },
                },
                "required": ["score"],
            },
        },
    },
]

score_parser = JsonOutputKeyToolsParser(key_name='_score', first_tool_only=True)

llm_with_tools = llm.bind_tools(score_tools, tool_choice="required")

### Retrieval Grader

retrieval_grader_prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. 
    If the document contains keywords related to the user question, grade it as relevant. 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
    Here is the retrieved document: 
    \n\n {document} \n\n
    Here is the user question: {question}
    """,
    input_variables=["question", "document"],
)

retrieval_grader = retrieval_grader_prompt | llm_with_tools| score_parser

### Generate

generator_prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise\n
    Question: {question}
    Context: {context}
    Answer: 
    """,
    input_variables=["question", "document"],
)

# Chain
generator = generator_prompt | llm | StrOutputParser()

### Hallucination Grader

# Prompt
hallucination_prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. 
    Give a binary 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. 
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    Here are the facts:
    \n ------- \n
    {documents}
    \n ------- \n
    Here is the answer: {generation}  
    """,
    input_variables=["generation", "documents"],
)

hallucination_grader = hallucination_prompt | llm_with_tools | score_parser

## Answer Grader

answer_grader_prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is useful to resolve a question. 
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. 
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    Here is the answer:
    \n ------- \n
    {generation}
    \n ------- \n
    Here is the question: {question} 
    """,
    input_variables=["generation", "question"],
)

answer_grader = answer_grader_prompt | llm_with_tools | score_parser

### Question Re-writer

question_rewriter_prompt = PromptTemplate(
    template="""You a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval. 
    Look at the input and try to reason about the underlying semantic intent / meaning.
    Here is the initial question: {question}
    Formulate an improved question. Only question, no additional information.
    """,
    input_variables=["question"],
)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
question_rewriter = question_rewriter_prompt | llm | StrOutputParser()

### Create Graph

## State

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """
    question: str
    generation: str
    documents: List[str]

from langchain.schema import Document

## Nodes

def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = generator.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    
    return {"documents": filtered_docs, "question": question}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


## Conditional edge

def decide_to_generate(state):
    """
    Determines whether to generate an answer, e-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    
    # Build graph
from langgraph.graph import START, END, StateGraph

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query

# Add edges
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile
self_rag_graph = workflow.compile()
