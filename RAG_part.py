
##########   RAG

import os
from llama_index.embeddings.cohere.base import CohereEmbedding
import weaviate
from weaviate.classes.init import Auth
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import VectorStoreIndex

wcd_url = os.environ["WEAVIATE_CLUSTER_URL"]
wcd_api_key = os.environ["WEAVIATE_READER_API_KEY"]
wcd_admin_api_key = os.environ["WEAVIATE_ADMIN_API_KEY"]
openai_api_key = os.environ["OPENAI_API_KEY"]
cohere_api_key=os.environ["COHERE_API_KEY"]



from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import (
    get_response_synthesizer,  # pyright: ignore[reportUnknownVariableType]
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai.base import (  # pyright: ignore[reportMissingTypeStubs]
    OpenAI,
)
from llama_index.postprocessor.cohere_rerank.base import (  # pyright: ignore[reportMissingTypeStubs]
    CohereRerank,
)
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.vector_stores.types import VectorStoreQueryMode



from llama_index.embeddings.openai.base import (  # pyright: ignore[reportMissingTypeStubs]
    OpenAIEmbedding,
)

message="Qui est quentin ?"
my_coll_name="HPFO"
#my_coll_name='Automne_2024'


llm_model = OpenAI(
    model="gpt-4o",
    #model = "gpt-4o-mini-2024-07-18",

    api_key=openai_api_key,
    temperature=0.0,
)



reranker_model = CohereRerank(
    api_key=cohere_api_key, 
    model="rerank-multilingual-v3.0", 
    top_n=4
)

embedding_model = CohereEmbedding(
    cohere_api_key=cohere_api_key,
    model_name="embed-multilingual-v3.0",
    input_type="search_query",
)


#embedding_model=OpenAIEmbedding(api_key=openai_api_key, model="text-embedding-3-small", dimensions=1536)


hybrid_search=True
query_mode = VectorStoreQueryMode.DEFAULT
if hybrid_search:
    query_mode = VectorStoreQueryMode.HYBRID

similarity_cutoff=None


def return_retriever(index):
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
        filters=None,
        vector_store_query_mode=query_mode,
        alpha=0.9
    )
    return retriever

def return_node_postprocessors():

    node_postprocessors= []

    if similarity_cutoff:
        node_postprocessors.append(
            SimilarityPostprocessor(
                similarity_cutoff=similarity_cutoff
            )
        )

    if reranker_model is not None:
        node_postprocessors.append(reranker_model)

    return node_postprocessors


def return_response_synthesizer():

    def create_default_rag_prompt() -> PromptTemplate:
        return PromptTemplate(
            (
                "Voici des sources d'information classees dans l'ordre de leur pertinence avec pour chacune un nom de document et du texte pouvant contenir des elements de reponses.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Tu es un assistant expert de la documentation interne de l'entreprise."
                "En te servant uniquement des sources d'information fournies precedemment, reponds en Francais de maniere concise.\n"
                "Si tu n es pas certain de ta reponse, dis simplement que tu ne sais pas.\n"
                "A la fin de ta reponse, cite la ou les sources ayant permis de repondre à la question en respectant le format suivant si possible : (source : [file_name], [page_number], [url]).\n\n"
                "Question: {query_str}\n"
                "Reponse: "
            ),
            prompt_type=PromptType.QUESTION_ANSWER,
        )


    def create_default_rag_as_tool_prompt() -> PromptTemplate:
        return PromptTemplate(
            (
                """
            ### Context ###\n
            You are retrieving information from the company’s knowledge base to assist in answering a specific query.\n
            ### Information ###\n
            {context_str}\n
            ### Instructions ###\n
            1. Only return chunks of information relevant to answer the query.\n"
            2. If no relevant information is found, respond: 'No relevant information found to answer this query.'\n"
            3. Do not generate a response, just return all the relevant chunks with their corresponding metadata.\n"
            ### Query ###\n
            {query_str}\n
            ### Relevant Information with metadata ###\n
            """
            ),
            prompt_type=PromptType.QUESTION_ANSWER,
        )

    response_synthesizer = get_response_synthesizer(
        llm=llm_model,
        text_qa_template=create_default_rag_prompt(),
        response_mode=ResponseMode.COMPACT,
        use_async=False,
        streaming=True,
        structured_answer_filtering=False,
        verbose=False,
    )
    return response_synthesizer


import time

start = time.time()


with weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,  # Replace with your Weaviate Cloud URL
    auth_credentials=Auth.api_key(wcd_admin_api_key),  # Replace with your Weaviate Cloud key
    headers={'X-OpenAI-Api-key': openai_api_key}  # Replace with appropriate header key/value pair for the required API
    ) as client:
    print("------------RAG")
    print(client.is_ready())
    print("------------")


    collections = client.collections.list_all()
    print(collections)

    weaviate_vector_store=WeaviateVectorStore(weaviate_client=client, index_name=my_coll_name)
    index=VectorStoreIndex.from_vector_store(vector_store=weaviate_vector_store,embed_model=embedding_model)

    retriever=return_retriever(index)
    node_postprocessors=return_node_postprocessors()
    response_synthesizer=return_response_synthesizer()

    query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=node_postprocessors,
    response_synthesizer=response_synthesizer,
    )

    response = query_engine.query(message)
    response.print_response_stream()
#sys:1: ResourceWarning: unclosed <ssl.SSLSocket fd=14, family=2, type=1, proto=6, laddr=('192.168.1.105', 42524), raddr=('34.96.76.122', 443)>


print("time",time.time() - start)
