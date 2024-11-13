
import requests
import json
import os
import weaviate
import weaviate.classes as wvc
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter

from unstructured.chunking.title import chunk_by_title
from unstructured.partition.md import partition_md
from unstructured.partition.auto import partition


wcd_url = os.environ["WEAVIATE_CLUSTER_URL"]
wcd_api_key = os.environ["WEAVIATE_READER_API_KEY"]
wcd_admin_api_key = os.environ["WEAVIATE_ADMIN_API_KEY"]
openai_api_key = os.environ["OPENAI_API_KEY"]
toto = os.environ["TOTO"]

my_coll_name='Automne_2024'


def create(client,coll_name):
    collection = client.collections.create(
        name=coll_name,
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),  # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
        generative_config=wvc.config.Configure.Generative.openai()  # Ensure the `generative-openai` module is used for generative queries
    )
    return collection


def load_test(client,coll_name):
    resp = requests.get('https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json')
    data = json.loads(resp.text)  # Load data

    question_objs = list()
    for i, d in enumerate(data):
        question_objs.append({
            "answer": d["Answer"],
            "question": d["Question"],
            "category": d["Category"],
        })

    collection = client.collections.get(coll_name)
    collection.data.insert_many(question_objs)


def load_custom(client,coll_name):
    pass

#query_type can be: fetch_objects, bm25, near_text
def respond(client,coll_name,query_type,query_param):
    collection = client.collections.get(coll_name)
    response = eval("collection.query."+query_type+"(query='"+query_param+"',limit=5)")
    #response = collection.query.near_text(query=query_param,limit=5)
    for i in range(len(response.objects)):
        print(response.objects[i].properties)


def delete(client,coll_name,key,value):
    collection = client.collections.get(coll_name)
    collection.data.delete_many(
                    where=Filter.by_property(key).like(value))


def read(client,coll_name,vector_flag):
    collection = client.collections.get(coll_name)
    for item in collection.iterator(include_vector=vector_flag):
        print(item.uuid, item.properties)
        if vector_flag: print(item.vector)



with weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,  # Replace with your Weaviate Cloud URL
    auth_credentials=Auth.api_key(wcd_admin_api_key),  # Replace with your Weaviate Cloud key
    headers={
        'X-OpenAI-Api-key': openai_api_key  # Replace with appropriate header key/value pair for the required API
    }
) as client:  # Use this context manager to ensure the connection is closed
    print(client.is_ready())


    #load(client,my_coll_name)
    #read(client,my_coll_name,False)

    #delete(client,my_coll_name,"question","*")
    #create(client,my_coll_name)
    #read(client,my_coll_name,False)
    #client.collections.delete(my_coll_name)
    respond(client,my_coll_name,"near_text",'biology')


'''
documents = SimpleDirectoryReader("/home/sandra/Documents").load_data()
index = VectorStoreIndex.from_documents(documents)
print(index)
'''
#raise ImportError("`llama-index-readers-file` package not found")
#pip install llama-index

'''




index="Automne-2024"

with WeaviateClientContext(
           EnvVar[str]("WEAVIATE_API_KEY", cast_fct=str).value ,
           EnvVar[str]("WEAVIATE_CLUSTER_URL", cast_fct=str).value ,
           EnvVar[str]("OPENAI_API_KEY", cast_fct=str).value 
        ) as weaviate_client:

            mycol=weaviate_client.collections.get(index)
            #mycol.data.insert_many("")

#vector_store_index=WeaviateVectorStoreIndexContext()


        nodes = file_to_nodes_unstructured(
            file,
            insert_config.file_metadata_keys,
            insert_config.chunk_metadata_keys,
            insert_config.excluded_embed_metadata_keys,
            insert_config.excluded_llm_metadata_keys,
            insert_config.max_characters,
        )

        with WeaviateVectorStoreIndexContext(
            index=index_name,
            vector_database_config=self.weaviate_database_config,
            embedding_model=self.embedding_model,
            openai_api_key=self.openai_api_key,
        ) as vector_store_index:
            vector_store_index.insert_nodes(nodes)


'''