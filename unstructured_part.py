import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.md import partition_md
from unstructured.partition.auto import partition
from unstructured.partition.text import partition_text

import weaviate
from weaviate.classes.init import Auth

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.weaviate import WeaviateVectorStore

from llama_index.embeddings.cohere.base import CohereEmbedding


wcd_url = os.environ["WEAVIATE_CLUSTER_URL"]
wcd_api_key = os.environ["WEAVIATE_READER_API_KEY"]
wcd_admin_api_key = os.environ["WEAVIATE_ADMIN_API_KEY"]
openai_api_key = os.environ["OPENAI_API_KEY"]
cohere_api_key=os.environ["COHERE_API_KEY"]


#my_coll_name='Automne_2024'
my_coll_name='HPFO'


#we need to chose the parameters

file_metadata={"Title": "none","Pages_number":"none"} #donnes par lutilisateur
file_metadata_keys=file_metadata.keys()
max_characters=2048
chunk_metadata_keys=[] #donnes ou?
excluded_embed_metadata_keys=['Title', 'Pages_number'] # a exclure de lembedding
excluded_llm_metadata_keys=['Title', 'Pages_number'] # a exclure du llm

def file_to_nodes_unstructured(filename, file_metadata_keys, chunk_metadata_keys, excluded_embed_metadata_keys,
            excluded_llm_metadata_keys, max_characters):

    #partitions = partition_md(text=file.content)
    partitions = partition_text(filename=filename)
    print(partitions[0])
    #file can be a txt file (str), a byte file, or a partition file (liste)

    #on doit sassurer quil ny a pas doverlap entre ces 2 listes:
    #assert_no_file_and_chunk_metadata_overlap(file_metadata_keys, chunk_metadata_keys)
    #on doit sassurer que ces 2 variables match (expected and provided):
    #assert_file_metadata_consistency(file_metadata_keys, file.metadata.keys())


    chunks = chunk_by_title(
    partitions,
    max_characters=max_characters,
    multipage_sections=True,  # respect page boundaries if False
    combine_text_under_n_chars=max_characters,
    )
    #print(chunks[0].metadata.orig_elements)

    nodes = []
    for chunk in chunks:
        node = TextNode(text=chunk.text)
        node.metadata.update(file_metadata)

        for key in chunk_metadata_keys:
            metadata_value = chunk.metadata.fields.get(key, None)
            node.metadata[key] = str(metadata_value)

        node.excluded_embed_metadata_keys = excluded_embed_metadata_keys
        node.excluded_llm_metadata_keys = excluded_llm_metadata_keys
        nodes.append(node)

    return nodes

#filename="/home/sandra/data/test.txt"
filename="/home/sandra/LLM.txt"
nodes=file_to_nodes_unstructured(filename, file_metadata_keys, chunk_metadata_keys, excluded_embed_metadata_keys,
            excluded_llm_metadata_keys, max_characters)
print("------------")
print("node0 trial 1",nodes[0].metadata,nodes[0].text)
print("lenchunk", len(nodes))




embed_model = CohereEmbedding(
    api_key=cohere_api_key,
    model_name="embed-english-v3.0",
    input_type="search_query",
)


from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.readers.file import FlatReader

blogs = SimpleDirectoryReader('/home/sandra/data').load_data()
# chunk up the blog posts into nodes
parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20)
#nodes = parser.get_nodes_from_documents(blogs)
print("------------")
print("node0 trial 2:",nodes[0].metadata,nodes[0].text )
print("------------")


def write(coll_name):

    if client.collections.exists(coll_name):
        client.collections.delete(coll_name)

    weaviate_vector_store=WeaviateVectorStore(weaviate_client=client, index_name=coll_name)

    index=VectorStoreIndex.from_vector_store(vector_store=weaviate_vector_store)#,embed_model=embed_model)
    index.insert_nodes(nodes)
    #storage_context = StorageContext.from_defaults(vector_store = vector_store)

    # set up the index
    #index = VectorStoreIndex(nodes, storage_context = storage_context)

    #DEBUG:
    #Here is the set of all errors: \'pages number\' is not a valid property name. 
    #Property names in Weaviate are restricted to valid GraphQL names, which must be “/[_A-Za-z][_0-9A-Za-z]{0,230}/”.
    collection = client.collections.get(coll_name)
    print(client.batch.failed_objects)
    print(collection.batch.failed_objects)



def read(coll_name):
    collection = client.collections.get(coll_name)
    for item in collection.iterator(include_vector=True):
        print(item.uuid, item.properties)



with weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,  # Replace with your Weaviate Cloud URL
    auth_credentials=Auth.api_key(wcd_admin_api_key),  # Replace with your Weaviate Cloud key
    headers={'X-OpenAI-Api-key': openai_api_key}  # Replace with appropriate header key/value pair for the required API
    ) as client:

    if my_coll_name=="HPFO":
        print("------------Read db")
        print(client.is_ready())
        print("------------")

        read(my_coll_name)

    elif my_coll_name=="Automne_2024":

        print("------------Insert node")
        print(client.is_ready())
        print("------------")

        #write(my_coll_name)
        
        print("------------Read db")
        print(client.is_ready())
        print("------------")

        read(my_coll_name)







