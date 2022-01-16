import logging
from haystack.utils import convert_files_to_dicts, fetch_archive_from_http, print_answers
from haystack.nodes import FARMReader, PreProcessor, ElasticsearchRetriever
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.pipelines import ExtractiveQAPipeline


def tutorial1_basic_qa_pipeline():
    logger = logging.getLogger(__name__)

# Initialise doc store
    #document_store = ElasticsearchDocumentStore()
    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", port=9200, index="document")


    #Define cleaning function
    CleaningFunction = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True
    )
    # Convert files to dicts
    docs = convert_files_to_dicts("DataDocx", split_paragraphs=True)
    
    # Process dicts
    
    # Write dicts to DB
    document_store.write_documents(docs)

    # Initialise retriever 
    retriever = ElasticsearchRetriever(document_store=document_store)

    # Load a  local model or any of the QA models on
    # Hugging Face's model hub (https://huggingface.co/models)

    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

    pipeline = ExtractiveQAPipeline(reader, retriever)
        
    ## Voilà! Ask a question!
    prediction = pipeline.run(query="When is the deadline?")
    
    print("\n\nSimplified output:\n")
    print_answers(prediction, details="minimum")

if __name__ == "__main__":
    tutorial1_basic_qa_pipeline()