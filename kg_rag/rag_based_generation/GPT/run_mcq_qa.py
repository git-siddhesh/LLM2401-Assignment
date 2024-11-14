'''
This script takes the MCQ style questions from the csv file and save the result as another csv file. 
Before running this script, make sure to configure the filepaths in config.yaml file.
Command line argument should be either 'gpt-4' or 'gpt-35-turbo'
'''

from kg_rag.utility import *
import sys


from tqdm import tqdm
CHAT_MODEL_ID = sys.argv[1]  # gemini-1.5-flash

QUESTION_PATH = config_data["MCQ_PATH"]
SYSTEM_PROMPT = system_prompts["MCQ_QUESTION"]
DOMAIN_KNOWLEDGE_PROMPT = system_prompts["DOMAIN_KNOWLEDGE_PROMPT"]
CONTEXT_VOLUME = int(config_data["CONTEXT_VOLUME"])
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = float(config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = float(config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"])
VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
NODE_CONTEXT_PATH = config_data["NODE_CONTEXT_PATH"]
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"]
TEMPERATURE = config_data["LLM_TEMPERATURE"]
SAVE_PATH = config_data["SAVE_RESULTS_PATH"]


CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID

save_name = "_".join(CHAT_MODEL_ID.split("-"))+"_kg_rag_based_mcq_{mode}.csv"


vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
embedding_function_for_context_retrieval = load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)
edge_evidence = False


MODE = "0"
MODE = "1"
MODE = "2"
MODE = "3"
### MODE 0: Original KG_RAG                     ### 
### MODE 1: jsonlize the context from KG search ### 
### MODE 2: Add the prior domain knowledge      ### 
### MODE 3: Combine MODE 1 & 2                  ### 
import re
def jsonize_context(context):
    pattern_variant = r'Variant (.*) associates Disease (.*)'
    pattern_gene = r'Disease (.*) associates Gene (.*)'

    entities = {}
    data = {
        'variants': [],
        'genes': [],
    }
    # Iterate over each key-value pair in the context
    for key, values in context.items():
        # Iterate over each line in the list of values
        for line in values:
            # Check if the line matches the variant pattern
            match_variant = re.search(pattern_variant, line)
            if match_variant:
                variant, disease = match_variant.groups()
                k = entities.get(disease.strip(), {'variants': [], 'genes': []})
                k['variants'].append(variant.strip())
                entities[disease.strip()] = k

            # Check if the line matches the gene pattern
            match_gene = re.search(pattern_gene, line)
            if match_gene:
                disease, gene = match_gene.groups()
                k = entities.get(disease.strip(), {'variants': [], 'genes': []})
                k['genes'].append(gene.strip())
                entities[disease.strip()] = k
    return entities

def jsonize_context_v2(context):
    pattern_variant = r'Variant (.*) associates Disease (.*)'
    pattern_gene = r'Disease (.*) associates Gene (.*)'

    entities = {}
    data = {
        'variants': [],
        'genes': [],
    }
    # Iterate over each key-value pair in the context
    for key, values in context.items():
        entities[key] = {
                        'variants': [],
                        'genes': [],
                    }
        # Iterate over each line in the list of values
        for line in values:
            # Check if the line matches the variant pattern
            match_variant = re.search(pattern_variant, line)
            if match_variant:
                variant, disease = match_variant.groups()
                entities.get(key)['variants'].append(variant.strip())

            # Check if the line matches the gene pattern
            match_gene = re.search(pattern_gene, line)
            if match_gene:
                disease, gene = match_gene.groups()
                entities.get(key)['genes'].append(gene.strip())
    return entities

def main():
    start_time = time.time()
    question_df = pd.read_csv(QUESTION_PATH)
    answer_list = []
    for index, row in tqdm(question_df.iterrows(), total=306):
        try: 
            question = row["text"]
            if MODE == "0":
                ### MODE 0: Original KG_RAG                     ### 
                context_dict = retrieve_context(row["text"], vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID)
                
                enriched_prompt = "Context: "+ ". ".join([". ".join(value) for key, value in context_dict.items()]) + "\n" + "Question: "+ question
                print(enriched_prompt)
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            # if MODE == "1":
            #     ### MODE 1: jsonlize the context from KG search ### 
            #     ### Please implement the first strategy here    ###
            #     output = '...'
            if MODE == "1":
                context_dict = retrieve_context(
                    row["text"], vectorstore, embedding_function_for_context_retrieval,
                    node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                    QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID
                )

                context = jsonize_context_v2(context_dict)
                enriched_prompt = f"Context: {context}\nQuestion: {question}"
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)
                # output = output.replace('{{', '{').replace('}}', '}').strip().replace("```json","").replace("```", "").strip()
                # output = json.loads(output)

            # if MODE == "2":
            #     ### MODE 2: Add the prior domain knowledge      ### 
            #     ### Please implement the second strategy here   ###
            #     output = '...'
            if MODE == "2":
                # Generate domain knowledge from the LLM based on the question topic
                domain_knowledge = get_Gemini_response(row["text"], DOMAIN_KNOWLEDGE_PROMPT, temperature=TEMPERATURE)
                
                # Retrieve the context using the KG-RAG method
                context_dict = retrieve_context(
                    domain_knowledge, vectorstore, embedding_function_for_context_retrieval,
                    node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                    QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID
                )
                
                # Append generated domain knowledge to the retrieved context
                enriched_context = f"{". ".join([". ".join(value) for key, value in context_dict.items()])}\nAdditional Knowledge: {domain_knowledge}"
        
                # Create an enriched prompt for the question
                enriched_prompt = f"Context: {enriched_context}\n NOTE: Above answer may be hallucination. Please provide correct answer from context only.\nQuestion: {row['text']}"
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            # if MODE == "3":
            #     ### MODE 3: Combine MODE 1 & 2                  ### 
            #     ### Please implement the third strategy here    ###
            #     output = '...'
            if MODE == "3":
                # Generate domain knowledge from the LLM based on the question topic
                domain_knowledge = get_Gemini_response(row["text"], DOMAIN_KNOWLEDGE_PROMPT, temperature=TEMPERATURE)
                
                # Retrieve the context using the KG-RAG method
                context_dict = retrieve_context(
                    domain_knowledge, vectorstore, embedding_function_for_context_retrieval,
                    node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                    QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID
                )
                
                # Append generated domain knowledge to the retrieved context
                context = jsonize_context_v2(context_dict)

                enriched_context = f"{context}\nAdditional Knowledge: {domain_knowledge}"
        
                # Create an enriched prompt for the question
                enriched_prompt = f"Context: {enriched_context}\n NOTE: Above answer may be hallucination. Please provide correct answer from context only.\nQuestion: {row['text']}"
                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            answer_list.append((row["text"], row["correct_node"], output))
        except Exception as e:
            print("Error in processing question: ", row["text"])
            print("Error: ", e)
            answer_list.append((row["text"], row["correct_node"], "Error"))


    answer_df = pd.DataFrame(answer_list, columns=["question", "correct_answer", "llm_answer"])
    output_file = os.path.join(SAVE_PATH, f"{save_name}".format(mode=MODE),)
    answer_df.to_csv(output_file, index=False, header=True) 
    print("Save the model outputs in ", output_file)
    print("Completed in {} min".format((time.time()-start_time)/60))

        
        
if __name__ == "__main__":
    main()


