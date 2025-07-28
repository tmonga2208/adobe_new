import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import time # For measuring processing time

# --- Configuration ---
class AppConfig:
    def __init__(self):
        self.chromadb_persist_path: str = "../vbl_chromaa" # Path to your ChromaDB data
        self.collection_name: str = "vbl-local"         # Name of your collection
        self.embedding_model_name: str = "all-MiniLM-L6-v2" # This model is ~90MB, well within 1GB
        self.top_k_results: int = 5 # Number of top relevant chunks to retrieve for context
        # Placeholder for a local LLM path/model (if you set one up)
        # For actual local LLM, you'd use libraries like llama-cpp-python, transformers with specific models
        self.local_llm_model_path: Optional[str] = None # e.g., "path/to/your/quantized_phi2.gguf"

# --- Persona and Job Definition ---
class Persona:
    def __init__(self, role: str, expertise: str, focus_areas: List[str]):
        self.role = role
        self.expertise = expertise
        self.focus_areas = focus_areas

    def get_description(self) -> str:
        return f"Role: {self.role}. Expertise: {self.expertise}. Focus Areas: {', '.join(self.focus_areas)}."

class JobToBeDone:
    def __init__(self, task_description: str):
        self.task_description = task_description

    def get_description(self) -> str:
        return f"Task: {self.task_description}"


# === Core Retrieval System ===
class DocumentAnalystSystem:
    def __init__(self, config: AppConfig):
        self.config = config
        self.embedding_fn = SentenceTransformerEmbeddingFunction(model_name=config.embedding_model_name)
        self.client = chromadb.PersistentClient(path=config.chromadb_persist_path)
        try:
            self.collection = self.client.get_collection(config.collection_name, embedding_function=self.embedding_fn)
            print(f"Loaded ChromaDB collection: '{config.collection_name}' with {self.collection.count()} items.")
        except Exception as e:
            print(f"ERROR: Could not load ChromaDB collection '{config.collection_name}'. Make sure you've run the ingestion pipeline. Error: {e}")
            exit()

        # Initialize local LLM if path is provided (conceptual/placeholder)
        self.llm_model = None
        if self.config.local_llm_model_path:
            try:
                # This is a placeholder. Actual implementation depends on your chosen local LLM framework.
                # Example with transformers:
                # from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
                # self.tokenizer = AutoTokenizer.from_pretrained(self.config.local_llm_model_path)
                # self.llm_model = AutoModelForCausalLM.from_pretrained(self.config.local_llm_model_path)
                # self.llm_pipeline = pipeline("text-generation", model=self.llm_model, tokenizer=self.tokenizer)
                print(f"Attempting to load local LLM from: {self.config.local_llm_model_path} (Placeholder)")
                # For this demo, just a boolean to indicate "LLM available"
                self.llm_model = True
            except Exception as e:
                print(f"WARNING: Could not load local LLM from {self.config.local_llm_model_path}. Falling back to simple summary. Error: {e}")
                self.llm_model = None
        else:
            print("No local LLM path provided. Will use basic summarization.")


    def _craft_enhanced_query(self, user_query: str, persona: Persona, job: JobToBeDone) -> str:
        """
        Crafts an enhanced query by incorporating persona and job-to-be-done details.
        This helps the embedding model find more relevant documents.
        """
        prompt = (
            f"Given the persona: '{persona.get_description()}' "
            f"and the job-to-be-done: '{job.get_description()}', "
            f"find relevant information for the following user question: '{user_query}'."
            f"Consider topics related to {', '.join(persona.focus_areas)}."
        )
        return prompt

    def retrieve_relevant_sections(self, user_query: str, persona: Persona, job: JobToBeDone) -> Tuple[List[str], List[float]]:
        """
        Retrieves the most relevant document sections based on the enhanced query.
        """
        enhanced_query = self._craft_enhanced_query(user_query, persona, job)
        print(f"\nCrafted Enhanced Query for Retrieval:\n'{enhanced_query}'\n")

        # ChromaDB query will use the embedding function defined during collection creation
        results = self.collection.query(
            query_texts=[enhanced_query],
            n_results=self.config.top_k_results
        )

        docs = results["documents"][0]
        # Distances are typically L2 distance for SentenceTransformers in Chroma, lower is better.
        # Ensure 'inf' for scores if no results, so rejection logic works.
        scores = results.get("distances", [[float('inf')]*self.config.top_k_results])[0]

        print(f"\n🔍 Retrieved Top {len(docs)} Relevant Sections:")
        for i, (doc, score) in enumerate(zip(docs, scores)):
            print(f"\n--- Section {i+1} (Score: {score:.4f}) ---")
            print(f"{doc[:500]}...\n") # Print first 500 chars for preview

        return docs, scores

    def analyze_and_synthesize_answer(self, user_query: str, retrieved_docs: List[str], persona: Persona, job: JobToBeDone) -> str:
        """
        Synthesizes an answer tailored to the persona and job-to-be-done.
        Attempts to use a local LLM if available, otherwise provides a basic summary.
        """
        print("\n🤖 Analyzing and Synthesizing Answer:")
        print("="*80)

        if not retrieved_docs:
            return "No relevant information found to synthesize an answer."

        # Construct a prompt for the LLM
        llm_prompt = (
            f"You are an intelligent document analyst. Your role is '{persona.role}', with expertise in '{persona.expertise}' "
            f"and focus areas in {', '.join(persona.focus_areas)}. Your current task is: '{job.task_description}'.\n\n"
            f"Based on the following retrieved relevant document sections, and considering your persona and task, "
            f"provide a comprehensive and prioritized answer to the user's question. If the information is not directly available "
            f"in the provided sections, state that you cannot answer based on the given context. Avoid hallucinations.\n\n"
            f"User Question: {user_query}\n\n"
            f"Retrieved Document Sections:\n"
        )
        for i, doc in enumerate(retrieved_docs):
            llm_prompt += f"--- Section {i+1} ---\n{doc}\n\n"

        llm_prompt += "Prioritized Answer (from the perspective of the persona, fulfilling the job-to-be-done):"

        synthesized_answer = ""
        if self.llm_model:
            print("\n--- Sending prompt to Local LLM ---")
            try:
                # This is where your actual local LLM call would go
                # Example with a simple text manipulation as a stand-in for a tiny LLM
                # For a real LLM, this would be model.generate(llm_prompt) or similar
                # A very basic "LLM" that just tries to answer based on the first few words of the query
                # and concatenates relevant docs. Not truly intelligent, but shows structure.
                simulated_llm_response = f"As a {persona.role} focused on {', '.join(persona.focus_areas)}, and with the task to {job.task_description}, here is the analysis for '{user_query}':\n\n"
                simulated_llm_response += "Based on the provided sections:\n"
                for i, doc in enumerate(retrieved_docs):
                    simulated_llm_response += f"Section {i+1}: {doc[:200]}...\n" # Summarize each doc
                synthesized_answer = simulated_llm_response + "\n\n(Note: This is a simplified answer. Integrate a small local LLM like Phi-2 for true synthesis.)"

                # If using a real local LLM (e.g., with llama-cpp-python):
                # from llama_cpp import Llama
                # llm = Llama(model_path=self.config.local_llm_model_path)
                # output = llm(llm_prompt, max_tokens=256, stop=["Q:", "\n\n"], echo=False)
                # synthesized_answer = output["choices"][0]["text"]

            except Exception as e:
                synthesized_answer = f"Error calling local LLM: {e}. Falling back to basic summary."
                print(f"Error details: {e}")
        else:
            # Fallback if no LLM is integrated or failed to load
            synthesized_answer = (
                f"No LLM available for advanced synthesis. Here's the most relevant section "
                f"from the perspective of a {persona.role} tasked with '{job.task_description}':\n\n"
                f"{retrieved_docs[0]}"
            )

        print("="*80)
        return synthesized_answer

# === Main Execution ===
if __name__ == "__main__":
    start_total_time = time.time() # Start timer for overall execution

    app_config = AppConfig()
    # You might set a local LLM path here if you have one
    # app_config.local_llm_model_path = "path/to/your/phi-2-gguf-model.bin"

    analyst_system = DocumentAnalystSystem(app_config)

    # --- Sample Test Case 1: Academic Research ---
    print("\n--- Running Sample Test Case 1: Academic Research ---")
    phd_researcher_persona = Persona(
        role="PhD Researcher in Computational Biology",
        expertise="Graph Neural Networks, Drug Discovery, Bioinformatics",
        focus_areas=["methodologies", "datasets", "performance benchmarks", "novel architectures", "biological applications"]
    )
    literature_review_job = JobToBeDone(
        task_description="Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks."
    )
    user_question_1 = "What are the key advancements in graph neural networks for drug discovery, specifically regarding their methodologies and typical datasets used?"

    print(f"\nPersona: {phd_researcher_persona.get_description()}")
    print(f"Job: {literature_review_job.get_description()}")
    print(f"User Question: {user_question_1}")

    start_retrieval_time = time.time()
    retrieved_sections_1, scores_1 = analyst_system.retrieve_relevant_sections(
        user_question_1,
        phd_researcher_persona,
        literature_review_job
    )
    end_retrieval_time = time.time()
    print(f"Retrieval time: {end_retrieval_time - start_retrieval_time:.2f} seconds.")


    # Decide if we have enough confidence to answer based on retrieval scores
    # For L2 distance (SentenceTransformer default), a lower score means higher similarity.
    # You'll need to fine-tune this threshold based on your data and model.
    # A score of > 1.0 might mean very little similarity, but it depends on embedding space.
    # A more robust check might involve analyzing the distribution of scores or setting a fixed max distance.
    confidence_threshold = 0.8 # Example threshold: adjust based on experimentation (lower is better for L2)

    if not retrieved_sections_1 or scores_1[0] > confidence_threshold:
        print("❌ Sorry, this question is outside the knowledge of the documents or confidence is too low.")
    else:
        start_synthesis_time = time.time()
        final_answer_1 = analyst_system.analyze_and_synthesize_answer(
            user_question_1,
            retrieved_sections_1,
            phd_researcher_persona,
            literature_review_job
        )
        end_synthesis_time = time.time()
        print(f"Synthesis time: {end_synthesis_time - start_synthesis_time:.2f} seconds.")
        print(f"\nFinal Analyst Answer (for Test Case 1):\n{final_answer_1}")

    end_total_time = time.time()
    print(f"\nTotal processing time for query: {end_total_time - start_total_time:.2f} seconds.")
    print("\n" + "="*100 + "\n")

    # --- Add more test cases here if needed ---
    # Remember to create new Persona and JobToBeDone objects for each scenario.