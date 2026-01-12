# modules/generator.py


from typing import List, Dict, Tuple
import re
from streamlit.delta_generator import DeltaGenerator 

from modules.llm_interface import get_llm, GenParams
from modules.retriever import retrieve
from modules.embedder import Embedder
from modules.vectorstore import VectorStore

# --- Helper for formatting evidence---
def _fmt_evidence(evidences: List[Dict], max_len: int = 4000) -> str:
    return "\n".join([f"[E{i+1}] (p.{e.get('page', 'N/A')}, {e.get('section', 'Section')}): {e['text']}" for i, e in enumerate(evidences)])[:max_len]

# --- High-Performance Q&A Strategy---
def _decompose_question(complex_question: str, llm) -> List[str]:
    is_complex = complex_question.count('?') > 1 or len(complex_question.split()) > 20
    if not is_complex:
        return [complex_question]
    prompt = ("Decompose the user's question into a simple, numbered list of individual questions.\n\n"
              f"USER QUESTION: \"{complex_question}\"\n\nDECOMPOSED QUESTIONS:\n1. ")
    params = GenParams(max_tokens=300, temperature=0.0)
    decomposed_string = "1. " + llm.ask(prompt, params)
    questions = [q.strip() for q in re.findall(r'^\d+\.\s*(.*)', decomposed_string, re.MULTILINE) if q.strip()]
    return questions or [complex_question]

def _answer_single_question(question: str, store: VectorStore, embedder: Embedder, llm) -> Tuple[str, List[Dict]]:
    evidence = retrieve(question, store, embedder, final_k=3)
    if not evidence: return "No relevant information was found for this question.", []
    prompt = ("Using ONLY the provided evidence, answer the user's question in 1-3 concise sentences. "
              "Cite every sentence with [E#]. If the answer isn't in the evidence, state that clearly.\n\n"
              f"EVIDENCE:\n{_fmt_evidence(evidence)}\n\n"
              f"QUESTION: {question}\n\nANSWER:")
    params = GenParams(max_tokens=250, temperature=0.0)
    return llm.ask(prompt, params), evidence

def answer_q_and_a_decomposed(question: str, store: VectorStore, embedder: Embedder, **kwargs) -> Tuple[str, List[Dict]]:
    llm = get_llm()
    sub_questions = _decompose_question(question, llm)
    final_answers, all_evidence = [], []
    for sub_q in sub_questions:
        answer, evidence = _answer_single_question(sub_q, store, embedder, llm)
        if len(sub_questions) > 1: final_answers.append(f"**{sub_q}**\n{answer}")
        else: final_answers.append(answer)
        all_evidence.extend(evidence)
    unique_evidence, seen_ids = [], set()
    for ev in all_evidence:
        if ev.get('id') not in seen_ids:
            unique_evidence.append(ev); seen_ids.add(ev.get('id'))
    return "\n\n".join(final_answers), unique_evidence

# --- Intelligent Query Generation for Novelty Analysis---
def generate_novelty_analysis(question: str, store: VectorStore, embedder: Embedder, *, target_words: int, max_new_tokens: int) -> Tuple[str, List[Dict]]:
    llm = get_llm()
    question_gen_prompt = (
        "You are a research assistant. A user wants to find the 'novelty' or 'contribution' of a document. "
        "Generate a numbered list of 3-4 specific questions to identify the unique aspects of the work.\n"
        "Example questions:\n1. What is the main proposed solution or model?\n2. How does this solution differ from previous methods?\n3. What are the key stated contributions?\n\n"
        f"Now, generate a similar list for the user's request: '{question}'\n\nGENERATED QUESTIONS:\n1. "
    )
    params_qgen = GenParams(max_tokens=200, temperature=0.2)
    generated_qs_str = "1. " + llm.ask(question_gen_prompt, params_qgen)
    sub_questions = [q.strip() for q in re.findall(r'^\d+\.\s*(.*)', generated_qs_str, re.MULTILINE) if q.strip()]
    if not sub_questions: sub_questions = ["What is the primary contribution of this work?"]
    
    answers, all_evidence = [], []
    for sub_q in sub_questions:
        answer, evidence = _answer_single_question(sub_q, store, embedder, llm)
        answers.append(f"**{sub_q}**\n{answer}")
        all_evidence.extend(evidence)
    
    synthesis_context = "\n\n".join(answers)
    synthesis_prompt = (
        "You are a tech analyst. Synthesize the following Questions and Answers into a coherent 'Novelty Analysis' report. "
        "Structure it into sections like 'Main Contribution' and 'Key Differentiators'. "
        f"The report should be about {target_words} words.\n\n"
        f"QUESTIONS AND ANSWERS:\n---\n{synthesis_context}\n---\n\n"
        "NOVELTY ANALYSIS REPORT:"
    )
    params_synthesis = GenParams(max_tokens=max_new_tokens, temperature=0.3)
    final_report = llm.ask(synthesis_prompt, params_synthesis)
    
    unique_evidence, seen_ids = [], set()
    for ev in all_evidence:
        if ev.get('id') not in seen_ids:
            unique_evidence.append(ev); seen_ids.add(ev.get('id'))
            
    return final_report, unique_evidence

# ----------------------- Summarize + Review (Map-Reduce) -----------------------

def map_chunk_to_summary(chunk: Dict, task: str, llm) -> str:
    prompt = (f"A user is trying to '{task}'. Based on the text below, extract and summarize the most important points in 2-3 sentences. "
              "Ignore irrelevant details. If no key points are found, respond with 'No key points found'.\n\n"
              f"TEXT:\n---\n{chunk['text']}\n---\n\nKey Points Summary:")
    params = GenParams(max_tokens=256, temperature=0.1)
    summary = llm.ask(prompt, params)
    return f"Points from page {chunk.get('page', 'N/A')}, section '{chunk.get('section', 'Unknown')}':\n{summary}"

def generate_summary_map_reduce(task: str, all_chunks: List[Dict], *, target_words: int, max_new_tokens: int, progress_placeholder: DeltaGenerator) -> Tuple[str, List[Dict]]:
    llm = get_llm()
    intermediate_summaries = []
    
    if progress_placeholder:
        progress_bar = progress_placeholder.progress(0, text="Step 1/3: Mapping document sections...")
    total_chunks = len(all_chunks)
    
    for i, chunk in enumerate(all_chunks):
        try:
            summary = map_chunk_to_summary(chunk, task, llm)
            if "no key points found" not in summary.lower() and len(summary) > 20:
                intermediate_summaries.append(summary)
        except Exception as e:
            print(f"A chunk failed to summarize: {e}")
        if progress_placeholder:
            progress_bar.progress((i + 1) / total_chunks, text=f"Step 1/3: Mapping sections... ({i+1}/{total_chunks})")

    if not intermediate_summaries:
        if progress_placeholder: progress_placeholder.empty()
        return "Could not generate a summary as no key points were found in the document.", all_chunks

    if progress_placeholder:
        progress_bar.progress(1.0, text="Step 2/3: Combining summaries...")
        
    def group_texts(texts: List[str], group_size: int) -> List[str]:
        return ["\n\n".join(texts[i:i + group_size]) for i in range(0, len(texts), group_size)]

    current_summaries = intermediate_summaries
    
    while len(current_summaries) > 4:  #Target 4 super-summaries instead of 5
        # Combine groups of 4 instead of 5
        grouped_summaries = group_texts(current_summaries, 4)
        next_level_summaries = []
        
        for i, group in enumerate(grouped_summaries):
            if progress_placeholder:
                 progress_bar.progress(i / len(grouped_summaries), text=f"Step 2/3: Combining {len(current_summaries)} summaries...")
            
            prompt = ("You are a summarization assistant. Combine the following summaries into a single, more concise summary. "
                      "Retain all key facts, figures, and concepts.\n\n"
                      f"SUMMARIES TO COMBINE:\n---\n{group}\n---\n\n"
                      "CONCISE SUMMARY:")
            # Slightly reduce max_tokens to be safer
            params = GenParams(max_tokens=350, temperature=0.2)
            combined_summary = llm.ask(prompt, params)
            next_level_summaries.append(combined_summary)
        
        current_summaries = next_level_summaries
    
    if progress_placeholder:
        progress_bar.progress(1.0, text="Step 3/3: Generating final report...")
    
    final_context = "\n\n".join(current_summaries)
    final_prompt = ("You are a professional report writer. Synthesize the following key points into a single, well-structured final report for the task: "
                    f"'{task}'.\nThe output should be about {target_words} words. Write the report directly and cohesively.\n\n"
                    f"KEY POINTS:\n---\n{final_context}\n---\n\nFinal Report:")
    params = GenParams(max_tokens=max_new_tokens, temperature=0.4, top_p=0.9)
    final_output = llm.ask(final_prompt, params)
    
    if progress_placeholder: progress_placeholder.empty()
    return final_output, all_chunks

