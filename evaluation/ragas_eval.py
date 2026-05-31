import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).parent.parent))

from knowledge_base.retriever import retrieve

def generate_answer(question: str) -> tuple:
    import requests
    retrieved = retrieve(question, top_k=5)
    context_texts = [r["text"] for r in retrieved]
    context = "\n".join(context_texts)
    
    prompt = f"""You are a medical triage assistant. Based on the following clinical context, answer the patient's question clearly and concisely.

Context:
{context}

Question: {question}

Provide a clear triage assessment in 3-4 sentences."""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "medassist", "prompt": prompt, "stream": False},
            timeout=300
        )
        data = response.json()
        answer = data.get("response", "Unable to generate response")
        if not answer:
            answer = "Unable to generate response"
    except Exception as e:
        print(f"Error: {e}")
        answer = "Unable to generate response"
    
    return answer, context_texts

def run_ragas_evaluation():
    with open("evaluation/test_sets/manual_eval_set.json") as f:
        eval_pairs = json.load(f)
    
    print(f"Running Ragas evaluation on {len(eval_pairs)} examples...")
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for i, pair in enumerate(eval_pairs):
        question = pair["question"]
        ground_truth = pair["answer"]
        print(f"Processing {i+1}/{len(eval_pairs)}: {question[:60]}...")
        answer, context_texts = generate_answer(question)
        questions.append(question)
        answers.append(answer)
        contexts.append(context_texts)
        ground_truths.append(ground_truth)
        print(f"  Answer: {answer[:80]}...")
    
    print("\nAll answers generated. Running Ragas scoring...")
    
    from datasets import Dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })
    
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_community.embeddings import HuggingFaceEmbeddings
    
    groq_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
        max_retries=3,
        request_timeout=60
    )
    
    ragas_llm = LangchainLLMWrapper(groq_llm)
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=embeddings,
        raise_exceptions=False
    )
    
    def safe_float(val):
        try:
            f = float(val)
            return 0.0 if str(f) == "nan" else f
        except:
            return 0.0
    
    scores = {
        "faithfulness": safe_float(result["faithfulness"]),
        "answer_relevancy": safe_float(result["answer_relevancy"]),
        "context_precision": safe_float(result["context_precision"]),
        "context_recall": safe_float(result["context_recall"])
    }
    
    print("\n=== RAGAS EVALUATION RESULTS ===")
    for metric, score in scores.items():
        status = "PASS" if score >= 0.60 else "NEEDS IMPROVEMENT"
        print(f"{metric}: {score:.3f} [{status}]")
    
    output_path = Path("evaluation/ragas_scores.json")
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)
    
    print(f"\nScores saved to {output_path}")
    return scores

if __name__ == "__main__":
    run_ragas_evaluation()