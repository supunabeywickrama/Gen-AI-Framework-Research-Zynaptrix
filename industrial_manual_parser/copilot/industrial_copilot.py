import os
from openai import OpenAI
from dotenv import load_dotenv
from semantic_search import search_manual

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")

client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = """You are an expert industrial maintenance assistant and Copilot. 
You answer technical questions based *only* on the provided manual context chunks. 
If the answer is not in the context, clearly state that you do not have enough information to answer.
Always be professional, precise, and format complex instructions into clear steps."""

def ask_copilot(question):
    print(f"User Question: {question}")
    
    # 1. Retrieve context chunks from Vector DB
    retrieved_chunks = search_manual(question, limit=4)
    if not retrieved_chunks:
        print("No context could be retrieved. Ensure database has been seeded.")
        return
        
    context_text = "\n\n".join(
        [f"--- Context from Chunk {c['chunk_id']} ---\n{c['text']}" for c in retrieved_chunks]
    )
    
    user_message = f"Question: {question}\n\nContext Information:\n{context_text}"
    
    print("\nReasoning with GPT (gpt-4o-mini)...")
    
    # 2. Ask GPT
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.2 # Keep it focused and deterministic
    )
    
    answer = response.choices[0].message.content
    
    print("\n" + "="*50)
    print("COPILOT ANSWER:")
    print("="*50)
    print(answer)
    print("="*50)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print("Welcome to the Industrial AI Copilot!")
        question = input("Enter your maintenance question: ")
        
    ask_copilot(question)
