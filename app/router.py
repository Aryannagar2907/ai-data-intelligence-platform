from openai import OpenAI

clinet = OpenAI()

def route(question: str):
    promt= f"""
    Hey you are an intelligent router.

    You have to classify the user question into one of these:
    - SQL ( if the question is related to SQL queries, database, you question is like a SQL query)
    - RAG (if question is about document/ knowledge)
    - Reason (if logical/ general reasoning)

    only return one word : SQL or RAG or REASON

    Question: {question}
    """
    response= clinet.chat.completions.create(
        model= 'gpt-4o-mini',
        messages=[{"role":"user", "content": promt}]
    )
    decision= response.choices[0].message.content.strip()
    return decision