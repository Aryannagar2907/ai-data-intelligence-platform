from openai import OpenAI
import sqlite3

client = OpenAI()


def is_safe_query(query: str):
    query = query.lower()

    validation = ["drop", "delete", "update", "insert", "alter"]

    return not any(word in query for word in validation)


def run_sql_agent(question: str):

    prompt = f"""
You are an expert SQL generator.

Convert the question into SQL.

Table: customers(id, name, age, city, country, email, phone, created_at)

Rules:
- Only SELECT queries allowed
- CTE (WITH) allowed
- No explanation
- No comments

Question: {question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    sql_query = response.choices[0].message.content.strip()

    # Clean markdown if present
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

    # ✅ Correct validation
    sql_lower = sql_query.lower()

    if not (sql_lower.startswith("select") or sql_lower.startswith("with")):
        return {"error": "Only SELECT/CTE queries allowed"}

    if not is_safe_query(sql_query):
        return {"error": "Unsafe query detected"}

    # Connect to SQLite
    conn = sqlite3.connect("data/data.db")
    cursor = conn.cursor()

    try:
        cursor.execute(sql_query)

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        result = [dict(zip(columns, row)) for row in rows]

    except Exception as e:
        result = {"error": str(e)}

    conn.close()

    return result