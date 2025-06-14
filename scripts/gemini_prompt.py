from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
gemini_apikey = os.getenv("GEMINI_APIKEY")


def get_response(chunks, query):
    client = genai.Client(api_key=gemini_apikey)

    context = "\n\n".join(chunks)
    prompt = f"""Use the following context to answer the question (only give the answer, no formatting).
                Context: {context}
                Question: {query}
                Answer:"""

    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=prompt
    )
    return response.text
