from scripts.retriever import get_relevant_chunks
from scripts.gemini_prompt import get_response
import sys
import os
from contextlib import contextmanager


@contextmanager
def suppress_stdout():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout


print("Welcome to WikiRAG Chat! Type your question below (type 'exit' to quit):")
while True:
    try:
        query = input('You: ').strip()
        print('\n')
        if query.lower() == 'exit':
            break
    except KeyboardInterrupt:
        break

    with suppress_stdout():
        chunks = get_relevant_chunks(query=query)
        response = get_response(chunks=chunks, query=query)

    print('Bot: ' + response + '\n')
