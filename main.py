from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
import streamlit as st
from openai import OpenAI
import tiktoken
import sqlite3
import numpy as np
import pandas as pd
from io import BytesIO, StringIO
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import re


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
app = FastAPI()

origins = [
    "http://localhost:3000",   
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,               
    allow_credentials=True,              
    allow_methods=["*"],                  
    allow_headers=["*"],                  
)

client = OpenAI(api_key=api_key)

DB_NAME = "embeddings.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT,
        content TEXT,
        embedding BLOB
    )
    ''')
    conn.commit()
    conn.close()

init_db()

def save_embedding(url: str, content: str, embedding: list):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
    DELETE FROM embeddings
    ''')
    
    cursor.execute('''
    INSERT INTO embeddings (url, content, embedding) VALUES (?, ?, ?)
    ''', (url, content, sqlite3.Binary(np.array(embedding, dtype=np.float32).tobytes())))
    
    conn.commit()
    conn.close()

def num_tokens_from_string(string):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def chunk_text(text, max_tokens=200):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        temp_chunk = current_chunk + [word]
        temp_text = " ".join(temp_chunk)
        new_token_count = num_tokens_from_string(temp_text)
        
        if new_token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def search_most_similar_embedding(user_embedding):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT content, embedding FROM embeddings')
    results = cursor.fetchall()
    conn.close()

    user_embedding = np.array(user_embedding)
    similarities = []

    for content, embedding_blob in results:
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        similarity = cosine_similarity(user_embedding, embedding)
        similarities.append((content, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_similar_contents = [content for content, sim in similarities[:2]]

    return top_similar_contents


class UserData(BaseModel):
    user_message: str

@app.post("/chat/")
async def chat(user_data: UserData):
    response = client.embeddings.create(
        input=user_data.user_message,
        model="text-embedding-ada-002"
    )
    user_embedding = response.data[0].embedding

    most_similar_content = search_most_similar_embedding(user_embedding)
    if most_similar_content:
        async def event_stream():
            stream = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": f"User message: {user_data.user_message}\n\nand this can be to related contents:\n{most_similar_content}\n\nAI:"}],
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        raise HTTPException(status_code=404, detail="No similar content found in embeddings.")


class SheetUrl(BaseModel):
    sheet_url: str

@app.post("/upload_google_sheet/")
async def upload_google_sheet(sheet_url: SheetUrl):
    df = pd.read_csv(convert_google_sheet_url(sheet_url.sheet_url))

    
    df = df.fillna('')  
    dict_data = df.to_dict(orient='records')
    
    docs = [
        ", ".join([f"{key}: {entry[key]}" for key in entry])  
        for entry in dict_data
    ]
    content = "\n".join(docs)
    if content:
        chunks = chunk_text(content, max_tokens=500)
        for chunk in chunks:
            response = client.embeddings.create(
                input=chunk,
                model="text-embedding-ada-002"
            )
            embedding = response.data[0].embedding

            save_embedding(sheet_url.sheet_url, chunk, embedding)

    return {"message": "Data scraped and embeddings created successfully.", "data": dict_data}

def convert_google_sheet_url(url):
    pattern = r'https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)(/edit#gid=(\d+)|/edit.*)?'

    replacement = lambda m: f'https://docs.google.com/spreadsheets/d/{m.group(1)}/export?' + (f'gid={m.group(3)}&' if m.group(3) else '') + 'format=csv'

    new_url = re.sub(pattern, replacement, url)

    return new_url



