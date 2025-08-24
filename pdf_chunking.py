import pdfplumber 
import re
from datetime import datetime

start_time = datetime.now()

whole_pdf_text = ""
with pdfplumber.open("Series1.pdf") as pdf : 

    for page in pdf.pages : 
        whole_pdf_text += page.extract_text()

matches = re.findall(r"Exercice [0-9]+" , whole_pdf_text)
print(matches)
    
chunks = re.split(r"Exercice [0-9]+" , whole_pdf_text)


chunks = chunks[1:len(chunks)]

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embeddings = model.encode(chunks)

print(embeddings.shape)  




import faiss
from pickle import dump , load

"""d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

faiss.write_index(index , "faiss_index.bin")"""




index = faiss.read_index("faiss_index.bin")

print(index.ntotal)






query = "excerice qui contient les coures suivant : enregistrement , tableau , fichier "
query_emb = model.encode([query] , convert_to_numpy=True)
distances , indices = index.search(query_emb , k=2)

retrieved_chunks = [chunks[i] for i in indices[0]]



prompt = f"""
ce sont des excercice predefinis

{retrieved_chunks}

    maintenent , generer  3 nouveau  exercices  en francais avec un exemple d'execution

    difficulté : difficile
     

"""

import ollama
import requests


client = ollama.Client()
model = "mistral"


response = client.generate(model=model , prompt=prompt )

end_time = datetime.now()

print(end_time - start_time , "sec")
print(response.response)





