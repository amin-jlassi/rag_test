import pdfplumber
import faiss
import re
import os 
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from pickle import load , dump
from datetime import datetime


start = datetime.now()

load_dotenv()

openai_api_key = os.getenv('openai_api_key')

client = OpenAI(api_key=openai_api_key)




class PdfLoader: 
    def __init__(self , file_path):
        self.file_path = file_path
    
    def exctract_text(self) : 
        text = ""
        with pdfplumber.open(self.file_path) as pdf : 
            for page in pdf.pages : 
                text += page.extract_text()
        return text
    
    def exctract_tables(self) : 
        with pdfplumber.open(self.file_path) as pdf : 
            for page in pdf.pages : 
                table = page.extract_tables()
                print(table)    

        

class Chunker : 
    
    def __init__(self) :
        pass
    def start_chunking(self , text) : 
        chunks = re.split(r"Exercice [0-9]+" , text)
        
        print(chunks[0])
        return chunks[1:len(chunks)]
    
    def save_chunk(self , chunks) : 
        with open("chunks.pkl" , "ab") as f :
            dump(chunks , f)
            print('chunk saved successfuly ...')
    def load_chunks(self) : 
        chunk_list = []
        with open("chunks.pkl" , "rb") as f :
            EOF = False
            while not EOF :
                try : 
                    pdf_chunks = load(f)
                except : 
                    EOF = True
                else :
                    for chunk in pdf_chunks : 
                        chunk_list.append(chunk)
                    
        print(len(chunk_list))
        return chunk_list


class Embeddings : 

    def __init__(self  ):
        self.embeddings = None
    
    def generate_embeddings(self , chunks) : 
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embeddings = model.encode(chunks)
        

        return self.embeddings
    def save(self , index_path = "index.bin") :

        if self.embeddings is None : 
            raise ValueError("You must generate embeddings before saving")

        
        d = self.embeddings.shape[1]

        if os.path.exists(index_path) : 
            index = faiss.read_index(index_path)
            index.add(self.embeddings)
            print(f"Loaded existing index with {index.ntotal} vectors before adding.")
        else :
            index = faiss.IndexFlatL2(d) 
            index.add(self.embeddings) 
            print("Created new FAISS index.") 
        
        faiss.write_index(index , index_path)
        print("New embeddings saved. Total vectors in index:", index.ntotal)
    
    def get_index(self , index_path = "index.bin") : 
        index = faiss.read_index(index_path)
        return index



        




    
from utils import exist , savePdf_inFile


pdf_path = "dataset/Series2.pdf"
if not exist(pdf_path) :
    loader = PdfLoader(pdf_path)
    text = loader.exctract_text()
    chunks = Chunker().start_chunking(text=text)
    Chunker().save_chunk(chunks)
    
    embeddings = Embeddings()
    x = embeddings.generate_embeddings(chunks=chunks)
    print(x)
    embeddings.save()
    savePdf_inFile(pdf_path)
else :
    print("file already exists")

index = Embeddings().get_index()
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

chunks = Chunker().load_chunks()





query = "enregistrement , tableau , fichier"
query_emb = model.encode([query] , convert_to_numpy=True)

distances , indices = index.search(query_emb , k=2)
retrieved_chunks = [chunks[i] for i in indices[0]] 

print(distances)

prompt = f"""
Voici des exemples d'exercices :

{retrieved_chunks}

Maintenant, génère 2 nouveaux exercices en français, 
en gardant le même style que les exemples, 
et fournis pour chacun un exemple d'exécution.
nb : si tu vas générer des tableaus tu doit les encadrer (pour etre claire)
"""


response = client.responses.create(
    model="gpt-4o-mini" , 
    input=prompt , 
    store=True , 
    temperature=0.8
)
end = datetime.now()
print(end-start)
print(response.output_text)  