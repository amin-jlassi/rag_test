import pdfplumber
import re
import os 
from dotenv import load_dotenv
from openai import OpenAI
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
        chunks = re.split(r"EXERCICE [0-9]+" , text)
        matches = re.match(r"EXERCICE [0-9]+" , text)

        print(matches)
        
        return chunks
    
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

pdf_path = "dataset/Series2.pdf"
loader = PdfLoader(pdf_path)
text = loader.exctract_text()
chunks = Chunker().start_chunking(text=text)
print(len(chunks))