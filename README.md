# local_RAG
### install ollama and LLM models 
   install llama3.1 7B
  ```
  ollama run llama 3.1
  ```
  install llama3.2 3B 
  ```
  ollama run llama3.2:3b
  ```
  install embedding model
  ```
  ollama pull nomic-embed-text
  ```
### Create a virtual environment
```
 python3 -m venv venv
```
### Activate it 
```
source venv/bin/activate
```

### Install the required packages in root folder
```
pip install -r requirements.txt
```

### install tesserect-ocr in your laptop or machine 
```
  sudo apt install -y tesseract-ocr
```
### install german language tesserect-ocr 
```
  sudo apt-get install tesseract-ocr-deu
```
### add it to your path 
```
 export TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
```
 
### Navigate to the backend folder
```
 cd backend 
 ```
### Run the backend
``` 
 uvicorn main:app --reload
```

### Navigate to the frontend folder in separete terminal 
 ```
 cd frontend
 ```
### run the frontend 
```
  streamlit run main.py
```  

### Daynmic prompt 
  you can add or copy past a prompt in the prompt.txt file 








>>>>>>> a13395b (Track backend folder as regular directory)
