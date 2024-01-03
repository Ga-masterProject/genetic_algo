pip3 install fastapi  
pip3 install uvicorn  
python3 -m venv venv
source venv/bin/activate
pip3 install -U sentence-transformers
pip3 install deap  
pip3 install pandas
uvicorn api:app --reload
