# Master-Project

# UNE APPROCHE AUTOMATIQUE POUR LA RECOMMANDATION DES DÉVELOPPEURS POUR DES TACHES DE DÉVELOPPEMENT DE LOGICIEL

first you need to setup kaggle api
To set up the Kaggle API, follow these steps:

Go to the Kaggle website and sign in to your account.
Click on your profile icon in the top right corner of the screen.
Select "Account" from the dropdown menu.
Click on the "API" tab.
Click on the "Create New API Token" button.
Enter a name for your API token and click on the "Create" button.
Your API token will be displayed on the screen. Copy and save it in a safe place.
To use the Kaggle API, you will need to install the Kaggle API client. You can do this by running the following command in your terminal:

pip install kaggle

after that install this requirements:
pip3 install fastapi  
pip3 install uvicorn  
python3 -m venv venv
source venv/bin/activate
pip3 install -U sentence-transformers
pip3 install deap  
pip3 install pandas
uvicorn api:app --reload
