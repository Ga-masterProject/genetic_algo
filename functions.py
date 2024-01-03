from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


STOPWORDS = set(stopwords.words('english'))

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# transform text to vector
def text_to_vector(sentence):
    return model.encode(sentence, convert_to_tensor=True)

# Define a function to clean the text
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    # Remove any HTML tags
    text = re.sub('<.*?>', '', text)
    # Remove non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase and tokenize
    words = word_tokenize(text.lower())
    # Remove stopwords
    words = [word for word in words if word not in STOPWORDS]
    return ' '.join(words)

# calculate competition similarity 
def calculate_competition_similarity(new_comp_vector, user_competitions):
    
    # Initialize the highest similarity score and corresponding competition ID
    highest_similarity = 0
    most_similar_competition_id = None
    
    # Loop over each competition the user has participated in
    for _, row in user_competitions.iterrows():
        # Get the vector for the competition description
        competition = row['CompetitionSlug'] + ' ' + row['CompetitionTitle'] + ' ' + row['CompetitionSubTitle'] 
        competition_vector = text_to_vector(competition)
        
        # Calculate the cosine similarity with the new competition
        similarity_score = cosine_similarity([new_comp_vector], [competition_vector])[0][0]
        
        # If this score is higher than the highest score, update the highest score and competition ID
        if similarity_score > highest_similarity:
            highest_similarity = similarity_score
            most_similar_competition_id = row['CompetitionId']
    
    return most_similar_competition_id, highest_similarity

def calculate_tags_similarity(new_comp_tags_vector, user_tags):
    
    # Initialize the highest similarity score 
    highest_similarity = 0
    
    # Loop over each tag the user have 
    for _, row in user_tags.iterrows():
        # Get the vector for the competition description
        tag = row['TagName']
        tag_vector = text_to_vector(tag)
        
        # Calculate the cosine similarity with the new competition tags
        similarity_score = cosine_similarity([new_comp_tags_vector], [tag_vector])[0][0]
        
        # If this score is higher than the highest score, update the highest score 
        if similarity_score > highest_similarity:
            highest_similarity = similarity_score
    
    return highest_similarity