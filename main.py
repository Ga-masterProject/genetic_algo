import sqlite3
import pandas as pd
from functions import calculate_tags_similarity, text_to_vector, clean_text, calculate_competition_similarity
from genticAlgo import GA
from sklearn.metrics.pairwise import cosine_similarity

def main(new_competition):

#     # connect to kaggle local database
    conn = sqlite3.connect('data3.db')
    c = conn.cursor()
  
    users_df = pd.read_sql('''
    SELECT * from users
    ''', conn)
    teams_df = pd.read_sql('''
    SELECT * from teams
    ''', conn)
    tags_df = pd.read_sql('''
    SELECT * from tags
    ''', conn)
    competitions_df = pd.read_sql('''
    SELECT * from competitions
    ''', conn)

    newCompetition = new_competition[1] + ' ' + new_competition[2] + ' ' + new_competition[3]
    newCompetition = clean_text(newCompetition)
    newCompetition = text_to_vector(newCompetition)
    tags_cleaned = clean_text(new_competition[4])
    tags_vec = text_to_vector(tags_cleaned)
    print('tags vectorize')

    # Create a dataframe for unique competitions
    unique_competitions = competitions_df.drop_duplicates(subset='CompetitionId', keep='first').copy()
    # Add a new column for the similarity score
    unique_competitions['SimilarityScore'] = 0.0
    # Calculate similarity scores for each unique competition
    for index, competition in unique_competitions.iterrows():
        comp = (competition['CompetitionSlug'] if competition['CompetitionSlug'] is not None else '')+ ' ' + (competition['CompetitionTitle'] if competition['CompetitionTitle'] is not None else '') + ' ' + (competition['CompetitionSubTitle'] if competition['CompetitionSubTitle'] is not None else '')
        comp_vector = text_to_vector(clean_text(comp))
        similarity_score = cosine_similarity([newCompetition], [comp_vector])[0][0]
        unique_competitions.at[index, 'SimilarityScore'] = similarity_score
    
    # Create a dataframe for unique tags
    unique_tags = tags_df.drop_duplicates(subset='TagId', keep='first').copy()
    # Add a new column for the similarity score
    unique_tags['SimilarityScore'] = 0.0
    # Calculate similarity scores for each unique tag
    for index, row in unique_tags.iterrows():
        tag = row['TagName']
        tag_vector = text_to_vector(clean_text(tag))
        # Calculate the cosine similarity with the new competition tags
        similarity_score = cosine_similarity([tags_vec], [tag_vector])[0][0]
        unique_tags.at[index, 'SimilarityScore'] = similarity_score


    comp_similarity_dict = dict(zip(unique_competitions.CompetitionId, unique_competitions.SimilarityScore))
    users_df['MostSimilarCompetitionId'] = None
    users_df['CompetitionScore'] = 0.0

    tags_similarity_dict = dict(zip(unique_tags.TagId, unique_tags.SimilarityScore))
    users_df['MostSimilarTagId'] = None
    users_df['TagScore'] = 0.0

    users_df['Weight'] = 0.0
    users_df['DaysFromLastSubmission'] = 0.0
    users_df['PrivateLeaderboardRank'] = 0.0
    users_df['Medal'] = 4.0
    print("give competition similarity to each user")
    for index, user in users_df.iterrows():
        user_id = user['UserId']
        # Filter competitions for the specific user
        user_competitions = competitions_df[competitions_df['UserId'] == user_id]

        # Filter tags for the specific user
        user_tags = tags_df[tags_df['UserId'] == user_id]

        # Initialize variables to store the highest similarity score and corresponding competition ID
        highest_similarity = 0
        most_similar_competition_id = None
        last_comp_id = None
        combined_comp_similarity = 0
        competition_number = 0
        maxDate = float('inf')
        rank = 0

        # Initialize variables to store the highest similarity score and corresponding tag ID
        highest_tag_similarity = 0
        most_similar_tag_id = None

        # Find the most similar competition for the user
        for _, comp_row in user_competitions.iterrows():
            comp_id = comp_row['CompetitionId']
            similarity_score = comp_similarity_dict.get(comp_id, 0)
            if similarity_score < 0 : 
                similarity_score = 0
                most_similar_competition_id = comp_id

            if similarity_score >= 0.5:
                combined_comp_similarity += similarity_score
                days = teams_df[(teams_df['UserId'] == user_id) & (teams_df['CompetitionId'] == comp_id)]['DaysFromLastSubmission']
        
                if maxDate >= days.iloc[0] :
                    maxDate = days.iloc[0]
                    last_comp_id = comp_id
        
                competition_number+=1

            elif similarity_score >= highest_similarity:
                highest_similarity = similarity_score
                most_similar_competition_id = comp_id

        if competition_number > 0:
            highest_similarity = combined_comp_similarity/competition_number
            most_similar_competition_id = last_comp_id

        medal = teams_df[(teams_df['UserId'] == user_id) & (teams_df['CompetitionId'] == most_similar_competition_id)]['Medal']
        PrivateLeaderboardRank = teams_df[(teams_df['UserId'] == user_id) & (teams_df['CompetitionId'] == most_similar_competition_id)]['PrivateLeaderboardRank']
        DaysFromLastSubmission = teams_df[(teams_df['UserId'] == user_id) & (teams_df['CompetitionId'] == most_similar_competition_id)]['DaysFromLastSubmission']

        # Find the most similar tag for the user
        for _, tag_row in user_tags.iterrows():
            tag_id = tag_row['TagId']
            similarity_score = tags_similarity_dict.get(tag_id, 0)

            if similarity_score > highest_tag_similarity:
                highest_tag_similarity = similarity_score
                most_similar_tag_id = tag_id
            
        Weight = tags_df[(tags_df['UserId'] == user_id) & (tags_df['TagId'] == most_similar_tag_id)]['Weight']

        # Assign the most similar competition ID and the similarity score
        users_df.at[index, 'MostSimilarCompetitionId'] = most_similar_competition_id
        users_df.at[index, 'CompetitionScore'] = highest_similarity

        # Assign the most similar tag ID and the similarity score
        users_df.at[index, 'MostSimilarTagId'] = most_similar_tag_id
        users_df.at[index, 'DaysFromLastSubmission'] = DaysFromLastSubmission.iloc[0]
        users_df.at[index, 'PrivateLeaderboardRank'] = PrivateLeaderboardRank.iloc[0]

        if(most_similar_tag_id == 0):
            users_df.at[index, 'TagScore'] = 0
    
        else: 
            users_df.at[index, 'TagScore'] = highest_tag_similarity

        if Weight.empty:
            users_df.at[index, 'Weight'] = 0
        else:
            users_df.at[index, 'Weight'] = Weight.iloc[0]

        if medal.empty:
            users_df.at[index, 'Medal'] = 4.0
        else:
            users_df.at[index, 'Medal'] = medal.iloc[0]
        
    print("most similar competition completed")

    print("start GA")
    best_teams = GA(users_df, teams_df, new_competition[0])
    
    # example of best_teams = [[10258726, 2062758, 1166873]]
    id_tuple = tuple(best_teams)

    best_teams = ','.join([str(name) for name in best_teams])

    # Construct the placeholder string ('?,?,?') for the SQL IN clause
    placeholders = ','.join('?' for _ in id_tuple)
    # Create the SQL query string using the placeholders
    query = f'SELECT DisplayName FROM users WHERE UserId IN ({placeholders})'
    # Execute the query, passing the id_tuple as a parameter
    c.execute(query, id_tuple)
    # Fetch the results
    results = c.fetchall()
    # example of results [('canlion',), ('Theo Viel',), ('Nazarko99',)]
    names_combined = ', '.join([name[0] for name in results])
    # names_combined = 'canlion, Theo Viel, Nazarko99'
    return best_teams, names_combined
