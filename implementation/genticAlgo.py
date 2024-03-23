#nsga3
import numpy as np
from deap import base, creator, tools, algorithms
import random
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from time import time
def GA(users_df, teams_df, id):
    # Desired team size
    TEAM_SIZE = 5 

    # Placeholder for actual Kaggle user data (IDs)
    kaggle_users = users_df['UserId'].tolist()

    # Revised function to count the total number of times developers have worked together.
    def count_collab(dev1, dev2):
        # Find the teams for each developer
        dev1_teams = teams_df[teams_df['UserId'] == dev1][['TeamId', 'CompetitionId']]
        dev2_teams = teams_df[teams_df['UserId'] == dev2][['TeamId', 'CompetitionId']]

        # Check for common teams in the same competition
        common_teams = dev1_teams.merge(dev2_teams, on=['TeamId', 'CompetitionId'])

        # Return the count of common teams
        return len(common_teams)
    
    # Define the fitness function for teams
    # penaltie for individual with the same userId 
    def evalTeam(individual):    
        for i in range(len(individual)):
            for j in range(i+1,len(individual)):
                if(individual[i]==individual[j]):
                    individual[i] = random.choice(kaggle_users)
                
        # Calculate the expertise score for the team
        total_score = 0
        for user_id in individual:
            user_data = users_df.loc[users_df['UserId'] == user_id]
            if not user_data.empty and user_data['Medal'].iloc[0] != 0:
                performanceTier = user_data['PerformanceTier'].iloc[0]
                competition_score = user_data['CompetitionScore'].iloc[0]
                tags_score = user_data['TagScore'].iloc[0]
                # medal_score = user_data['Medal'].iloc[0]
                weight = user_data['Weight'].iloc[0]
                rank = user_data['PrivateLeaderboardRank'].iloc[0]
                maxRank=6430
                normalize_rank = 1 - (rank / maxRank)
                Tf = user_data['DaysFromLastSubmission'].iloc[0]
                total_score += 0.3 * competition_score  +  0.3 * normalize_rank + 0.2 * tags_score * (weight/958 ) +  0.1 * performanceTier/5 + 0.1 * (1 - Tf / 1421) 
                
        expertise_score = total_score / TEAM_SIZE
        

        collab = 0
        e = 0
        Ep = 0
        Vp = TEAM_SIZE
        for i in range(len(individual)):
            for j in range(i+1,len(individual)):
                e = count_collab(individual[i], individual[j])
                if(e > 0):
                    Ep += 1
                    collab += e
        collab_score = Ep / (Vp * ((Vp - 1)/2)) * collab
        
        return expertise_score, -collab_score


    # Set up the DEAP framework
    creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.Fitness)

    # Initialize DEAP's toolbox
    toolbox = base.Toolbox()

    def create_population(n, kaggle_users, team_size):
        # Shuffle the user IDs to ensure random selection
        shuffled_users = random.sample(kaggle_users, len(kaggle_users))
        # Create the population by slicing the shuffled list into teams
        population = [creator.Individual(shuffled_users[i:i + team_size]) for i in range(0, n * team_size, team_size)]
        return population

    # Attribute generator - randomly select a user ID from the pool
    toolbox.register("population", create_population, n=500, kaggle_users=kaggle_users, team_size=TEAM_SIZE)

    def custom_mutate(individual, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                # unique_users = list(set(kaggle_users) - set(individual))
                individual[i] = random.choice(kaggle_users)
        return individual,

    # Genetic operators
    toolbox.register("evaluate", evalTeam)
    # Crossover function for mating 
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", custom_mutate, indpb=0.5)
    # ref_points = tools.uniform_reference_points(nobj=2, p=24)  # Adjust P as needed
    # toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
    toolbox.register("select", tools.selNSGA2)

    # Parameters for the genetic algorithm
    NGEN = 2000
    MU = 500
    CXPB = 0.4

    # Initialize statistics object and logbook
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    # Create initial population and evaluate it
    pop = toolbox.population(n=MU)
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    # Compile statistics about the population
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    # Begin the generational process
    for gen in range(1, NGEN + 1):
        t = time()      
        # Generate offspring and evaluate them
        offspring = algorithms.varAnd(pop, toolbox, CXPB, 1 - CXPB)
      
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population from parents and offspring
        pop = toolbox.select(offspring+pop, MU)
        # Compile statistics about the new population
        record = stats.compile(pop)
        print(logbook.stream)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(f'current {(time() - t):2f}')
        print(f'gen:{gen}, estimate: {(2000 - gen ) * (time() - t):2f}')
    # Extract fitness values from the population
    expertise_scores = [ind.fitness.values[0] for ind in pop]
    collab_scores = [-ind.fitness.values[1] for ind in pop]  # Negate to show original values

    # Plot the Pareto front
    plt.scatter(expertise_scores, collab_scores)
    plt.title('Pareto Front')
    plt.xlabel('Expertise Score')
    plt.ylabel('Collaboration Score')
    filename = f'pareto_front{id}.png'
    plt.savefig(filename)    
    
    # Sort the final population into Pareto fronts
    pareto_fronts = tools.sortNondominated(pop, len(pop))

        # Extract the first Pareto front (best teams)
    best_teams = pareto_fronts[0]
    return best_teams[0]

