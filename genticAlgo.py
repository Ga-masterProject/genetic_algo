#nsga3
import numpy as np
from deap import base, creator, tools, algorithms
import random
def GA(users_df, teams_df):
    # Desired team size
    TEAM_SIZE = 4 

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
                    return 0, 0
            
        # Calculate the expertise score for the team
        total_score = 0
        for user_id in individual:
            user_data = users_df.loc[users_df['UserId'] == user_id]
            if not user_data.empty and user_data['Medal'].iloc[0] != 0:
                performanceTier = user_data['PerformanceTier'].iloc[0]
                competition_score = user_data['CompetitionScore'].iloc[0]
                tags_score = user_data['TagScore'].iloc[0]
                medal_score = user_data['Medal'].iloc[0]
                weight = user_data['Weight'].iloc[0]
                rank = user_data['PrivateLeaderboardRank'].iloc[0]
                maxRank=6430
                normalize_rank = 1 - (rank / maxRank)
                Tf = user_data['DaysFromLastSubmission'].iloc[0]
                total_score += (0.5 * competition_score + 0.1 * tags_score + 0.1 * normalize_rank + 0.3 * performanceTier ) * (1 - Tf / 1421) / (medal_score)
                
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

    def custom_mutate(individual, user_list, indpb, pop):
        for i in range(len(individual)):
            if random.random() < indpb:
                # Choose mutation type (inside or outside population)
                if random.random() < 0.5:
                    # Mutate with a user from the current population
                    population_users = [user for ind in pop for user in ind if user != individual[i]]
                    if population_users:
                        individual[i] = random.choice(population_users)
                else:
                    # Mutate with a user from the entire user list
                    individual[i] = random.choice(user_list)
        return individual,

    # Genetic operators
    toolbox.register("evaluate", evalTeam)
    # Crossover function for mating 
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", custom_mutate, user_list=kaggle_users, indpb=0.5)
    ref_points = tools.uniform_reference_points(nobj=2, p=24)  # Adjust P as needed
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

    # Parameters for the genetic algorithm
    NGEN = 2000
    MU = 500
    CXPB = 0.5

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
    print(logbook.stream)
    # Begin the generational process
    for gen in range(1, NGEN + 1):
        # Generate offspring and evaluate them
        
        # Manual Crossover
        offspring = [toolbox.clone(ind) for ind in pop]
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

                # Manual Mutation
        for ind in offspring:
            if random.random() < 1 - CXPB:
                toolbox.mutate(ind,pop=pop)
                del ind.fitness.values
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, MU)

        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    best_teams = tools.selBest(pop, 1)
    
    return best_teams


   
