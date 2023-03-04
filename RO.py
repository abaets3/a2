import mlrose_hiive
import numpy as np
import matplotlib.pyplot as plt
import time

# QUICK REMARK ON CODE REUSE
# I AGGRESSIVELY USED CODE FROM THE MLROSE-HIIVE GITHUB
# I COPY AND PASTED A LOT!

def hyper_parameter_tuning_GA():
    max_attempts = 500
    max_iterations = 500
    problem_size = 50
    queens = 20
    problem = mlrose_hiive.QueensGenerator().generate(seed=1, size=queens)
    ga = mlrose_hiive.GARunner(problem=problem,experiment_name="name",output_directory=None,seed=1,iteration_list=np.arange(max_iterations),population_sizes=[200], mutation_rates=[.2])
    df_run_stats, df_run_curves = ga.run()
    best_fitness = df_run_curves['Fitness'].min()
    best_runs = df_run_curves[df_run_curves['Fitness'] == best_fitness]
    print(best_runs)

def hyper_parameter_tuning_SA():
    max_attempts = 1000
    max_iterations = 1000
    problem_size = 100
    queens = 20
    problem = mlrose_hiive.TSPGenerator().generate(seed=1, number_of_cities=100)
    ga = mlrose_hiive.SARunner(problem=problem,experiment_name="name",output_directory=None,seed=1,iteration_list=np.arange(max_iterations), temperature_list=[0.1, 0.5, 0.75, 1.0, 2.0, 5.0], decay_list=[mlrose_hiive.GeomDecay])
    df_run_stats, df_run_curves = ga.run()
    best_fitness = df_run_curves['Fitness'].max()
    best_runs = df_run_curves[df_run_curves['Fitness'] == best_fitness]
    print(best_runs)

def hyper_parameter_tuning_MIMIC():
    max_attempts = 10
    max_iterations = 10
    problem_size = 50
    queens = 20
    problem = mlrose_hiive.QueensGenerator().generate(seed=1, size=queens)
    problem.set_mimic_fast_mode(True)
    ga = mlrose_hiive.MIMICRunner(problem=problem,experiment_name="name",output_directory=None,seed=1,iteration_list=np.arange(max_iterations),population_sizes=[200], keep_percent_list=[.5], use_fast_mimic=True)
    df_run_stats, df_run_curves = ga.run()
    best_fitness = df_run_curves['Fitness'].min()
    best_runs = df_run_curves[df_run_curves['Fitness'] == best_fitness]
    print(best_runs)    

def run_experiment_four_peaks():
    max_attempts = 1000
    max_iterations = 1000
    problem_size = 100

    fitness = mlrose_hiive.FourPeaks()

    start = time.time()
    problem_fit = mlrose_hiive.DiscreteOpt(length=problem_size, fitness_fn=fitness, maximize=True)
    _, _, curve = mlrose_hiive.genetic_alg(problem_fit, random_state=1, curve=True, max_iters=max_iterations, max_attempts=max_attempts, pop_size=200, mutation_prob=.1)
    print(f"Time for GA: {time.time()-start}")
    print(f"Function Evals for GA: {curve[-1,1]}")
    plt.plot(range(1,len(curve[:,0])+1), curve[:,0], label="GA")

    start = time.time()
    problem_fit = mlrose_hiive.DiscreteOpt(length=problem_size, fitness_fn=fitness, maximize=True)
    _, _, curve = mlrose_hiive.random_hill_climb(problem_fit, random_state=1, curve=True, max_iters=max_iterations, max_attempts=max_attempts)
    print(f"Time for RHC: {time.time()-start}")
    print(f"Function Evals for RHC: {curve[-1,1]}")
    plt.plot(range(1,len(curve[:,0])+1), curve[:,0], label="RHC")

    start = time.time()
    problem_fit = mlrose_hiive.DiscreteOpt(length=problem_size, fitness_fn=fitness, maximize=True)
    _, _, curve = mlrose_hiive.simulated_annealing(problem_fit, random_state=1, curve=True, max_iters=max_iterations, max_attempts=max_attempts)
    print(f"Time for SA: {time.time()-start}")
    print(f"Function Evals for SA: {curve[-1,1]}")
    plt.plot(range(1,len(curve[:,0])+1), curve[:,0], label="SA")

    start = time.time()
    problem_fit = mlrose_hiive.DiscreteOpt(length=problem_size, fitness_fn=fitness, maximize=True)
    problem_fit.set_mimic_fast_mode(True)
    _, _, curve = mlrose_hiive.mimic(problem_fit, random_state=1, curve=True, max_iters=max_iterations, max_attempts=max_attempts, pop_size=100, keep_pct=.1)
    print(f"Time for MIMIC: {time.time()-start}")
    print(f"Function Evals for MIMIC: {curve[-1,1]}")
    plt.plot(range(1,len(curve[:,0])+1), curve[:,0], label="MIMIC")

    plt.xlabel("Iterations")
    plt.ylabel("Fitness Score")
    plt.title(f"Fitness Score versus Iterations")
    plt.legend(loc="upper left")
    plt.savefig(f"plots/FourPeaks.png")
    plt.close()

    sizes = [50, 75, 100, 125, 150, 175, 200]
    ga = []
    rhc = []
    sa = []
    mimic = []
    for i in sizes:
        fitness = mlrose_hiive.FourPeaks()
        problem_fit = mlrose_hiive.DiscreteOpt(length=i, fitness_fn=fitness, maximize=True)
        _, _, curve = mlrose_hiive.genetic_alg(problem_fit, random_state=1, curve=True, max_iters=max_iterations, max_attempts=max_attempts)
        ga.append(curve[-1:,0])

        problem_fit = mlrose_hiive.DiscreteOpt(length=i, fitness_fn=fitness, maximize=True)
        _, _, curve = mlrose_hiive.random_hill_climb(problem_fit, random_state=1, curve=True, max_iters=max_iterations, max_attempts=max_attempts)
        rhc.append(curve[-1:,0])

        problem_fit = mlrose_hiive.DiscreteOpt(length=i, fitness_fn=fitness, maximize=True)
        _, _, curve = mlrose_hiive.simulated_annealing(problem_fit, random_state=1, curve=True, max_iters=max_iterations, max_attempts=max_attempts)
        sa.append(curve[-1:,0])

        problem_fit = mlrose_hiive.DiscreteOpt(length=i, fitness_fn=fitness, maximize=True)
        problem_fit.set_mimic_fast_mode(True)
        _, _, curve = mlrose_hiive.mimic(problem_fit, random_state=1, curve=True, max_iters=max_iterations, max_attempts=max_attempts, pop_size=100, keep_pct=.1)
        mimic.append(curve[-1:,0])
    
    plt.plot(sizes, ga, label="GA")
    plt.plot(sizes, sa, label="SA")
    plt.plot(sizes, rhc, label="RHC")
    plt.plot(sizes, mimic, label="MIMIC")

    plt.xlabel("Problem Size")
    plt.ylabel("Fitness Score")
    plt.title(f"Fitness Score versus Problem Size")
    plt.legend(loc="upper left")
    plt.savefig(f"plots/FourPeaksProblemSize.png")
    plt.close()

def run_experiment_continuous_peaks():
    max_attempts = 3000
    max_iterations = 3000
    problem_size = 100 

    fitness = mlrose_hiive.ContinuousPeaks()
    problem_fit = mlrose_hiive.DiscreteOpt(length=problem_size, fitness_fn=fitness, maximize=True)

    start = time.time()
    problem_fit = mlrose_hiive.DiscreteOpt(length=problem_size, fitness_fn=fitness, maximize=True)
    _, _, curve = mlrose_hiive.genetic_alg(problem_fit, random_state=1, curve=True, max_iters=max_iterations, max_attempts=max_attempts, pop_size=150, mutation_prob=.01)
    print(f"Time for GA: {time.time()-start}")
    print(f"Function Evals for GA: {curve[-1,1]}")
    plt.plot(range(1,len(curve[:,0])+1), curve[:,0], label="GA")

    start = time.time()
    problem_fit = mlrose_hiive.DiscreteOpt(length=problem_size, fitness_fn=fitness, maximize=True)
    _, _, curve = mlrose_hiive.random_hill_climb(problem_fit, random_state=1, curve=True, max_iters=max_iterations, max_attempts=max_attempts)
    print(f"Time for RHC: {time.time()-start}")
    print(f"Function Evals for RHC: {curve[-1,1]}")
    plt.plot(range(1,len(curve[:,0])+1), curve[:,0], label="RHC")

    start = time.time()
    problem_fit = mlrose_hiive.DiscreteOpt(length=problem_size, fitness_fn=fitness, maximize=True)
    _, _, curve = mlrose_hiive.simulated_annealing(problem_fit, random_state=1, curve=True, max_iters=max_iterations, max_attempts=max_attempts, schedule=mlrose_hiive.GeomDecay(init_temp=10.0))
    print(f"Time for SA: {time.time()-start}")
    print(f"Function Evals for SA: {curve[-1,1]}")
    plt.plot(range(1,len(curve[:,0])+1), curve[:,0], label="SA")

    start = time.time()
    problem_fit = mlrose_hiive.DiscreteOpt(length=problem_size, fitness_fn=fitness, maximize=True)
    problem_fit.set_mimic_fast_mode(True)
    _, _, curve = mlrose_hiive.mimic(problem_fit, random_state=1, curve=True, max_iters=max_iterations, max_attempts=max_attempts, pop_size=100, keep_pct=.12)
    print(f"Time for MIMIC: {time.time()-start}")
    print(f"FUnction Evals for MIMIC: {curve[-1,1]}")
    plt.plot(range(1,len(curve[:,0])+1), curve[:,0], label="MIMIC")

    plt.xlabel("Iterations")
    plt.ylabel("Fitness Score")
    plt.title(f"Fitness Score versus Iterations")
    plt.legend(loc="upper left")
    plt.savefig(f"plots/ContinuousPeaks.png")
    plt.close()

    sizes = [50, 75, 100, 125, 150, 175, 200]
    ga = []
    rhc = []
    sa = []
    mimic = []
    for i in sizes:
        fitness = mlrose_hiive.ContinuousPeaks()
        problem_fit = mlrose_hiive.DiscreteOpt(length=i, fitness_fn=fitness, maximize=True)
        _, _, curve = mlrose_hiive.genetic_alg(problem_fit, random_state=1, curve=True, max_iters=max_iterations, max_attempts=max_attempts, pop_size=150, mutation_prob=.01)
        ga.append(curve[-1:,0])

        problem_fit = mlrose_hiive.DiscreteOpt(length=i, fitness_fn=fitness, maximize=True)
        _, _, curve = mlrose_hiive.random_hill_climb(problem_fit, random_state=1, curve=True, max_iters=max_iterations, max_attempts=max_attempts)
        rhc.append(curve[-1:,0])

        problem_fit = mlrose_hiive.DiscreteOpt(length=i, fitness_fn=fitness, maximize=True)
        _, _, curve = mlrose_hiive.simulated_annealing(problem_fit, random_state=1, curve=True, max_iters=max_iterations, max_attempts=max_attempts, schedule=mlrose_hiive.GeomDecay(init_temp=10.0))
        sa.append(curve[-1:,0])

        problem_fit = mlrose_hiive.DiscreteOpt(length=i, fitness_fn=fitness, maximize=True)
        problem_fit.set_mimic_fast_mode(True)
        _, _, curve = mlrose_hiive.mimic(problem_fit, random_state=1, curve=True, max_iters=max_iterations, max_attempts=max_attempts, pop_size=100, keep_pct=.12)
        mimic.append(curve[-1:,0])
    
    plt.plot(sizes, ga, label="GA")
    plt.plot(sizes, sa, label="SA")
    plt.plot(sizes, rhc, label="RHC")
    plt.plot(sizes, mimic, label="MIMIC")

    plt.xlabel("Problem Size")
    plt.ylabel("Fitness Score")
    plt.title(f"Fitness Score versus Problem Size")
    plt.legend(loc="upper left")
    plt.savefig(f"plots/ContinuousPeaksProblemSize.png")
    plt.close()

def run_experiment_one_max():
    for i in [64, 128, 256, 512]:
        max_attempts = 50
        max_iterations = 50
        problem_size = i
        fitness = mlrose_hiive.OneMax()
        problem_fit = mlrose_hiive.DiscreteOpt(length=problem_size, fitness_fn=fitness, maximize=True)

        start = time.time()
        _, _, curve = mlrose_hiive.genetic_alg(problem_fit, random_state=1, curve=True, max_iters=max_iterations, max_attempts=max_attempts, pop_size=200, mutation_prob=.2)
        if i == 64: print(f"Time for GA: {time.time()-start}")
        if i == 64: print(f"Function Evals for GA: {curve[-1,1]}")
        plt.plot(range(1,len(curve[:,0])+1), curve[:,0], label="GA")

        start = time.time()
        _, _, curve = mlrose_hiive.random_hill_climb(problem_fit, random_state=1, curve=True, max_iters=max_iterations, max_attempts=max_attempts)
        if i == 64: print(f"Time for RHC: {time.time()-start}")
        if i == 64: print(f"Function Evals for RHC: {curve[-1,1]}")
        plt.plot(range(1,len(curve[:,0])+1), curve[:,0], label="RHC")

        start = time.time()
        _, _, curve = mlrose_hiive.simulated_annealing(problem_fit, random_state=1, curve=True, max_iters=max_iterations, max_attempts=max_attempts)
        if i == 64: print(f"Time for SA: {time.time()-start}")
        if i == 64: print(f"Function Evals for SA: {curve[-1,1]}")
        plt.plot(range(1,len(curve[:,0])+1), curve[:,0], label="SA")

        start = time.time()
        problem_fit.set_mimic_fast_mode(True)
        _, _, curve = mlrose_hiive.mimic(problem_fit, random_state=1, curve=True, max_iters=max_iterations, max_attempts=max_attempts, pop_size=300, keep_pct=.1)
        if i == 64: print(f"Time for MIMIC: {time.time()-start}")
        if i == 64: print(f"FUnction Evals for MIMIC: {curve[-1,1]}")
        plt.plot(range(1,len(curve[:,0])+1), curve[:,0], label="MIMIC")

        plt.xlabel("Iterations")
        plt.ylabel("Fitness Score")
        plt.title(f"Fitness Score versus Iterations with a Problem Size of {i}")
        plt.legend(loc="lower right")
        plt.savefig(f"plots/OneMaxProblemSizeOf{i}.png")
        plt.close()


# Running these three functions will generate all the charts used in the RO section of the report
run_experiment_four_peaks()
run_experiment_continuous_peaks()
run_experiment_one_max()

# The hyper_parameter_tuning_* functions were used to play around with the algorithms.
# They do not generate any charts or data used in the write up.