import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time

import mlrose_hiive

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score


# QUICK REMARK ON CODE REUSE
# I AGGRESSIVELY USED CODE FROM THE MLROSE-HIIVE GITHUB
# I COPY AND PASTED A LOT!


# Loading the data taken from A1

#### GETTING THE DATA ####

# The wine quality dataset
wine = pd.read_csv('data/wine.csv')
wine = wine.drop('Id', axis=1)

wine_y = wine[["quality"]].copy()
wine_x = wine.drop("quality", axis=1)

wine_x_train, wine_x_test, wine_y_train, wine_y_test = train_test_split(wine_x,wine_y,test_size=.10,random_state=0)

# The pumpkin seeds dataset
pumpkin = pd.read_csv('data/pumpkin.csv')

pumpkin_y = pumpkin[["Class"]].copy()
pumpkin_x = pumpkin.drop("Class", axis=1)

pumpkin_x_train, pumpkin_x_test, pumpkin_y_train, pumpkin_y_test = train_test_split(pumpkin_x,pumpkin_y,test_size=.10,random_state=0)

# Normalize feature data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(pumpkin_x_train)
X_test_scaled = scaler.transform(pumpkin_x_test)

# One hot encode target values
one_hot = OneHotEncoder()
y_train_hot = one_hot.fit_transform(pumpkin_y_train.values.reshape(-1, 1)).todense()
y_test_hot = one_hot.transform(pumpkin_y_test.values.reshape(-1, 1)).todense()


### GD #### 
grid_search_parameters = {
    'max_iters': [2000],
    'learning_rate': [1e-2],
    'activation': [mlrose_hiive.relu],
    'restarts': [0],
} 

nnr = mlrose_hiive.NNGSRunner(
    x_train=X_train_scaled,
    y_train=y_train_hot,
    x_test=X_test_scaled,
    y_test=y_test_hot,
    experiment_name='name',
    algorithm=mlrose_hiive.algorithms.gd.gradient_descent,
    grid_search_parameters=grid_search_parameters,
    iteration_list=range(1, 2001),
    hidden_layer_sizes=[[128]],
    bias=True,
    early_stopping=False,
    clip_max=5,
    max_attempts=2000,
    n_jobs=5,
    seed=1,
    output_directory=None,
    grid_search_scorer_method=accuracy_score
)

start = time.time()
run_stats_df, curves_df, cv_results_df, grid_search_cv = nnr.run()

print("Gradient Descent Numbers:")

y_train_pred = grid_search_cv.predict(X_train_scaled)
y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
print(f"Training Acc: {y_train_accuracy}")

y_test_pred = grid_search_cv.predict(X_test_scaled)
y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
print(f"Test Acc: {y_test_accuracy}")

disp = metrics.ConfusionMatrixDisplay.from_predictions(np.argmax(np.asarray(y_test_hot), axis=1), np.argmax(np.asarray(y_test_pred), axis=1))
disp.figure_.suptitle(f"Confusion Matrix for Gradient Descent")
plt.savefig(f"plots/Confusion Matrix for Gradient Descent.png")
plt.close()

print(f"Function Evaluations: {curves_df['FEvals'][2000]}")
print(f"Wall clock time: {time.time() - start}")

print("Raw data: ")
print(curves_df)

plt.plot(range(1,len(curves_df['Fitness'])+1), curves_df['Fitness'], label="GD")



### RHC ####
grid_search_parameters = {
    'max_iters': [2000],
    'learning_rate': [1e-2],
    'activation': [mlrose_hiive.relu],
    'restarts': [0],
} 

nnr = mlrose_hiive.NNGSRunner(
    x_train=X_train_scaled,
    y_train=y_train_hot,
    x_test=X_test_scaled,
    y_test=y_test_hot,
    experiment_name='name',
    algorithm=mlrose_hiive.algorithms.rhc.random_hill_climb,
    grid_search_parameters=grid_search_parameters,
    iteration_list=range(1, 2001),
    hidden_layer_sizes=[[128]],
    bias=True,
    early_stopping=False,
    clip_max=5,
    max_attempts=2000,
    n_jobs=5,
    seed=1,
    output_directory=None
)

start = time.time()
run_stats_df, curves_df, cv_results_df, grid_search_cv = nnr.run()

print("Random Hill Climbing Numbers:")

y_train_pred = grid_search_cv.predict(X_train_scaled)
y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
print(f"Training Acc: {y_train_accuracy}")

y_test_pred = grid_search_cv.predict(X_test_scaled)
y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
print(f"Test Acc: {y_test_accuracy}")

disp = metrics.ConfusionMatrixDisplay.from_predictions(np.argmax(np.asarray(y_test_hot), axis=1), np.argmax(np.asarray(y_test_pred), axis=1))
disp.figure_.suptitle(f"Confusion Matrix for Random Hill Climbing")
plt.savefig(f"plots/Confusion Matrix for Random Hill Climbing.png")
plt.close()

print(f"Function Evaluations: {curves_df['FEvals'][2000]}")
print(f"Wall clock time: {time.time() - start}")

print("Raw data: ")
print(curves_df)

plt.plot(range(1,len(curves_df['Fitness'])+1), curves_df['Fitness'], label="RHC")


### SA ###

grid_search_parameters = {
    'max_iters': [2000],
    'learning_rate': [1e-2],
    'activation': [mlrose_hiive.relu],
    'restarts': [0],
} 

nnr = mlrose_hiive.NNGSRunner(
    x_train=X_train_scaled,
    y_train=y_train_hot,
    x_test=X_test_scaled,
    y_test=y_test_hot,
    experiment_name='name',
    algorithm=mlrose_hiive.algorithms.sa.simulated_annealing,
    grid_search_parameters=grid_search_parameters,
    iteration_list=range(1, 2001),
    hidden_layer_sizes=[[128]],
    bias=True,
    early_stopping=False,
    clip_max=5,
    max_attempts=1000,
    n_jobs=5,
    seed=1,
    output_directory=None
)

start = time.time()
run_stats_df, curves_df, cv_results_df, grid_search_cv = nnr.run()

print("Simulated Annealing Numbers:")

y_train_pred = grid_search_cv.predict(X_train_scaled)
y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
print(f"Training Acc: {y_train_accuracy}")

y_test_pred = grid_search_cv.predict(X_test_scaled)
y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
print(f"Test Acc: {y_test_accuracy}")

disp = metrics.ConfusionMatrixDisplay.from_predictions(np.argmax(np.asarray(y_test_hot), axis=1), np.argmax(np.asarray(y_test_pred), axis=1))
disp.figure_.suptitle(f"Confusion Matrix for Simulated Annealing")
plt.savefig(f"plots/Confusion Matrix for Simulated Annealing.png")
plt.close()

print(f"Function Evaluations: {curves_df['FEvals'][2000]}")
print(f"Wall clock time: {time.time() - start}")

print("Raw data: ")
print(curves_df)

plt.plot(range(1,len(curves_df['Fitness'])+1), curves_df['Fitness'], label="SA")


### GA #### 
grid_search_parameters = {
    'max_iters': [2000],
    'learning_rate': [1e-2],
    'activation': [mlrose_hiive.relu],
    'restarts': [0],
} 

nnr = mlrose_hiive.NNGSRunner(
    x_train=X_train_scaled,
    y_train=y_train_hot,
    x_test=X_test_scaled,
    y_test=y_test_hot,
    experiment_name='name',
    algorithm=mlrose_hiive.algorithms.ga.genetic_alg,
    grid_search_parameters=grid_search_parameters,
    iteration_list=range(1, 20001),
    hidden_layer_sizes=[[128]],
    bias=True,
    early_stopping=False,
    clip_max=5,
    max_attempts=2000,
    n_jobs=5,
    seed=1,
    output_directory=None
)

start = time.time()
run_stats_df, curves_df, cv_results_df, grid_search_cv = nnr.run()

print("Genetic Algorithm Numbers:")

y_train_pred = grid_search_cv.predict(X_train_scaled)
y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
print(f"Training Acc: {y_train_accuracy}")

y_test_pred = grid_search_cv.predict(X_test_scaled)
y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
print(f"Test Acc: {y_test_accuracy}")

disp = metrics.ConfusionMatrixDisplay.from_predictions(np.argmax(np.asarray(y_test_hot), axis=1), np.argmax(np.asarray(y_test_pred), axis=1))
disp.figure_.suptitle(f"Confusion Matrix for Genetic Algorithm")
plt.savefig(f"plots/Confusion Matrix for Genetic Algorithm.png")
plt.close()

print(f"Function Evaluations: {curves_df['FEvals'][2000]}")
print(f"Wall clock time: {time.time() - start}")

print("Raw data: ")
print(curves_df)

plt.plot(range(1,len(curves_df['Fitness'])+1), curves_df['Fitness'], label="GA")

plt.xlabel("Iterations")
plt.ylabel("Fitness Score (Loss)")
plt.ylim(0, 4)
plt.title(f"Fitness Score versus Iterations")
plt.legend(loc="upper right")
plt.savefig(f"plots/Neural Network curves.png")
plt.close()