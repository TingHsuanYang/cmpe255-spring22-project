# cmpe255-spring22-project

Dataset: <https://www.kaggle.com/datasets/xcherry/games-of-all-time-from-metacritic>

## Usage

### Flask Project

This project needs python3.8 or higher. There might be some problems when running with the M1 CPU. TensorFlow does not currently support the M1 Chip. Please check the  issue [comment](https://github.com/tensorflow/tensorflow/issues/46178#issuecomment-760392204).

1. run the script to install required libraries

    ```shell
    pip install -r requirements.txt
    ```

2. run the script to start flask project

    ```shell
    python main.py
    ```

3. import the request data with [Postman](https://www.postman.com/)  
   ```CMPE 255.postman_collection.json```

### Jupyter Project

Download [Jupyter](https://jupyter.org/) and run the files in notebooks

## Problems

* What is the expected reception from critics/users of a certain game type?
* Which is the most popular genre to create a game on?(given the type,rating and platform of the game).
* What is the general quality of game from a developer?
* What platform will the game be most popular on?

## Approach

* 4/19 - Teammates will be split to tackle different problems
* 4/26 - Each team member makes a model for their problem
* 5/3 - For problem with multiple members, the models are either used in conjunction or the better model is used. Decision depends on performance of models.
* 5/3 - Refine models with feedback from other group members
* 5/10 - Save the models and compile an api to generate predctions with

## Results / Learnings

Models are all subpar in their scores and accuracy. The predictions vary along what is basically a straight line where the predicted output is always in the same range
regardless of what the actual scores are. This is due to a combination of problems.

* Trying to predict numerical field with solely categorical features
* A large feature set once one hot encoded given the variety of unique categories
* A lack of significant correlation in the data

In case of regression,all the features except 2 are categorical which needs encoding and the number of observations are not spread evenly among all these unique categories which makes it difficult to find a relation in the dataset.
The lack of significant correlation in the data is the biggest factor most likely. While the project attempted to determine things like the reception of a game from its genres or platforms, it is probable that there simply isn't any significant signal. It doesn't necessarily matter what genre or platform a game is on. Some games will be good or bad regardless of genre or platform and thus it evens out over a large sample size. This means using these features cannot accurately determine metacritic scores and this problem is not a good application of machine learning.
