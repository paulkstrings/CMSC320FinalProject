# Uncovering Trends and Patterns in NBA Player Statistics
### Fall 2024 Data Science Project
### Paul Kettlestrings
I worked on this project by myself, and thus was the sole contributer for every part of the project.

## Introduction
For this project, I chose to look at NBA statistics. I love watching basketball, and I wanted to see if some assumptions I had about the game would be backed up by the data. I looked at the effect of position and experience on player performance, as well as how a player's minutes impacts their points per game. I also wanted to use the data to make some predictions about the upcoming season, such as who will win the MVP award and the Most Improved Player award.

## Data Curation
The dataset I am using for this project is [NBA Stats (1947-present)](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats/data), found on kaggle.com. The dataset includes numerous csv files with player stats, team stats, award winners, and more, going all the way back to the founding of the NBA. We will be using the PlayerPerGame.csv and PlayerAwardShares.csv files. PlayerPerGame.csv includes per game stats for every player to have played in the NBA, with a unique entry for every season. PlayerAwardShares contains the results of voting for numerous end-of-season awards, such as MVP, Defensive Player of the Year, and more.   

The first thing we need to do is import the libraries we'll be using throughout the project:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, pearsonr
import statsmodels.api as sm
```
Next, we have to use pandas to convert the two csv files into dataframes.
```python
player_per_game = pd.read_csv('Player_per_game.csv')
player_award_shares = pd.read_csv('Player_award_shares.csv')
```
To prepare these files for analysis, there are a few things we have to do. Firstly, although the data goes back to 1947, I want to only look at data from the year 2000 to the present, since the NBA has changed a lot over the years and statistics from long ago won't have as much relevance when trying to predict things in the future.
```python
player_per_game = player_per_game[player_per_game['season'] >= 2000]
player_award_shares = player_award_shares[player_award_shares['season'] >= 2000]
```
Now we'll start working on the player_per_game dataframe. The first thing we'll do is remove some of the columns, since they are irrelevant for our purposes.
```python
player_per_game = player_per_game.drop(columns=['birth_year', 'lg'])
```
Some players are present in the dataset multiple times for the same season due to being traded mid-season. In these cases, there is an entry for the player's performance on each team and an additional entry for their total stats for the whole season. We only care about their total stats, so we'll remove the entries for the individual teams.
```python
multi_team_players = player_per_game[player_per_game.duplicated(subset=['player', 'season'], keep=False)]['player'].unique()
player_per_game = player_per_game[(player_per_game['tm'] == 'TOT') | (~player_per_game['player'].isin(multi_team_players))]
player_per_game = player_per_game.reset_index(drop=True)
```
Some players play multiple positions over the course of the season. In these cases, they have multiple positions listed in the pos column (C-PF for example). To make things easier, we only care about the player's primary position, so we'll remove the secondary position in such cases.
```python
player_per_game['pos'] = player_per_game['pos'].str.split('-').str[0]
```
Finally, we're going to create a new column called total_contributions_per_game, which is the sum of a player's points, rebounds, and assists per game. This is to have a more complete measure of a player's all-around performance, which we will use later on.
```python
player_per_game['total_contributions_per_game'] = (player_per_game['pts_per_game'] + player_per_game['trb_per_game'] + player_per_game['ast_per_game'])
```
Now, we'll move on to the player_award_shares dataframe.
## Exploratory Data Analysis

## Primary Analysis

## Visualization

## Insights and Conclusions



