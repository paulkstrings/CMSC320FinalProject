# Uncovering Trends and Patterns in NBA Player Statistics
### Fall 2024 Data Science Project
### Paul Kettlestrings
I worked on this project by myself, and thus was the sole contributer for every part of the project.

## Introduction
For this project, I chose to look at NBA statistics. I love watching basketball, and I wanted to see if some assumptions I had about the game would be backed up by the data. I looked at the effect of position and experience on player performance, as well as how a player's minutes impacts their points per game. I also wanted to use the data to make some predictions about the upcoming season, such as who will win the MVP award and the Most Improved Player award.

## Data Curation
The dataset I am using for this project is [NBA Stats (1947-present)](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats/data), found on kaggle.com. The dataset includes numerous csv files with player stats, team stats, award winners, and more, going all the way back to the founding of the NBA. We will be using the PlayerPerGame.csv and PlayerAwardShares.csv files. PlayerPerGame.csv includes per game stats for every player to have played in the NBA, with a unique entry for every season. PlayerAwardShares contains the results of voting for numerous end-of-season awards, such as MVP, Defensive Player of the Year, and more.   

The first thing we need to do is import the libraries we'll be using throughout the project:
```import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, pearsonr
import statsmodels.api as sm```

To prepare these files for analysis, there are a few things we have to do. Firstly, although the data goes back to 1947, I want to only look at data from the year 2000 to the present, since the NBA has changed a lot over the years and statistics from long ago won't have as much relevance when trying to predict things in the future.

## Exploratory Data Analysis

## Primary Analysis

## Visualization

## Insights and Conclusions



