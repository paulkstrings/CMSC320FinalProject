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
Now, we'll move on to the player_award_shares dataframe. This one is a bit more complicated to work with. If a player is nominated for multiple awards in the same season, such as MVP and DPOY, those two nominations are given separate entries. In order to be able to eventually merge this with the player_per_game dataframe, we want to have only one entry per player per season, so we'll have to combine them somehow.   
First, we'll create a list containing all of the awards we have data for and create columns for each of the awards.
```python
awards = ['clutch_poy', 'dpoy', 'mip', 'nba mvp', 'nba roy', 'smoy']
award_columns = []
for award in awards:
  award_columns.append(f'{award.lower().replace(" ", "_")}_first')
  award_columns.append(f'{award.lower().replace(" ", "_")}_pts_won')
  award_columns.append(f'{award.lower().replace(" ", "_")}_share')
  award_columns.append(f'{award.lower().replace(" ", "_")}_winner')
```
Next, pivot the table to add the award columns, filling in any empty spots with the appropriate value:
```python
award_pivot = player_award_shares.pivot_table(
    index=['player_id', 'seas_id', 'season', 'player', 'age', 'tm'],
    columns='award',
    values=['first', 'pts_won', 'share', 'winner'],
    aggfunc={'first' : 'sum', 'pts_won' : 'sum', 'share' : 'sum', 'winner' : 'any'}, 
    fill_value=0
)

award_pivot.columns = [f'{col[1].lower().replace(" ", "_")}_{col[0]}' for col in award_pivot.columns]
award_pivot = award_pivot.reset_index()

for award in awards:
    winner_col = f'{award.lower().replace(" ", "_")}_winner'
    award_pivot[winner_col] = award_pivot[winner_col].astype(bool)
```
Finally, we can merge the player_per_game and player_award_shares dataframes together into a single dataframe that will contain all the information we need. We'll fill in the award data for players who weren't nominated for awards with the appropriate values.
```python
final_merged_data = pd.merge(player_per_game, award_pivot, on=['player_id', 'seas_id'], how='left')

award_share_columns = [f'{award.lower().replace(" ", "_")}_pts_won' for award in awards] + \
                      [f'{award.lower().replace(" ", "_")}_share' for award in awards]
final_merged_data[award_share_columns] = final_merged_data[award_share_columns].fillna(0)

winner_columns = [f'{award.lower().replace(" ", "_")}_winner' for award in awards]
final_merged_data[winner_columns] = final_merged_data[winner_columns].fillna(False)
```

Our dataframe is ready for work, and we can now move on to the next step.

## Exploratory Data Analysis
Now that we have our dataframe ready to go, we can perform some basic analysis to get a sense for the data. To do this, I decided to test some hypotheses I had about certain factors that could impact a player's performance.
### Does a player's position impact their rebounding numbers?
Intuition would tell you that centers and power forwards, who are generally the tallest players on the court, would get more rebounds per game than players in other positions. Let's see if that's really true!   
We'll use an ANOVA test for this since we have multiple categories that we will be comparing to see if there is any significant difference between them.   
We can use seaborn to make a box plot with position on the x-axis and rebounds per game on the y-axis.
```python
plt.figure()
sns.boxplot(data=final_merged_data, x='pos', y='trb_per_game')
plt.title("Rebounds per Game by Position")
plt.xlabel("Position")
plt.ylabel("Rebounds per Game")
plt.grid(True)
plt.show()

grouped_data = [group['trb_per_game'].values for name, group in final_merged_data.groupby('pos')]

f_stat, p_value = f_oneway(*grouped_data)

print(f"F-statistic: {f_stat}, p-value: {p_value}")
alpha = 0.05
if p_value < alpha:
    print("There is a significant difference in rebounds per game between positions.")
else:
    print("There is no significant difference in rebounds per game between positions.")
```
This results in the following plot:   
![Rebounds per Game Plot](RPG_Plot.jpg)
As we can see, centers and power forwards have the highest average rebounds per game, and the ANOVA test determined that there is a significant difference in rebounds per game between positions.

### Does playing more minutes lead to scoring more points?
Common sense would suggest that players who play more would score more points, both since they have more opportunities and because they are generally better than those who only play a small amount. Let's check to be sure, though.   
We'll use a Pearson correlation test here, since we are measuring the correlation between two variables, minutes per game and points per game, and we would expect them to be linearly correlated. We'll create a scatter plot with every season and plot the line of best fit to see if it is linear.
```python
corr, p_value = pearsonr(final_merged_data['mp_per_game'], final_merged_data['pts_per_game'])

plt.figure(figsize=(12, 6))
sns.regplot(x='mp_per_game', y='pts_per_game', data=final_merged_data, scatter_kws={'alpha':0.5})
plt.title('Correlation Between Minutes Played and Points Per Game')
plt.xlabel('Minutes Per Game')
plt.ylabel('Points Per Game')
plt.grid(True)
plt.text(0.1, 0.9, f'Correlation coefficient: {corr:.2f}\np-value: {p_value:.2g}',
         horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
plt.show()

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: significant correlation between minutes played and points scored.")
else:
    print("Fail to reject the null hypothesis: no significant correlation.")
```
![Points vs Minutes Plot](Points_Minutes.jpg)
As we can see, there is a clear linear correlation between minutes per game and points per game, as we would expect. 

### Does more experience in the NBA lead to increased performance?
Generally, players who last in the NBA for many years are the most talented players who are able to sustain a high level of play even as they age. Therefore, it would make sense that the longer you've been in the league, the better your performance would be. To determine if this is the case, we'll use the total_contributions_per_game stat we created earlier.






## Primary Analysis

## Visualization

## Insights and Conclusions



