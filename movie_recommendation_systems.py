# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

#import required packages
import pandas as pd
import statistics as st
import scipy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import datetime as dt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


#####import scores dataset from Movielens data 
data_scores = pd.read_csv('C:\\Users\\surbhi36152\\Downloads\\archive\\genome_scores.csv')

###get summaries
data_scores.head(10) ###what is relevance?
scores_summary=data_scores.describe()
len(data_scores['movieId'].unique().tolist())

###search for missing values
data_scores.isnull().any()

#####import rating dataset from Movielens data 
data_rating = pd.read_csv('C:\\Users\\surbhi36152\\Downloads\\archive\\rating.csv')
data_rating.head(10)
###get summaries
rating_summary=data_rating.describe()#not much poorly rated movie, movies is almost same as no.of users
len(data_rating['movieId'].unique().tolist())

###search for missing values
data_rating.isnull().any()


##import tags
data_tag = pd.read_csv('C:\\Users\\surbhi36152\\Downloads\\archive\\tag.csv')
len(data_tag['movieId'].unique().tolist())
data_tag.isnull().any()###nulll in tag
data_tag['tag'].isna().sum()
data_tag['tag'].value_counts().shape

##remove null tags
data_tag=data_tag.dropna()
data_tag['tag'].isna().sum()


##import movies, titles
data_movie = pd.read_csv('C:\\Users\\surbhi36152\\Downloads\\archive\\movie.csv')
data_movie.head()
len(data_movie['movieId'].unique().tolist())
data_movie.isnull().any()


###import imdbids ---not relevant data
data_link = pd.read_csv('C:\\Users\\surbhi36152\\Downloads\\archive\\link.csv')
len(data_link['movieId'].unique().tolist())
data_link.isnull().any()###nulll in tmdbId
data_link['tmdbId'].isna().sum()  ##no need of removal, this id not relevant

data_genome_tags = pd.read_csv('C:\\Users\\surbhi36152\\Downloads\\archive\\genome_tags.csv')
data_genome_tags.isnull().any()###nulll in tmdbId

##combine tagged datasets and find relevant tags like top 250 imdb and oscar 
data_tag_name=data_scores.merge(data_genome_tags,on=['tagId'],how='inner') ##not useful (only imdbID)
data_tag_name=data_tag_name.merge(data_movie,on=['movieId'],how='inner')
store=data_tag_name[(data_tag_name['tag']=='imdb top 250') & (data_tag_name['relevance']>0.9)].sort_values(by='relevance',ascending=False)

data_tag_name[(data_tag_name['tag']=='oscar') & (data_tag_name['relevance']>0.9)].sort_values(by='relevance',ascending=False)

most_tagged= pd.DataFrame(data_tag_name.groupby('tag').size().sort_values(ascending=False)).reset_index()

####merge imdbid with movie title
data_movie_link_new=data_movie.merge(data_link,on=['movieId']) ##not useful (only imdbID)

###merge movie with user ratings
data_rating_genre = data_rating.merge(data_movie,on='movieId', how='left')
data_rating_genre.columns

###check for nulls
data_rating_genre.isnull().any()

##get year out of title  ###very slow, can make faster
data_rating_genre['year'] =data_rating_genre['title'].str.extract('.*\((.*)\).*',expand = False)
data_rating_genre.iloc[:,3:].head(5)

# #######recent released movies


###########most viewed movies#############s
most_rated = pd.DataFrame(data_rating_genre.groupby('title').size().sort_values(ascending=False)).reset_index()
most_rated.columns=['title','Total Ratings']
most_rated.sum()

plt.figure(figsize = (20,10))
plt.ylabel("Total Ratings", fontsize = 12, labelpad = 0)
plt.xticks(rotation=60)
plt.title('Most rated movies in Imdb Top 250', fontsize=18)
ax = sns.barplot(x = 'title', y = 'Total Ratings', data = most_rated.iloc[1:10,:],  linewidth = 1.5, edgecolor = 'black')
plt.show()


 ###get top rated movies with minimum of 500 ratings
relevant_movies=most_rated[most_rated['Total Ratings']>500]['title']
Average_ratings= pd.DataFrame(data_rating_genre[data_rating_genre['title'].isin(list(relevant_movies))].groupby('title')['rating'].mean().sort_values(ascending=False)).reset_index()
Average_ratings.head(10)
Average_ratings.columns=['title','Average_ratings']

title_avg_rating_no_users = most_rated.merge(Average_ratings,on='title', how='inner')
weight_avg=pd.DataFrame(title_avg_rating_no_users.sort_values(by=['Total Ratings'],ascending=[False])).reset_index()


##plot avg ratings
plt.figure(figsize =(10, 4)) 
plt.ylabel('No. of movies', fontsize = 12) 
plt.xlabel('Avg Rating', fontsize = 12)  
title_avg_rating_no_users['Average_ratings'].hist(bins = 70)

##plot total ratings
plt.figure(figsize =(10, 4)) 
plt.ylabel('No. of movies', fontsize = 12) 
plt.xlabel('Total Ratings', fontsize = 12)  
title_avg_rating_no_users['Total Ratings'].hist(bins = 70)


# #######Popularity of genre ######################

movies['(no genres listed)'].sum()
movies.columns
movies[movies['(no genres listed)']==1].iloc[:,1:4].head(1)

genre_freq=movies.iloc[:,3:].sum(axis=0).sort_values(ascending=False) 
genre_freq=pd.DataFrame(genre_freq).reset_index()
genre_freq.columns=['Genre','No of occurences']


plt.figure(figsize = (20,10))
plt.ylabel("No. of occurences", fontsize = 12, labelpad = 0)
ax = sns.barplot(x = 'Genre', y = 'No of occurences', data = genre_freq,  linewidth = 1.5, edgecolor = 'black')
plt.show()



# #########FIRST METHOD#############################

##correl based on similar set of genres with highest ratings, simple recommendation
###filter data as 20Mn not able to process
data_rating_genre_filtered= data_rating_genre.iloc[:1000000,:]

##transform to get movies in columns
movie_rater = data_rating_genre_filtered.pivot_table(index='userId',columns='title',values='rating')
movie_rater.head()
movie_rater=movie_rater.fillna(0)

###get correlation from corrwith(pearson) takes times
correlations = movie_rater.corrwith(movie_rater['Toy Story (1995)'])
correlations.head()

recommendation = pd.DataFrame(correlations,columns=['Correlation']).reset_index()
recommendation.dropna(inplace=True)
recommendation = recommendation.join(most_rated['Total Ratings'])

recommendation.head()

recc = recommendation[recommendation['Total Ratings']>500].sort_values('Correlation',ascending=False).reset_index()
recc = recc.merge(data_movie,on='title', how='left').sort_values(by=['Correlation', 'Total Ratings'],ascending=[False, False])
#recc.sort_values(by=['Correlation', 'Total Ratings'],ascending=[False, False]).head(10)


# ###limitations, even if anyone genre matches for example 

# #Method 2

# #############content based filtering#####################

# ###genre based on genre movie neighbourhood based on cosine distance##########

movies = data_movie.join(data_movie.genres.str.get_dummies("|"))

# compute the cosine similarity 
cos_sim = cosine_similarity(movies.iloc[:,3:])


sabrina_top5 = np.argsort(cos_sim [6])[-5:][::-1]
movies[movies.index.isin(sabrina_top5)]['genres']

# #limitations : doesn't cater to user tailored suggestions by incorporating ratings

# Method 3

# ############collaborative filtering################

# ##user preferance based filtering#######

avg_rating = data_rating['rating'].mean() # calculate mean rating

data_rating_filtered= data_rating.iloc[:1000000,:]
preferance_matrix = data_rating_filtered[['userId', 'movieId', 'rating']].pivot(index='userId', columns='movieId', values='rating')

preferance_matrix = preferance_matrix - avg_rating # subtract avg rating i.e. adjust ratings

item_avg_rating = preferance_matrix.mean(axis=0)  ###get column avg
preferance_matrix = preferance_matrix - item_avg_rating # item avg made adjustment

user_avg_rating = preferance_matrix.mean(axis=1)  ##get row avg
preferance_matrix = preferance_matrix - user_avg_rating# item avg made adjustment

mat_avg=preferance_matrix.fillna(0) + user_avg_rating + item_avg_rating + avg_rating

mat_avg = preferance_matrix.values

##for user 1
np.nansum((mat_avg - mat_avg[700,:])**2,axis=1)[1:].argmin() # returns 99
# check it:
arr=np.nansum(mat_avg[4] - mat_avg[700]) # returns 0.0

np.where(~np.isnan(mat_avg[4]) & np.isnan(mat_avg[700]) == True)
mat_avg[4][[1,   10,   16]]##11th movie recommended



