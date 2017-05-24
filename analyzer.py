import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



#first step of data exploration- read in the data and output basic summary stats

#read in the CSV
games_data = pd.read_csv("games.csv")

#print the columns which describe the various attributes we have for each game (row) of the dataset
print(games_data.columns)

#print the number of columns & rows of the dataset i.e. the shape
print(games_data.shape)
#81,312 games (rows) and 20 attributes (columns)

#problem- can we predict the average score humans will give a new, yet-to-be released board game?
#we can use the given dataset to train a model and make this prediction

#let's plot all of the average ratings in the given dataset. Hence let's plot the "average_ratings" column
plt.hist(games_data["average_rating"])
plt.show()


#after showing the plot, we find an anomaly that there are an unusually large number of games with 0 rating
#let's find out why

#print out the games (rows) that has a "0" rating 
print(games_data[games_data["average_rating"]==0])
#so we see there are 24,380 such games with 0 ratings

#now let's see the other attributes of these games  other than just its "average_rating". 
#let's print out all the attributes of the games that have 0 average ratings, hence [:]
#by seeing the other attributes, we can possibly detect why there are so many 0 average_ratings. 
#to do this, make a new dataframe
print(games_data[games_data["average_rating"]==0].iloc[:])
#this shows us that for this game, and presumably other games with 0 ratings, there are no customer reviews

#to confirm this theory, let's print all the attributes of all games which has a non-zero rating
print(games_data[games_data["average_rating"]>0].iloc[0])
#as theorized, all of these games that have a non-zero rating have thousands of customer reviews

#to get rid of the extraneous data, let's remove all the games with 0 reviews. games_data now only contains games with > 0 user reviews
games_data = games_data[games_data["users_rated"]>0]

#side note, many ML algorithms don't work with data that has missing values. so let's remove any rows (Games) that have missing attributes
games_data = games_data.dropna(axis=0)

#so now we see that there may be different "sets" of games i.e. those without reviews, those with high reviews, etc
#so, use k-means clustering to group together similar rows (i.e. games)

kmeans_training_model = KMeans(n_clusters = 5, random_state = 1)

#we only want the attributes (columns) that have numeric data, not text data
cols = games_data._get_numeric_data()

#fit the training model
kmeans_training_model.fit(cols)

#get the cluster assignment labels
labels = kmeans_training_model.labels_

#before we plot the clustering results, we need to perform dimensionality reduction using PCA.  this is because
#currently we have too many attributes (dimensions/columns). its hard to make a comprehensible graph that plots all of these 
from sklearn.decomposition import PCA
pca_2 = PCA(2)

#fit the model
cols_to_be_plotted = pca_2.fit_transform(cols)

#scatter plot of each game
plt.scatter(x=cols_to_be_plotted[:,0], y=cols_to_be_plotted[:,1], c=labels)

plt.show()	

#find the attribute which has the strongest corrleation with "average_rating" so that it can be used to 
#help predict average_rating
games_data.corr()["average_rating"]




