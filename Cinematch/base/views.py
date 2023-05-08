# Import necessary libraries
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import JsonResponse
from django.http import HttpResponse

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import json

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import firestore
import requests
from collections import OrderedDict

from fuzzywuzzy import fuzz
from fuzzywuzzy import process


# Load Firebase credentials from the given file
cred = credentials.Certificate("cinematch-97725-firebase-adminsdk-79jxv-732c8c53e3.json")
# Initialize the Firebase application with the provided credentials
firebase_admin.initialize_app(cred)


IDArray = []
NameArray = []
Recommended = []

# View function for the main endpoint
def MainView(request, userid):
    # Connect to the Firestore database
    db = firestore.client()
    # Get the document reference for the specified user ID
    doc_ref = db.collection('users').document(userid)
    # Get the snapshot of the document
    doc_snapshot = doc_ref.get()
    
    # Check if the document exists
    if doc_snapshot.exists:
        # Get the 'movies_liked' field from the document
        movies_liked = doc_snapshot.get('movies_liked')
        # Get the 'movies_in_watchlist' field from the document
        movies_in_watchlist = doc_snapshot.get('movies_in_watchlist')
        # Get the 'comments' field from the document
        comments = doc_snapshot.get('comments')
        # Get the 'rated_movies' field from the document
        rated_movies = doc_snapshot.get('rated_movies')

        # Check if 'movies_liked' field exists and has data
        if movies_liked:
            response_data = {'movies': movies_liked}
            # Convert the movies_liked dictionary into a list of movie IDs
            movie_ids = list(movies_liked)
            # Add the movie IDs to the IDArray
            IDArray.extend(movie_ids)

        # Check if 'movies_in_watchlist' field exists and has data
        if movies_in_watchlist:
            response_data = {'movies': movies_in_watchlist}
            # Convert the movies_in_watchlist dictionary into a list of movie IDs
            movie_ids = list(movies_in_watchlist)
            # Add the movie IDs to the IDArray
            IDArray.extend(movie_ids)
        
        # Check if 'comments' field exists and has data
        if comments:
            # Iterate through the comments dictionary
            for movie_id, comment in comments.items():
                # Check if the CommentAnalyzer function returns [5] for the comment
                if(CommentAnalyzer(comment))==[5]:
                    # Add the movie ID to the IDArray
                    IDArray.append(movie_id)

        # Check if 'rated_movies' field exists and has data
        if rated_movies:
            # Iterate through the rated_movies dictionary
            for movie_id, rate in rated_movies.items():
                # Check if the rating is "5"
                if rate=="5":
                    # Add the movie ID to the IDArray
                    IDArray.append(movie_id)

        # Convert the movie IDs to integers and remove duplicates
        newIDArray = [int(x) for x in IDArray]
        removedRepeatedItems = list(set(newIDArray))

        # Fetch movie names from the API based on the movie IDs
        for i in removedRepeatedItems:
            url = f"https://api.themoviedb.org/3/movie/{i}?api_key=3faff41c09329d5872ba9f5823910a23"
            response = requests.get(url)
            data = response.json()
            movie_name = data.get('title')
            # Add the movie name to the NameArray
            NameArray.append(movie_name)

        # Load the movies CSV file
        movies = pd.read_csv('ml-latest-small/movies.csv')
        # Get the list of titles from the 'movies' DataFrame
        csv_titles = movies['title'].tolist()

        # Preprocess the CSV titles
        preprocessed_csv_titles = [title for title in csv_titles]

        # Function to find the best match for a given movie title
        def find_best_match(movie_title, csv_titles):
            # Find the title with the highest fuzzy ratio match
            best_match = max(csv_titles, key=lambda x: fuzz.ratio(movie_title, x))
            return best_match

        # Iterate through the movie names in NameArray
        for i in NameArray:
            # Find the best match for each movie name in the CSV titles
            best_match = find_best_match(i, preprocessed_csv_titles)
            # Get the recommended movies based on the best match
            Recommended.append(Recommender(best_match))

        # Flatten the Recommended list of lists into a single list
        flattened_list = [item for sublist in zip(*Recommended) for item in sublist]

        # Prepare the response data with the flattened list of recommended movies
        response_data = {'list_data': flattened_list}
        return JsonResponse(response_data)

    # If the document does not exist for the given user ID, return an HTTP response
    else:
        return HttpResponse("Document does not exist.")

# Define the view function that will be called when the API endpoint is accessed
def Recommender(movie):
    # Load the movies and ratings datasets
    movies = pd.read_csv('ml-latest-small/movies.csv')
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    
    # Merge the two datasets on the 'movieId' column
    data = pd.merge(ratings, movies, on='movieId')
    
    # Create a pivot table with 'movieId' as the index, 'userId' as the columns, and 'rating' as the values.
    # Fill missing values with 0
    pivot_table = data.pivot_table(index='movieId', columns='userId', values='rating').fillna(0)
    
    # Convert the pivot table to a sparse matrix to save memory
    sparse_matrix = csr_matrix(pivot_table.values)
    
    # Define the nearest neighbors model with 'cosine' similarity metric and 'brute' algorithm
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    
    # Fit the model to the sparse matrix
    model.fit(sparse_matrix)
    
    # Get the movieId of the input movie title
    toy_story_id = movies[movies['title'] == movie]['movieId'].iloc[0]
    
    # Get the index of the input movie in the pivot table
    toy_story_index = pivot_table.index.get_loc(toy_story_id)
    
    # Find the 10 nearest neighbors to the input movie based on user ratings
    # Get the distances and indices of the neighbors
    distances, indices = model.kneighbors(pivot_table.iloc[toy_story_index, :].values.reshape(1, -1), n_neighbors=11)
    
    similar_movies = []

    for index in indices.flatten()[1:]:
        movie_title = movies[movies['movieId'] == pivot_table.iloc[index].name]['title'].iloc[0]
        similar_movies.append(movie_title)
    #Recommended.append(similar_movies)
    return similar_movies

def CommentAnalyzer(my_comment):
    # set of stop words to be removed from the comments
    stop_words = set(stopwords.words('english'))
    # word lemmatizer to normalize the words
    lemmatizer = WordNetLemmatizer()

    # function to preprocess the comments
    def preprocess_comment(comment):
        # tokenize and lowercase the comment
        words = word_tokenize(comment.lower())
        # remove stop words from the comment
        words = [word for word in words if word not in stop_words]
        # normalize words in the comment
        words = [lemmatizer.lemmatize(word) for word in words]
        # join the words back into a string
        comment = ' '.join(words)
        return comment
    
    # function to convert sentiment to rating
    def sentiment_to_rating(sentiment):
        if sentiment > 0:
            return 5
        elif sentiment == 0:
            return 3
        else:
            return 1

    # list of comments to analyze
    comments = [my_comment]
    # preprocess the comments
    preprocessed_comments = [preprocess_comment(comment) for comment in comments]
    # calculate sentiment polarity for each comment using TextBlob
    sentiments = [TextBlob(comment).sentiment.polarity for comment in preprocessed_comments]
    # convert sentiment polarity to rating using sentiment_to_rating function
    ratings_predicted = [sentiment_to_rating(sentiment) for sentiment in sentiments]
    # return the predicted ratings as a JSON response
    return ratings_predicted