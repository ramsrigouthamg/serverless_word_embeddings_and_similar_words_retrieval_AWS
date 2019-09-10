# NLP - Serverless deployment of word embeddings and retrieving most similar words using Kmeans, AWS Aurora Serverless and Lambda
 
Refer to this [Medium Post](https://medium.com/@ramsrigouthamg/nlp-serverless-deployment-of-word-embeddings-and-retrieving-most-similar-words-using-kmeans-aws-51f129297995?sk=5a0909fa1dc212653812de152c14d83a) for detailed explanation of this repository.

This repository contains all the code to place a word embeddings file (.vec) onto a database for running most_similar function in a serverless manner to retrieve the most similar words for a given query word.
 
The file **sample_word2vec.vec** is the word embeddings used as an example to show the functionality. It is similar in format to the real word2vec files loaded by gensim.

The file **create_clusters_insert_in_database.py** contains all the code to -
1) Load the word2vec file
2) Create clusters using SKlearn MinibatchKmeans.
3) Merge clusters with less than a few elements to a nearby cluster.
3) Upload the word, word embedding and cluster_id to an Amazon Aurora serverless database.

The file **lambda_getsimilarwords_deployment_package/aws_lambda_deployment_getsimilarwords.py** contains the code that is deployed on AWS lambda.
The lambda function can be called with input as:
{ "word": "cat" }

The returned output is in the following format :
{ "similar": [ "kitten","dog",...............] }

**Note:** The Lambda functions requires pymysql to connect to database and numpy to run cosine similarity in the same cluster and find the closest word vector to a given word vector. Pymysql is already added to the lambda package but numpy is obtained by adding predefined **lambda layer** for Sklearn and Python 3.6 already provided by AWS.




 

