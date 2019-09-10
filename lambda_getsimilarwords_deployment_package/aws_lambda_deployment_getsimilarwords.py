
import numpy as np
from numpy import dot
from numpy.linalg import norm

# Aurora serverless database initialisation
from pymysql import connections
DBHOST = ""
DBPORT = int(3306)
DBUSER = "admin"
DBPASSWORD = ""
DATABASE = "wordvectors"
dbcon = connections.Connection(host = DBHOST, port=DBPORT,user=DBUSER,password=DBPASSWORD,db=DATABASE)
cursor = dbcon.cursor()

def get_index_and_vector(word):
    sql_getcluster_index_and_wordvector = 'SELECT cluster_id,vector FROM wordvectors WHERE word = (%s)'
    cursor.execute(sql_getcluster_index_and_wordvector, word)
    # Fetch all the rows in a list of lists.
    cluster_index, wordvector = cursor.fetchall()[0]
    wordvector = wordvector.replace("[","").replace("]","")
    wordvector = wordvector.strip().split()
    wordvector = np.asarray(wordvector, dtype='float32')
    return cluster_index, wordvector

def retrieve_similar(clusterid,originalwordvec,q_word):
    sql_get_similar_words = 'SELECT word,vector FROM wordvectors WHERE cluster_id = (%s) AND word!=(%s) '
    cursor.execute(sql_get_similar_words, (clusterid,q_word))
    all_vals = cursor.fetchall()
    word_similarity_index=[]
    for word, wordvector in all_vals:
        wordvector = wordvector.replace("[", "").replace("]", "")
        wordvector = wordvector.strip().split()
        wordvector = np.asarray(wordvector, dtype='float32')
        cos_sim = dot(originalwordvec, wordvector) / (norm(wordvector) * norm(wordvector))
        word_similarity_index.append((word,cos_sim))

    sorted_vals = sorted(word_similarity_index, key=lambda x: x[1],reverse= True)
    output =[]
    for a,_ in sorted_vals:
        output.append(a)
    return output

def get_similar_words(q_word):
    cluster_index, wordvec = get_index_and_vector(q_word)
    similar = retrieve_similar(cluster_index,wordvec,q_word)
    return similar

def getsimilarwords(event, context):
    query_word = event['word']
    similar_words = get_similar_words(query_word)
    print("similar words ", similar_words)
    return {"similar":similar_words}



