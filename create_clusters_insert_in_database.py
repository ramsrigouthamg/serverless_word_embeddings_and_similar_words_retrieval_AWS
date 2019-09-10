
import traceback
import time
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

# Modifiable parameters
# Batch size for training kmeans - minibatchkmeans
batchsize = 10
no_of_clusters = 4
word2vec_filename = 'sample_word2vec.vec'
debug = True

kmeans = MiniBatchKMeans(n_clusters=no_of_clusters,
                         init='k-means++',
                         max_iter=300,
                         random_state=0,
                         reassignment_ratio = 0.01,
                         batch_size=batchsize)

# Aurora serverless database initialisation
from pymysql import connections
DBHOST = ""
DBPORT = int(3306)
DBUSER = "admin"
DBPASSWORD = ""
DATABASE = "wordvectors"
dbcon = connections.Connection(host = DBHOST, port=DBPORT,user=DBUSER,password=DBPASSWORD,db=DATABASE)
cursor = dbcon.cursor()



def load_vectors_from_file(fname):
    X = []
    wordnames = []
    with open(fname, encoding='utf-8') as f:
        first_line = f.readline()
        dimension = int(first_line.split()[-1])
        if debug == True:
            print("first line word2vec ",first_line)
        firstNlines = f.readlines()
        for line in firstNlines:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                assert (len(values[1:]) == dimension)
                X.append(coefs)
                wordnames.append(word)
            except:
                pass

    return dimension,wordnames,X

def get_iterator_indexes(a_size,batch_size):
    loop_indexes = []
    for i in range(0, a_size, batch_size):
        start = i
        end = min(i + batchsize, a_size)
        loop_indexes.append((start,end))
    return  loop_indexes


def train_kmeans(indexes,X,kmeans_obj):
    for index in indexes:
        start = index[0]
        end = index[1]
        tempX = X[start:end]
        tempX = np.array(tempX)
        kmeans_obj.partial_fit(tempX)
    return kmeans_obj

def kmeans_print_prediction(kmeans_obj,words_array,X,indexes):
    count_cluster = {}
    cluster_index_and_words = {}
    for i in range(kmeans_obj.n_clusters):
        cluster_index_and_words[i]=[]
    print ("cluster_index_and_words ",cluster_index_and_words)
    for index in indexes:
        start = index[0]
        end = index[1]
        tempX = X[start:end]
        tempX = np.array(tempX)
        cluster_idx = kmeans.predict(tempX)
        print ("Words ",words_array[start:end])
        print ("Cluster indexes ",cluster_idx)
        for word,idx in zip(words_array[start:end], cluster_idx):
            print ("word ",word," idx ",idx)
            cluster_index_and_words[idx].append(word)
            if idx in count_cluster:
                count_cluster[idx] = count_cluster[idx] + 1
            else:
                count_cluster[idx] = 1
    cluster_counts = sorted(count_cluster.items(), key=lambda x: x[1])
    return cluster_counts,cluster_index_and_words

def get_closest_cluster_index(clusters_count,kmeans_obj):
    cluster_centers = kmeans_obj.cluster_centers_
    print ("cluster centers ",cluster_centers)
    clusters_with_low_count = []
    threshold_count = 4
    for each in clusters_count:
        if each[1] <= threshold_count:
            clusters_with_low_count.append(each[0])
    print ("clusters_with_low_count ",clusters_with_low_count)
    search_cluster=[]
    for index,val in enumerate(cluster_centers):
        if index not in clusters_with_low_count:
            search_cluster.append(np.asarray(val, dtype='float32'))
    merge_clusters={}
    for clust in clusters_with_low_count:
        search_index_array = cluster_centers[clust]
        neigh = NearestNeighbors(metric='cosine')
        neigh.fit(search_cluster)
        closest = neigh.kneighbors([search_index_array], 1, return_distance=False)[0][0]
        merge_clusters[clust]=closest
        print ("search_index_array ",search_index_array)
        print ("closest ",closest)

    return merge_clusters

def merge_clusters (cluster_word_ids,merge_clusters_ids):
    merged_cluster_word_ids = cluster_word_ids
    for val in merge_clusters_ids:
        cluster_to_merge = val
        cluster_to_merge_into = merge_clusters_ids[val]
        merged_cluster_word_ids[cluster_to_merge_into].extend(merged_cluster_word_ids[cluster_to_merge])
        del merged_cluster_word_ids[cluster_to_merge]

    return merged_cluster_word_ids


dimensions, words, word_vectors = load_vectors_from_file(word2vec_filename)
array_size = len(word_vectors)
loop_indexes =   get_iterator_indexes(array_size,batchsize)


kmeans = train_kmeans(loop_indexes,word_vectors,kmeans)

cluster_counts,cluster_word_ids = kmeans_print_prediction(kmeans,words, word_vectors,loop_indexes)

merge_clusters_ids = get_closest_cluster_index(cluster_counts,kmeans)

final_dict = merge_clusters(cluster_word_ids,merge_clusters_ids)


print ("final_dict ",final_dict)

for id in final_dict:
    database_upload_payload =[]
    sql_pixabay = """INSERT INTO wordvectors(word,cluster_id,vector)
             VALUES (%s, %s, %s) on duplicate key update
             word=VALUES(word),cluster_id=VALUES(cluster_id),vector=VALUES(vector)"""
    for each_word in final_dict[id]:
        index = words.index(each_word)
        print ("word ",each_word," cluster id: ",id, " Wordvector ",word_vectors[index])
        database_upload_payload.append((each_word,id,str(word_vectors[index])))

    try:
        # Execute the SQL command
        cursor.executemany(sql_pixabay, database_upload_payload)
        # Commit your changes in the database
        dbcon.commit()
    except:
        # Rollback in case there is any error
        print("Exception Occured")
        traceback.print_exc()
        dbcon.rollback()






