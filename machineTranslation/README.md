# NLP - Naive Machine Translation & LSH


[epf_nlp/machineTranslation at main · GabrielTruong/epf_nlp](https://github.com/GabrielTruong/epf_nlp/tree/main/machineTranslation)

## Objectives

The goal of this assignment is to implement a naive machine translation that translates English words into French words. Then we will see how locally sensitive hashing works. 

## **The word embeddings data**

First we load the data, into dictionaries with words as keys and the 300 dimensional array that embeds the word as values. 

```python
en_embeddings_subset = pickle.load(open("en_embeddings.p", "rb"))
fr_embeddings_subset = pickle.load(open("fr_embeddings.p", "rb"))
```

Then we need to load the dictionaries that the English words to French words. 

```python
en_fr_train = get_dict('en-fr.train.txt')
en_fr_test = get_dict('en-fr.test.txt')
```

## Generate embedding and transform matrices

### Translating English dictionary to French using embeddings

We create a `get_matrices` function which takes the loaded data and returns the matrices `X` that contains the word embedding for an English word and `Y` that contains the word embedding for the French version of the word.

```python
def get_matrices(en_fr, french_vecs, english_vecs):
    """
    Input:
        en_fr: English to French dictionary
        french_vecs: French words to their corresponding word embeddings.
        english_vecs: English words to their corresponding word embeddings.
    Output: 
        X: a matrix where the columns are the English embeddings.
        Y: a matrix where the columns correspong to the French embeddings.
        R: the projection matrix that minimizes the F norm ||X R -Y||^2.
    """

    # X_l and Y_l are lists of the english and french word embeddings
    X_l = list() 
    Y_l = list() 

    # get the english words (the keys in the dictionary) and store in a set()
    english_set = set(english_vecs.keys())

    # get the french words (keys in the dictionary) and store in a set()
    french_set = set(french_vecs.keys())

    # store the french words that are part of the english-french dictionary (these are the values of the dictionary)
    french_words = set(en_fr.values())

    # loop through all english, french word pairs in the english french dictionary
    for en_word, fr_word in en_fr.items():

        # check that the french word has an embedding and that the english word has an embedding
        if fr_word in french_set and en_word in english_set:

            # get the english embedding
            en_vec = english_vecs[en_word]

            # get the french embedding
            fr_vec = french_vecs[fr_word]

            # add the english embedding to the list
            X_l.append(en_vec) 

            # add the french embedding to the list
            Y_l.append(fr_vec) 

    # stack the vectors of X_l into a matrix X
    X = np.vstack(X_l)

    # stack the vectors of Y_l into a matrix Y
    Y = np.vstack(Y_l)

    return X, Y
```

## Translations

Now that we have the `X` and `Y` matrices, we need to find a transformation matrix `R` that will multiple Engish embddings vector `e` to get a new embedding vector `f`. Then we find the nearest neighbors to `f` in French embeddings to get the most similar word to the English one.

### Loss Function

To find the right `R` matrix trasnformation, we need to minimize the following equation: $argmin = ||XR - Y||$.  At the end we have to compute the squared Frobenius norm of the difference and divide it by *𝑚.* 

```python
def compute_loss(X, Y, R):
    '''
    Inputs: 
        X: a matrix of dimension (m,n) where the columns are the English embeddings.
        Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
        R: a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings.
    Outputs:
        L: a matrix of dimension (m,n) - the value of the loss function for given X, Y and R.
    '''
    # m is the number of rows in X
    m, _ = X.shape
    
    # diff is XR - Y
    diff = np.matmul(X,R) - Y 

    # diff_squared is the element-wise square of the difference
    diff_squared = np.square(diff)

    # sum_diff_squared is the sum of the squared elements
    sum_diff_squared = np.sum(diff_squared)

    # loss i the sum_diff_squard divided by the number of examples (m)
    loss = sum_diff_squared/m
    return loss
```

### Gradient of the loss

Once we got our loss function, we will use the gradient descent algorithm to find the best `R` matrix. We thus need to create a function that will compute the gradient of our loss function. 

```python
def compute_gradient(X, Y, R):
    '''
    Inputs: 
        X: a matrix of dimension (m,n) where the columns are the English embeddings.
        Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
        R: a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings.
    Outputs:
        g: a matrix of dimension (n,n) - gradient of the loss function L for given X, Y and R.
    '''
    # m is the number of rows in X
    m, _ =  X.shape 

    # gradient is X^T(XR - Y) * 2/m
    gradient = 2/m*np.matmul(X.T, (np.matmul(X,R)-Y))  
    return gradient
```

### Gradient Descent

We now implement the `align_embeddings` function that will perform the gradient descent algorithm in order to find the optimum `R` matrix. 

```python
def align_embeddings(X, Y, train_steps=100, learning_rate=0.0003):
    '''
    Inputs:
        X: a matrix of dimension (m,n) where the columns are the English embeddings.
        Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
        train_steps: positive int - describes how many steps will gradient descent algorithm do.
        learning_rate: positive float - describes how big steps will  gradient descent algorithm do.
    Outputs:
        R: a matrix of dimension (n,n) - the projection matrix that minimizes the F norm ||X R -Y||^2
    '''
    np.random.seed(129)

    # the number of columns in X is the number of dimensions for a word vector (e.g. 300)
    # R is a square matrix with length equal to the number of dimensions in th  word embedding
    R = np.random.rand(X.shape[1], X.shape[1])

    for i in range(train_steps):
        if i % 25 == 0:
            print(f"loss at iteration {i} is: {compute_loss(X, Y, R):.4f}")
        # use the function that you defined to compute the gradient
        gradient = compute_gradient(X,Y,R)  

        # update R by subtracting the learning rate times gradient
        R -= learning_rate*gradient  
    return R
```

## Test the translation

### K-Nearest neighbor algorithm

The KNN algorithm will help find the closest french embedding to the exact English embeddings that got transformed by the `R` transformation matrix. The cosine similarity is used again, we explained the concept in the previous practical work. Check it if needed. 

```python
def nearest_neighbor(v, candidates, k=1):
    """
    Input:
      - v, the vector you are going find the nearest neighbor for
      - candidates: a set of vectors where we will find the neighbors
      - k: top k nearest neighbors to find
    Output:
      - k_idx: the indices of the top k closest vectors in sorted form
    """
    similarity_l = []

    # for each candidate vector...
    for row in candidates:
        # get the cosine similarity
        cos_similarity = cosine_similarity(v,row)  
        # append the similarity to the list
        similarity_l.append(cos_similarity)
         
        
    # sort the similarity list and get the indices of the sorted list
    sorted_ids = np.argsort(similarity_l) 

    # get the indices of the k most similar candidate vectors
    k_idx = sorted_ids[-k:]  # argsort return ordered from lower to highest
                                  # and the closests have the higher cos similarities
    return k_idx
```

### Accuracy of the translation

Now we will study the accuracy of our translation using the accuracy metric defined as followed: $\text{accuracy}=\frac{\#(\text{correct predictions})}{\#(\text{total predictions})}$. 

To do that, we will perform the following steps:

- Iterate over transformed English word embeddings and check if the closest French word vector belongs to French word that is the actual translation.
- Obtain an index of the closest French embedding by using `nearest_neighbor` (with argument `k=1`), and compare it to the index of the English embedding you have just transformed.
- Keep track of the number of times you get the correct translation.
- Calculate accuracy

```python
def test_vocabulary(X, Y, R):
    '''
    Input:
        X: a matrix where the columns are the English embeddings.
        Y: a matrix where the columns correspong to the French embeddings.
        R: the transform matrix which translates word embeddings from
        English to French word vector space.
    Output:
        accuracy: for the English to French capitals
    '''

    # The prediction is X times R
    pred =  np.matmul(X,R)  

    # initialize the number correct to zero
    num_correct =  0 

    # loop through each row in pred (each transformed embedding)
    for i in range(len(pred)):
        # get the index of the nearest neighbor of pred at row 'i'; also pass in the candidates in Y
        pred_idx = nearest_neighbor(pred[i],Y,k=1) 

        # if the index of the nearest neighbor equals the row of i... \
        if pred_idx == i:
            # increment the number correct by 1.
            num_correct += 1  

    # accuracy is the number correct divided by the number of rows in 'pred' (also number of rows in X)
    accuracy = num_correct/len(pred) 

    return accuracy
```

Using this first technique, we reach an accuracy of 0.557 which is great. We translated words from english to french without ever seeing them. 


## Locality Sensitive Hashing

In this part, we will try to find tweets that are similar to a given tweet using locality sensitive hashing.

### Get the document embeddings

Text documents are sequences of words. This sequence is ordered so that the text has a specific meaning. We can ignore the order of words to train more efficiently and still obtain an effective model. This method is called the Bag-of-words document model.

Preivously we got the word embeddings. Now we want to get document embeddings. We will do that by summing up the embeddings of each word in the document. If we cannot have the embedding of a word, we will ignore it. This is what the following fonction does:

```python 
def get_document_embedding(tweet, en_embeddings): 
    '''
    Input:
        - tweet: a string
        - en_embeddings: a dictionary of word embeddings
    Output:
        - doc_embedding: sum of all word embeddings in the tweet
    '''
    doc_embedding = np.zeros(300)

    # process the document into a list of words (process the tweet)
    processed_doc = process_tweet(tweet)
    for word in processed_doc:
        # add the word embedding to the running total for the document embedding
        doc_embedding += en_embeddings.get(word,0)  
    return doc_embedding
```

In the `get_document_vecs` function we will store all the tweet embeddings into a dictionary thanks to the `get_document_embedding` function.

```python
def get_document_vecs(all_docs, en_embeddings):
    '''
    Input:
        - all_docs: list of strings - all tweets in our dataset.
        - en_embeddings: dictionary with words as the keys and their embeddings as the values.
    Output:
        - document_vec_matrix: matrix of tweet embeddings.
        - ind2Doc_dict: dictionary with indices of tweets in vecs as keys and their embeddings as the values.
    '''

    # the dictionary's key is an index (integer) that identifies a specific tweet
    # the value is the document embedding for that document
    ind2Doc_dict = {} 

    # this is list that will store the document vectors
    document_vec_l = [] 

    for i, doc in enumerate(all_docs):

        # get the document embedding of the tweet
        doc_embedding = get_document_embedding(doc,en_embeddings) 

        # save the document embedding into the ind2Tweet dictionary at index i
        ind2Doc_dict[i] = doc_embedding 

        # append the document embedding to the list of document vectors
         
        document_vec_l.append(doc_embedding)


    # convert the list of document vectors into a 2D array (each row is a document vector)
    document_vec_matrix = np.vstack(document_vec_l)

    return document_vec_matrix, ind2Doc_dict
```

### Looking up the tweets

We now have a vector of dimension (10000,300) that respectively corresponds to the number of tweets and the embeddings. Now we can use the cosine similarity to find the tweet that will match the best to an input tweet.

```python
### Store the document vectors into a dictionary
my_tweet = 'i am sad'
process_tweet(my_tweet)
tweet_embedding = get_document_embedding(my_tweet, en_embeddings_subset)
idx = np.argmax(cosine_similarity(document_vecs, tweet_embedding))
print(all_tweets[idx]) 
#-> @zoeeylim sad sad sad kid :( it's ok I help you watch the match HAHAHAHAHA
```

### Finding the most similar tweets with LSH

We will use LSH to identify the most similar twwet. Instead of looking at all the 10000 vectors we will just search the nearest neighbors in a subsets.

**Choosing the number of planes**

* Each plane divides the space to $2$ parts.
* So $n$ planes divide the space into $2^{n}$ hash buckets.
* We want to organize 10,000 document vectors into buckets so that every bucket has about $~16$ vectors.
* For that we need $\frac{10000}{16}=625$ buckets.
* We're interested in $n$, number of planes, so that $2^{n}= 625$. Now, we can calculate $n=\log_{2}625 = 9.29 \approx 10$.

```python
# The number of planes. We use log2(625) to have ~16 vectors/bucket.
N_PLANES = 10
# Number of times to repeat the hashing to improve the search.
N_UNIVERSES = 25
```

For each vector we have to get the unique number to assign in to the hash bucket. We will to use hyperplanes to split the vector space. We can use it to split the vector into two parts. All vectors whose dot product with a plane’s normal vector is positive are on one side of the plane. All vectors whose dot product with the plane’s normal vector is negative are on the other side of the plane. 

For a vector, we can take its dot product with all the planes, then encode this information to assign the vector to a single hash bucket. When the vector is pointing to the opposite side of the hyperplane than normal, encode it by 0.
Otherwise, if the vector is on the same side as the normal vector, encode it by 1. 

We will try to implement this concept in the next function.

```python 
def hash_value_of_vector(v, planes):
    """Create a hash for a vector; hash_id says which random hash to use.
    Input:
        - v:  vector of tweet. It's dimension is (1, N_DIMS)
        - planes: matrix of dimension (N_DIMS, N_PLANES) - the set of planes that divide up the region
    Output:
        - res: a number which is used as a hash for your vector

    """
    # for the set of planes,
    # calculate the dot product between the vector and the matrix containing the planes
    # remember that planes has shape (300, 10)
    # The dot product will have the shape (1,10)
    dot_product = np.dot(v,planes)
    # get the sign of the dot product (1,10) shaped vector
    sign_of_dot_product = np.sign(dot_product)
    # set h to be false (eqivalent to 0 when used in operations) if the sign is negative,
    # and true (equivalent to 1) if the sign is positive (1,10) shaped vector
    h = sign_of_dot_product>=0
    # remove extra un-used dimensions (convert this from a 2D to a 1D array)
    h = np.squeeze(h)
    # initialize the hash value to 0
    hash_value = 0
    n_planes = planes.shape[1]
    for i in range(n_planes):
        # increment the hash value by 2^i * h_i
        hash_value += np.power(2,i)*h[i]
    # cast hash_value as an integer
    hash_value = int(hash_value)
    return hash_value
```

### Create a hash table

We now have a unique number for each vector (tweet), we now need to create a hash table so that given a `hash_id` we can quickly look up to the corresponding vectors. It will reduce the time of searching.

```python
def make_hash_table(vecs, planes):
    # number of planes is the number of columns in the planes matrix
    num_of_planes = planes.shape[1]
    # number of buckets is 2^(number of planes)
    num_buckets = 2**num_of_planes
    # create the hash table as a dictionary.
    # Keys are integers (0,1,2.. number of buckets)
    # Values are empty lists
    hash_table = {i:[] for i in range(num_buckets)}
    # create the id table as a dictionary.
    # Keys are integers (0,1,2... number of buckets)
    # Values are empty lists
    id_table = {i:[] for i in range(num_buckets)}
    # for each vector in 'vecs'
    for i, v in enumerate(vecs):
        # calculate the hash value for the vector
        h = hash_value_of_vector(v,planes)
    # store the vector into hash_table at key h,
        # by appending the vector v to the list at key h
        hash_table[h].append(v)
    # store the vector's index 'i' (each document is given a unique integer 0,1,2...)
        # the key is the h, and the 'i' is appended to the list at key h
        id_table[h].append(i)
    return hash_table, id_table
```

### Using KNN & LSH

We will combine KNN to locality sensitive hashing to seach for documents that are similar to a given document. The function will then take many inputs 
* `doc_id` is the index into the document list `all_tweets`.
* `v` is the document vector for the tweet in `all_tweets` at index `doc_id`.
* `planes_l` is the list of planes (the global variable created earlier).
* `k` is the number of nearest neighbors to search for.
* `num_universes_to_use`: to save time, we can use fewer than the total
number of available universes.  By default, it's set to `N_UNIVERSES`,
which is $25$ for this assignment.

Then the function will find a subset of candidate vectors that are in the same hash bucket as the input vector to perform the KNN search on this subset.


```python 
# This is the code used to do the fast nearest neighbor search. Feel free to go over it
def approximate_knn(doc_id, v, planes_l, k=1, num_universes_to_use=N_UNIVERSES):
    """Search for k-NN using hashes."""
    assert num_universes_to_use <= N_UNIVERSES

    # Vectors that will be checked as possible nearest neighbor
    vecs_to_consider_l = list()

    # list of document IDs
    ids_to_consider_l = list()

    # create a set for ids to consider, for faster checking if a document ID already exists in the set
    ids_to_consider_set = set()

    # loop through the universes of planes
    for universe_id in range(num_universes_to_use):

        # get the set of planes from the planes_l list, for this particular universe_id
        planes = planes_l[universe_id]

        # get the hash value of the vector for this set of planes
        hash_value = hash_value_of_vector(v, planes)

        # get the hash table for this particular universe_id
        hash_table = hash_tables[universe_id]

        # get the list of document vectors for this hash table, where the key is the hash_value
        document_vectors_l = hash_table[hash_value]

        # get the id_table for this particular universe_id
        id_table = id_tables[universe_id]

        # get the subset of documents to consider as nearest neighbors from this id_table dictionary
        new_ids_to_consider = id_table[hash_value]

        # remove the id of the document that we're searching
        if doc_id in new_ids_to_consider:
            new_ids_to_consider.remove(doc_id)
            print(f"removed doc_id {doc_id} of input vector from new_ids_to_search")

        # loop through the subset of document vectors to consider
        for i, new_id in enumerate(new_ids_to_consider):

            # if the document ID is not yet in the set ids_to_consider...
            if new_id not in ids_to_consider_set:
                # access document_vectors_l list at index i to get the embedding
                # then append it to the list of vectors to consider as possible nearest neighbors
                document_vector_at_i = document_vectors_l[i]

                # append the new_id (the index for the document) to the list of ids to consider
                vecs_to_consider_l.append(document_vector_at_i)
                ids_to_consider_l.append(new_id)

                # also add the new_id to the set of ids to consider
                # (use this to check if new_id is not already in the IDs to consider)
                ids_to_consider_set.add(new_id)

    # Now run k-NN on the smaller set of vecs-to-consider.
    print("Fast considering %d vecs" % len(vecs_to_consider_l))

    # convert the vecs to consider set to a list, then to a numpy array
    vecs_to_consider_arr = np.array(vecs_to_consider_l)

    # call nearest neighbors on the reduced list of candidate vectors
    nearest_neighbor_idx_l = nearest_neighbor(v, vecs_to_consider_arr, k=k)

    # Use the nearest neighbor index list as indices into the ids to consider
    # create a list of nearest neighbors by the document ids
    nearest_neighbor_ids = [ids_to_consider_l[idx]
                            for idx in nearest_neighbor_idx_l]

    return nearest_neighbor_ids

```