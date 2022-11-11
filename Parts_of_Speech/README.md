# NLP - Parts-of-Speech Tagging


[epf_nlp/machineTranslation at main · GabrielTruong/epf_nlp](https://github.com/GabrielTruong/epf_nlp/tree/main/machineTranslation)

## Objectives

In this practical work we study part-of-speech (POS) tagging. The POS is the process of assigning a part-of-speech tag (Noun,Verb,Adjective) to each word of an input text. Tagging is difficult because some words can have different tags depending of the context like the following example:

- The whole team played **well**. [adverb]
- You are doing **well** for yourself. [adjective]
- **Well**, this exercise took me forever to complete. [interjection]
- The **well** is dry. [noun]
- Tears were beginning to **well** in her eyes. [verb]

Distinguish the tag of a word in a sentence can help understand the meaning of a sentence. This would be critically important in search queries. Identifying the proper noun, the organization, the stock symbol, or anything similar would greatly improve everything ranging from speech recognition to search. In this report, we will resume the following parts:

- Learn how parts-of-speech tagging works
- Compute the transition matrix A in a Hidden Markov Model
- Compute the emission matrix B in a Hidden Markov Model

## Data

The data used are from the Wall Steet Journal. At our disposal we have a dataset for training and one for testing. The tagged training data has been preprocessed to form a vocabulary which is words from the training dataset that were used two or more times. 

The training set will be used to create the emission, transmission and tag counts.


## Parts-of-speech tagging

### Training

First we will perform parts-of-speech tagger on words that are not ambiguous. In the WSJ corpus, 86% of the token are unambiguous, only have one tag. The others are ambiguous.

Before predicting tags, we need to compute few dictionaries that will help us after. 

**Transition counts**

This dictionary computes the number of times each tag happened next to another tag. In a more mathematical notation, we need to compute:
$$P(t_i |t_{i-1}) \tag{1}$$, the probability of tag at position $$i$$ given the tag at position $$i-1$$.
The dic has then `(prev_tag,tag)` as keys and the number of times the tags apperead as values.

**Emission counts**

This dictionary computes the probability of a word given its tag: 
$$P(w_i|t_i)\tag{2}$$. 
The keys of the dictionary are `(tag,word)` and the values are the number of times that pair appeared in the training set.

**Tag counts**
This dictionary simply stores the tag as key and the nuber of times each tag appeared as values.

```python
def create_dictionaries(training_corpus, vocab):
    """
    Input: 
        training_corpus: a corpus where each line has a word followed by its tag.
        vocab: a dictionary where keys are words in vocabulary and value is an index
    Output: 
        emission_counts: a dictionary where the keys are (tag, word) and the values are the counts
        transition_counts: a dictionary where the keys are (prev_tag, tag) and the values are the counts
        tag_counts: a dictionary where the keys are the tags and the values are the counts
    """
    
    # initialize the dictionaries using defaultdict
    emission_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)
    
    # Initialize "prev_tag" (previous tag) with the start state, denoted by '--s--'
    prev_tag = '--s--' 
    
    # use 'i' to track the line number in the corpus
    i = 0 
    
    # Each item in the training corpus contains a word and its POS tag
    # Go through each word and its tag in the training corpus
    for word_tag in training_corpus:
        
        # Increment the word_tag count
        i += 1
        
        # Every 50,000 words, print the word count
        if i % 50000 == 0:
            print(f"word count = {i}")
            
        # get the word and tag using the get_word_tag helper function (imported from utils_pos.py)
        word, tag = get_word_tag(word_tag,vocab)
        
        # Increment the transition count for the previous word and tag
        transition_counts[(prev_tag, tag)] += 1 
        
        # Increment the emission count for the tag and word
        emission_counts[(tag, word)] += 1 

        # Increment the tag count
        tag_counts[tag] += 1

        # Set the previous tag to this tag (for the next iteration of the loop)
        prev_tag = tag 
                
    return emission_counts, transition_counts, tag_counts
```

We will create the dictionaries based on the training_corpus and we will see what they look like. 

```python
emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab)
```

```python 
print("transition examples: ")
for ex in list(transition_counts.items())[:3]:
    print(ex)
print()

print("emission examples: ")
for ex in list(emission_counts.items())[200:203]:
    print (ex)
print()

print("ambiguous word example: ")
for tup,cnt in emission_counts.items():
    if tup[1] == 'back': print (tup, cnt) 
```

### Testing

Let's test the accuracy of our parts-of-speech tagger using the `emission_counts` dictionary. We will predict the tags of every word in the `prep` corpus and compare it to the original `y` corpus. 

```python
def predict_pos(prep, y, emission_counts, vocab, states):
    '''
    Input: 
        prep: a preprocessed version of 'y'. A list with the 'word' component of the tuples.
        y: a corpus composed of a list of tuples where each tuple consists of (word, POS)
        emission_counts: a dictionary where the keys are (tag,word) tuples and the value is the count
        vocab: a dictionary where keys are words in vocabulary and value is an index
        states: a sorted list of all possible tags for this exercise
    Output: 
        accuracy: Number of times you classified a word correctly
    '''
    
    # Initialize the number of correct predictions to zero
    num_correct = 0
    
    # Get the (tag, word) tuples, stored as a set
    all_words = set(emission_counts.keys())
    
    # Get the number of (word, POS) tuples in the corpus 'y'
    total = len(y)
    for word, y_tup in zip(prep, y): 

        # Split the (word, POS) string into a list of two items
        y_tup_l = y_tup.split()
        
        # Verify that y_tup contain both word and POS
        if len(y_tup_l) == 2:
            
            # Set the true POS label for this word
            true_label = y_tup_l[1]

        else:
            # If the y_tup didn't contain word and POS, go to next word
            continue
    
        count_final = 0
        pos_final = ''
        
        # If the word is in the vocabulary...
        if word in vocab:
            for pos in states:
                        
                # define the key as the tuple containing the POS and word
                key = (pos,word)

                # check if the (pos, word) key exists in the emission_counts dictionary
                if key in emission_counts:  

                # get the emission count of the (pos,word) tuple 
                    count = emission_counts[key]
                    
                    # keep track of the POS with the largest count
                    if count > count_final:  

                        # update the final count (largest count)
                        count_final = count
                        # update the final POS
                        pos_final = pos

            # If the final POS (with the largest count) matches the true POS:
            if pos_final == true_label:
                
                # Update the number of correct predictions
                num_correct += 1 
            
    accuracy = num_correct / total
    
    return accuracy
```

We reached an accuracy of 0.8889. Now let's improve this model using hidden markov models. 

## Hidden Markov Models for Parts-of-Speech

Now we want to implement a Hidden Markov Model, model widely used in Natural Language Processessing. A Markov Model contains a number of states and the probability of transition between those states. 
In our case, the states represents the parts-of-speech. A Markov Model utilizes a transition matrix A. 

Let's implement `A` the transition probabilities matrix and `B` the emission probabilities matrix. To compute these matrices, we will perform smoothing done as follows:

$$ P(t_i | t_{i-1}) = \frac{C(t_{i-1}, t_{i}) + \alpha }{C(t_{i-1}) +\alpha * N}\tag{3}$$

- $N$ is the total number of tags
- $C(t_{i-1}, t_{i})$ is the count of the tuple (previous POS, current POS) in `transition_counts` dictionary.
- $C(t_{i-1})$ is the count of the previous POS in the `tag_counts` dictionary.
- $\alpha$ is a smoothing parameter.

```python
def create_transition_matrix(alpha, tag_counts, transition_counts):
    ''' 
    Input: 
        alpha: number used for smoothing
        tag_counts: a dictionary mapping each tag to its respective count
        transition_counts: transition count for the previous word and tag
    Output:
        A: matrix of dimension (num_tags,num_tags)
    '''
    # Get a sorted list of unique POS tags
    all_tags = sorted(tag_counts.keys())
    
    # Count the number of unique POS tags
    num_tags = len(all_tags)
    
    # Initialize the transition matrix 'A'
    A = np.zeros((num_tags,num_tags))
    
    # Get the unique transition tuples (previous POS, current POS)
    trans_keys = set(transition_counts.keys())
        
    # Go through each row of the transition matrix A
    for i in range(num_tags):
        
        # Go through each column of the transition matrix A
        for j in range(num_tags):

            # Initialize the count of the (prev POS, current POS) to zero
            count = 0
        
            # Define the tuple (prev POS, current POS)
            # Get the tag at position i and tag at position j (from the all_tags list)
            key = (all_tags[i],all_tags[j])

            # Check if the (prev POS, current POS) tuple 
            # exists in the transition counts dictionary
            if key in trans_keys :
                
                # Get count from the transition_counts dictionary 
                # for the (prev POS, current POS) tuple
                count = transition_counts[key]## ADD YOUR CODE HERE
                
            # Get the count of the previous tag (index position i) from tag_counts
            count_prev_tag = tag_counts[all_tags[i]]## ADD YOUR CODE HERE
            
            # Apply smoothing using count of the tuple, alpha, 
            # count of previous tag, alpha, and total number of tags
            A[i,j] = (count+ alpha) / (count_prev_tag+alpha*num_tags)## ADD YOUR CODE HERE
    
    return A
```


For `B` the emission matrix, we also use smoothing defined below:

$$P(w_i | t_i) = \frac{C(t_i, word_i)+ \alpha}{C(t_{i}) +\alpha * N}\tag{4}$$

- $C(t_i, word_i)$ is the number of times $word_i$ was associated with $tag_i$ in the training data (stored in `emission_counts` dictionary).
- $C(t_i)$ is the number of times $tag_i$ was in the training data (stored in `tag_counts` dictionary).
- $N$ is the number of words in the vocabulary
- $\alpha$ is a smoothing parameter. 

The matrix `B` is of dimension (num_tags, N), where num_tags is the number of possible parts-of-speech tags. 

```python
def create_emission_matrix(alpha, tag_counts, emission_counts, vocab):
    '''
    Input: 
        alpha: tuning parameter used in smoothing 
        tag_counts: a dictionary mapping each tag to its respective count
        emission_counts: a dictionary where the keys are (tag, word) and the values are the counts
        vocab: a dictionary where keys are words in vocabulary and value is an index.
               within the function it'll be treated as a list
    Output:
        B: a matrix of dimension (num_tags, len(vocab))
    '''
    
    # get the number of POS tag
    num_tags = len(tag_counts)
    
    # Get a list of all POS tags
    all_tags = sorted(tag_counts.keys())
    
    # Get the total number of unique words in the vocabulary
    num_words = len(vocab)
    
    # Initialize the emission matrix B with places for
    # tags in the rows and words in the columns
    B = np.zeros((num_tags, num_words))
    
    # Get a set of all (POS, word) tuples 
    # from the keys of the emission_counts dictionary
    emis_keys = set(list(emission_counts.keys()))
        
    # Go through each row (POS tags)
    for i in range(num_tags): 
        
        # Go through each column (words)
        for j in range(num_words):

            # Initialize the emission count for the (POS tag, word) to zero
            count = 0
                    
            # Define the (POS tag, word) tuple for this row and column
            key = (all_tags[i],vocab[j])

            # check if the (POS tag, word) tuple exists as a key in emission counts
            if key in emis_keys: 
        
                # Get the count of (POS tag, word) from the emission_counts d
                count = emission_counts[key]
                
            # Get the count of the POS tag
            count_tag = tag_counts[all_tags[i]]
                
            # Apply smoothing and store the smoothed value 
            # into the emission matrix B for this row and column
            B[i,j] = (count+alpha)/(count_tag+alpha*num_words) 
    return B
    ```