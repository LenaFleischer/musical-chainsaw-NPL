import math
import numpy as np
import time as tm

# make global: embeddings, all input words, hardcoded 2 words

# takes in filename, gets a dictionary of all embeddings in the file
def load_embeddings(filename):
    f = open(filename)
    line = f.readline()
    size = (int(line.split()[0]))
    embeddings = {}
    # global embeddings
    for i in range(size):
        line = f.readline().split()
        word = line[0].split('_')[0].lower()
        # if ':' in word or line[0].split('_')[1] == 'PROPN': # takes out proper nouns
        if ':' in word or '</s>' in word:
            continue
        embeddings[word] = [float(x) for x in line[1:]]
    return embeddings

# start_time=tm.time()
# embeddings = load_embeddings("model.txt")
# t = tm.time()-start_time
# print("Took", int(t/60), "minutes and", "{:.2f}".format(t - 60*int(t/60)), "seconds to load embeddings")

def spymaster(inputDict):
    # split the inputted dictionary into the appropriate dictionaries, base vector being 0,0,0,0,...
    our_words = {}
    their_words = {}
    neutral_words = {}
    assassin_word = {}
    base_value = np.full(300,0.0)
    for word in inputDict['our words']:
        our_words[word] = base_value
    for word in inputDict['their words']:
        their_words[word] = base_value
    for word in inputDict['neutral words']:
        neutral_words[word] = base_value
    assassin_word[inputDict['assassin word']] = base_value
    
    # call the setVectors method to make vectors actually be correct
    our_words = setVectors(our_words)
    their_words = setVectors(their_words)
    neutral_words = setVectors(neutral_words)
    assassin_word = setVectors(assassin_word)
    
    # find the ideal vector, and how many words it corrolates to without thinking about the other words [yet]
    ideal, number, words_clued_for = findIdealVector(our_words)
    print(ideal)
    print(number)
    print(words_clued_for)
    return

# finds and sets the vectors for a specific word
# input: codenames words
def setVectors(dictOfWords):
    for word in dictOfWords.keys():
        word_vector = embeddings.get(word)
        #parse through data and find the vector for the word
        dictOfWords[word] = word_vector
    return dictOfWords

# averages the vectors given to get a ideal vector for the clue
def calculateIdealVector(vectors):
    appendedVectors = [vectors[0]]
    i = 1
    while i<len(vectors):
        appendedVectors = np.vstack((appendedVectors,vectors[i]))
        i+=1
    return np.mean(appendedVectors.astype(float), axis = 0)

# finds the average distance between the current words
def getAveDistances(closest_dist, vectors, n):
    for i in range(len(closest_dist)):
        dist = 0.0
        for j in range(n):
            if j!= i:
                dist+=distance(vectors[i],vectors[j])
        closest_dist[i]=dist/(n-1)
    return closest_dist

# calculates if the possible vector has a better average distance than the word that is furthest away from the others
def vectorIsCloser(closest_dist, vectors, n, possible):
    dist = 0.0
    for i in range(len(closest_dist)):
        if i != np.argmax(closest_dist):
            dist+=distance(vectors[i],possible)
    dist = dist/(n-1)
    if dist<np.amax(closest_dist):
        return True
    else:
        return False

# finds the closest words in the dictionary, returns the average of them and the number of words the average corrolates to [hard coded to 2]
# @parameter: dictionary of teams words
# @returns: the ideal vector, number of words it corrolates to, and what those words are
def findIdealVector(dictOfWords):
    listOfVectors = list(dictOfWords.items())
    maxDistAllowed = .75 #TODO: no, find actual number for this
    
    if len(dictOfWords)>=4:
        num_words = 4
    else:
        num_words = len(dictOfWords)
    
    while True:
        # if there are more then 1 word left in the dictionary
        if num_words!=1:
            ave_distances = np.full(num_words,0.0) #the average distances
            closest_words = np.array([listOfVectors[0][0]]) #the closest together words
            vectors = np.full((num_words,300),0.0) #the actual vectors

            vectors[0] = listOfVectors[0][1]
            i = 1
            while i<num_words:
                closest_words = np.append(closest_words,listOfVectors[i][0]) 
                vectors[i] = listOfVectors[i][1]
                i+=1
            closest_dist = getAveDistances(ave_distances, vectors, num_words)
              
            # checks to see if there is a word that is on average closer to the others
            i = num_words
            while i<len(dictOfWords):
                if vectorIsCloser(closest_dist, vectors, num_words, listOfVectors[i][1]):
                    maxInd = np.argmax(closest_dist)
                    closest_words[maxInd]= listOfVectors[i][0]
                    vectors[maxInd]=listOfVectors[i][1]
                    closest_dist = getAveDistances(closest_dist, vectors, num_words)
                i+=1
                
            # if all the words are less than some distance apart, return
            if np.all(closest_dist<maxDistAllowed):
                return calculateIdealVector(vectors), num_words, closest_words
            # otherwise, try for 1 less word to connect (4 -> 3 words, 3 -> 2, etc)
            else:
                num_words = num_words-1
        else:
            return calculateIdealVector(listOfVectors[0][1]), 1, closest_words

# calculates and returns the euclidean distance between 2 vectors
def distance(v1,v2):
    inner = 0
    for i in range(len(v1)):
        inner += (v1[i]-v2[i])**2
    return math.sqrt(inner)

# checks the distance between the chosen ideal vector and all other words
# TODO: functionality
def checkVector(their_words, neutral_words, assassin_word, idealVector):
    closeness_allowed_their = 0 #TODO: no
    closeness_allowed_neutral = 0 #TODO: no
    closeness_allowed_assassin = 0 #TODO: no
    for word in their_words.keys():
        if distance(idealVector, their_words[word])<closeness_allowed_their:
            return "fuck"
    for word in neutral_words.keys():
        if distance(idealVector, neutral_words[word])<closeness_allowed_neutral:
            return "fuck"
    for word in assassin_word.keys():
            if distance(idealVector, assassin_word[word])<closeness_allowed_assassin:
                return "fuck"
    return "yeehaw"

inputDict = {'our words': ['chair', 'fruit', 'banana', 'backpack', 'apple', 'couch', 'bed'], 'their words': ['dinosaur', 'mug', 'computer'], \
             'neutral words': ['planet', 'france', 'bird'], 'assassin word': 'cup'}
spymaster(inputDict)


# input: ideal vector, all word embeddings, all input words
# output: clue word 
# create a dictionary where the key is the word, value is distance from ideal vector
# find the word associated with minimum distance
def getClue(embeddings,ideal_v):
    distance_dict = { w:distance(ideal_v, v) for w,v in embeddings.items()} # add in if w not in one of the input dictionaries
    clue = min(distance_dict, key=distance_dict.get)
    return clue

def improve_vector(ideal, their_vects, assassin_vect):
    for their_vect in their_vects:
        ideal = np.subtract(ideal, np.array(their_vect) * (1 / distance(ideal, their_vect)))
    ideal = np.subtract(ideal, np.array(assassin_vect) * (8 / distance(ideal, assassin_vect)))
    return ideal