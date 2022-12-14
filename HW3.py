import math
import numpy as np
import time as tm
import os
import re
import random 

# make global: embeddings, all input words
embeddings = {}
board_words = []

# takes in filename, gets a dictionary of all embeddings in the file
def load_embeddings(filename):
    f = open(filename)
    line = f.readline()
    size = (int(line.split()[0]))
    for i in range(int(size/2)):
        line = f.readline().split()
        word = line[0].split('_')[0].lower()
        if ':' in word or '</s>' in word:
            continue
        embeddings[word] = [float(x) for x in line[1:]]

#randomly generates input dictionary 
def generate_inputDict():
    f = open("codenames_default.txt")
    codenames = []
    line = f.readline()
    while line != "":
        line = line.strip('\n')
        codenames.append(line.lower())
        line = f.readline() #go to next line
        
    random.shuffle(codenames) #shuffle codenames
    
    #our words - 7, their words - 8, neutral words - 9, assassin word - 1
    our_words = codenames[:7]
    their_words = codenames[8:16]
    neutral_words = codenames[17:26]
    assassin_word = codenames[27:28]
    
    # TODO: for testing, remove
    print("Our Words:", len(our_words))
    print(our_words)
    print("Their Words:", len(their_words))
    print(their_words)
    print("Neutral Words:", len(neutral_words))
    print(neutral_words)
    print("Assassin Word:", len(assassin_word))
    print(assassin_word)
    
    return { 'our words':our_words, 'their words': their_words, 'neutral words': neutral_words, 'assassin word': assassin_word[0] } 
def spymaster(inputDict):
    # split the inputted dictionary into the appropriate dictionaries, base vector being 0,0,0,0,...
    # format is {"word": vector, "word 2": vector 2, ...}
    our_words = {}
    their_words = {}
    neutral_words = {}
    assassin_word = {}
    # assign the global all words
    board_words = inputDict.get('our words') + inputDict.get('their words') + inputDict.get('neutral words') + [inputDict.get('assassin word')]
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
    
    # find the ideal vector, and how many words it corrolates to
    ideal, words_clued_for = findIdealVector(our_words)
    ideal = improve_vector(ideal, list(their_words.values()), list(assassin_word.values())[0]) #their_vects should only be values, for assassin we specify so we get only the vector, not list  
    print("Clue:", getClue(ideal, board_words))
    print("Number:", len(words_clued_for))
    print("For:", words_clued_for)
    return

# finds and sets the vectors for a specific word
# input: codenames words
def setVectors(dictOfWords):
    for word in dictOfWords.keys():
        word_vector = embeddings.get(word)
        #parse through data and find the vector for the word
        dictOfWords[word] = word_vector
    return dictOfWords

# calculates and returns the cosine distance between 2 vectors
def distance(v1,v2):
    numerator = 0
    denominator1 = 0
    denominator2 = 0
    for i in range(len(v1)):
        numerator += (v1[i]*v2[i])
        denominator1 += (v1[i]**2)
        denominator2 += (v2[i]**2)
    return (numerator / (math.sqrt(denominator1)*math.sqrt(denominator2)))

# finds the average distance between the current words
def getAveDistances(vectors, n):
    ave_distances = np.full(n,0.0)
    for i in range(n):
        dist = 0.0
        for j in range(n):
            if j!= i:
                dist+=distance(vectors[i],vectors[j])
        ave_distances[i]=dist/(n-1)
    return ave_distances

# calculates if the possible vector has a better average distance than the word that is furthest away from the others
def vectorIsCloser(ave_distances, vectors, n, possible, index):
    startingAve = np.sum(getAveDistances(vectors, n))
    tempVects = np.full_like(vectors, 0.0)
    for i in range(len(vectors)):
        if i!= index:
            tempVects[i]=vectors[i]
        else:
            tempVects[i] = possible
    temp_ave_distances = np.sum(getAveDistances(tempVects, n))
    if temp_ave_distances<startingAve:
        return True
    else:
        return False

# finds the index that causes the maximum distance from the others
def findTrueMax(vectors, ave_distances):
    minDist = 300
    maxInd = np.argmax(ave_distances)
    if(len(vectors)>2):
        for i in range(len(vectors)):
            tempVectors = np.delete(vectors, i, 0)
            tempDist = np.sum(getAveDistances(tempVectors, len(tempVectors)))
            if tempDist<minDist:
                minDist = tempDist
                maxInd = i
    return maxInd

# finds the closest words in the dictionary, returns the average of them and the number of words the average corrolates to [hard coded to 2]
# @parameter: dictionary of teams words
# @returns: the ideal vector, number of words it corrolates to, and what those words are
def findIdealVector(dictOfWords):
    listOfVectors = list(dictOfWords.items())
    maxDistAllowed = 0.065
    
    if len(dictOfWords)>=4:
        num_words = 4
    else:
        num_words = len(dictOfWords)
    closest_words = np.full((num_words,1), " "*30) #the closest together words
    vectors = np.full((num_words,300),0.0) #the actual vectors
    for i in range(num_words):
        closest_words[i] = listOfVectors[i][0]
        vectors[i] = listOfVectors[i][1]
        i+=1
    
    while True:
        # if there are more then 1 word left in the dictionary
        if num_words>2:
            ave_distances = getAveDistances(vectors, num_words) 
            maxInd = findTrueMax(vectors,ave_distances)
            
            i = 0
            while i<len(dictOfWords): #for each word in the dictionary
                if listOfVectors[i][0] not in closest_words: #if the word is not already in the list
                    if vectorIsCloser(ave_distances, vectors, num_words, listOfVectors[i][1], maxInd): #and the word is closer then the farthest word
                        closest_words[maxInd]= listOfVectors[i][0]
                        vectors[maxInd]=listOfVectors[i][1]
                        ave_distances = getAveDistances(vectors, num_words)
                        # resets the other variables, and goes back through the loop of comparisons with the new list of words
                        i = -1
                        maxInd = findTrueMax(vectors,ave_distances)
                i+=1
            # if all the words are less than some distance apart, return
            if np.all(ave_distances<maxDistAllowed):
                return calculateIdealVector(vectors), closest_words
            # otherwise, remove the farthest word and rerun to check if there is a better word to fill in
            else:
                maxInd = findTrueMax(vectors, ave_distances)
                closest_words = np.delete(closest_words, maxInd, 0)
                vectors = np.delete(vectors, maxInd, 0)
                num_words = num_words-1
        else:
            if num_words==2:
                return closestTwoWords(list(dictOfWords.items()))
            # if theres only 1 word left, return it
            return calculateIdealVector(listOfVectors[0][1]), [listOfVectors[0][0]]
        
# finds only the closest 2 words
def closestTwoWords(listOfVectors):
    minDist = 300
    closest_words = [" "*30, " "*30]
    closest_vectors = [listOfVectors[0][1], listOfVectors[1][1]]
    for i in range(len(listOfVectors)):
        for j in range(len(listOfVectors)):
            if listOfVectors[i][1] != None and listOfVectors[j][1]!= None:
                dist = distance(listOfVectors[i][1],listOfVectors[j][1])
                if i!=j and minDist>distance(listOfVectors[i][1],listOfVectors[j][1]):
                    minDist = dist
                    closest_words=[listOfVectors[i][0], listOfVectors[j][0]]
                    closest_vectors=[listOfVectors[i][1], listOfVectors[j][1]]
    return calculateIdealVector(closest_vectors), closest_words
    
# averages the vectors given to get a ideal vector for the clue
def calculateIdealVector(vectors):
    appendedVectors = [vectors[0]]
    i = 1
    while i<len(vectors):
        appendedVectors = np.vstack((appendedVectors,vectors[i]))
        i+=1
    return np.mean(appendedVectors.astype(float), axis = 0)

def improve_vector(ideal, their_vects, assassin_vect):
    for their_vect in their_vects:
        ideal = np.subtract(ideal, np.array(their_vect) * (.009 / distance(ideal, their_vect)))
    ideal = np.subtract(ideal, np.array(assassin_vect) * (.05 / distance(ideal, assassin_vect)))
    return ideal

# input: ideal vector, all word embeddings, all input words
# output: clue word 
# create a dictionary where the key is the word, value is distance from ideal vector
# find the word associated with minimum distance
def getClue(ideal_v, board_words):
    loop = True
    distance_dict = { w:distance(ideal_v, v) for w,v in embeddings.items()} # add in if w not in one of the input dictionaries
    while loop: 
        # get clue with shortest distance
        clue = min(distance_dict, key=distance_dict.get)
        # check that the clue does not conflict with any words on the board
        loop = False
        for word in board_words: 
            if bool(re.search(clue, word)) or bool(re.search(word, clue)):
                distance_dict.pop(clue, None)
                loop = True
    return clue

#filename = os.path.join(os.getcwd(), "1", "model.txt")

#start_time=tm.time()
filename = "model.txt"
load_embeddings(filename)
inputDict=generate_inputDict()
spymaster(inputDict)
#t = tm.time()-start_time
#print("Took", int(t/60), "minutes and", "{:.2f}".format(t - 60*int(t/60)), "seconds find clue")
