import math
import numpy as np
import time as tm

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

start_time=tm.time()
embeddings = load_embeddings("model.txt")
t = tm.time()-start_time
print("Took", int(t/60), "minutes and", "{:.2f}".format(t - 60*int(t/60)), "seconds to load embeddings")

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
    ideal, number = findIdealVector(our_words)
    print(ideal)
    return

# finds and sets the vectors for a specific word
# input: codenames words
def setVectors(dictOfWords):
    for word in dictOfWords.keys():
        word_vector = embeddings.get(word)
        #parse through data and find the vector for the word
        dictOfWords[word] = word_vector
    return dictOfWords


# finds the closest words in the dictionary, returns the average of them and the number of words the average corrolates to [hard coded to 2]
# TODO: make this word for 3/4/1/etc number of words and make it not suck
def findIdealVector(dictOfWords):
    HARD_CODED_NUMBER = 2
    min_dist = 100
    w1 = ""
    w2 = ""
    for word1 in dictOfWords.keys():
        for word2 in dictOfWords.keys():
            if word1!=word2 and distance(dictOfWords[word1],dictOfWords[word2])<min_dist:
                min_dist = distance(dictOfWords[word1],dictOfWords[word2])
                w1 = word1
                w2 = word2
    ideal = np.full(300,0.0)
    for value in range(len(ideal)):
        ideal[value] = (dictOfWords[w1][value]+dictOfWords[w2][value])/2
    return ideal, HARD_CODED_NUMBER

# calculates and returns the euclidean distance between 2 vectors
def distance(v1,v2):
    return math.sqrt((v1[0]-v2[0])**2+(v1[1]-v2[1])**2+(v1[2]-v2[2])**2)

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

inputDict = {'our words': ['chair', 'fruit', 'banana'], 'their words': ['dinosaur', 'mug', 'computer'], \
             'neutral words': ['planet', 'france', 'bird'], 'assassin word': 'cup'}
spymaster(inputDict)
