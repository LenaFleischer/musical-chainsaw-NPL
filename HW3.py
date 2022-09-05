import math
import numpy as np
import time as tm
import os
import re
import random 

# make global: embeddings, all input words, hardcoded 2 words
embeddings = {}
board_words = []

# takes in filename, gets a dictionary of all embeddings in the file
def load_embeddings(filename):
    f = open(filename)
    line = f.readline()
    size = (int(line.split()[0]))
    # embeddings = {}
    # global embeddings
    for i in range(size):
        line = f.readline().split()
        word = line[0].split('_')[0].lower()
        # if ':' in word or line[0].split('_')[1] == 'PROPN': # takes out proper nouns
        if ':' in word or '</s>' in word:
            continue
        embeddings[word] = [float(x) for x in line[1:]]
    # return embeddings

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
    
    # find the ideal vector, and how many words it corrolates to without thinking about the other words [yet]
    ideal, number, words_clued_for = findIdealVector(our_words)
    improve_vector(ideal, their_words.values(), assassin_word.values()[0]) #their_vects should only be values, for assassin we specify so we get only the vector, not list  
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

# input: ideal vector, all word embeddings, all input words
# output: clue word 
# create a dictionary where the key is the word, value is distance from ideal vector
# find the word associated with minimum distance
def getClue(embeddings,ideal_v):
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

filename = os.path.join(os.getcwd(), "1", "model.txt")
load_embeddings(filename)

# # test getClue with some test data
# embeddings = {'hi': [-0.071887, 0.010656, 0.042305, 0.007555, -0.147592, -0.024493, -0.030536, 0.012723, -0.050257, -0.004533, 0.007236, -0.038965, -0.055029, -0.006958, -0.061072, 0.006441, 0.072523, 0.171766, -0.039602, 0.014234, -0.120872, 0.014155, 0.025606, -0.09797, -0.050894, -0.074114, -0.00664, 0.006004, 0.014234, 0.001044, 0.07475, 0.026083, -0.036103, -0.082066, -0.019721, -0.01487, -0.067116, -0.025129, -0.019006, -0.042942, -0.027355, 0.061072, -0.004394, 0.045486, -0.005765, -0.056619, -0.008191, -0.10306, -0.036739, 0.039284, 0.025606, -0.046759, -0.016381, 0.042623, -0.001064, 0.030854, -0.020278, -0.005726, -0.016938, -0.07475, 0.008827, -0.07634, -0.076977, 0.060754, 0.032604, -0.058528, -0.02831, 0.003062, 0.073796, 0.003578, 0.062663, 0.041351, 0.041669, -0.013042, -0.095426, 0.004334, -0.000142, 0.001541, 0.013439, 0.112602, 0.104968, -0.055029, -0.034353, -0.024015, 0.07634, 0.124053, -0.076659, 0.001501, -0.00336, 0.030695, -0.048667, 0.041033, 0.050257, -0.014393, -0.058846, 0.065526, 0.084611, 0.010576, 0.166677, 0.03817, 0.02831, 0.082066, 0.107513, 0.019006, 0.028787, 0.072205, -0.027196, -0.034512, -0.022425, 0.042305, 0.012644, -0.025924, -0.095426, -0.016302, 0.049303, 0.053756, -0.006282, -0.090972, 0.072842, -0.0897, -0.021312, 0.025765, -0.044214, 0.021948, 0.034194, 0.078885, 0.027514, 0.058528, 0.04644, 0.055029, 0.030218, 0.022584, -0.068706, -0.069661, -0.036262, 0.010059, -0.063299, 0.042623, -0.028787, 0.082702, -0.038965, -0.037534, 0.103696, 0.015348, 0.009224, -0.040238, -0.014473, -0.001759, 0.007077, 0.050576, 0.013757, -0.007157, 0.014791, -0.117055, -0.077295, -0.02497, 0.055983, -0.049939, 0.048667, 0.088428, 0.037693, -0.007395, 0.050257, 0.009463, -0.024174, -0.158407, 0.012723, 0.112602, -0.060436, 0.121509, 0.12787, 0.002276, -0.018131, -0.033399, -0.018449, 0.022902, 0.033558, -0.059482, -0.045804, 0.019085, -0.012644, 0.030218, 0.110058, 0.073796, 0.00839, 0.104332, 0.033399, -0.031968, 0.016143, 0.033399, -0.029582, 0.013916, 0.009741, 0.030218, -0.033081, 0.010417, 0.043896, -0.074432, 0.052484, -0.041669, -0.049939, 0.019562, -0.00672, -0.067434, -0.0897, -0.092245, 0.006839, -0.034194, -0.002435, -0.047713, 0.097334, 0.018847, -0.079839, 0.082702, -0.031968, -0.050894, 0.117055, 0.098606, -0.052802, -0.124689, -0.006083, 0.055665, 0.070933, 0.0598, -0.065526, 0.125326, -0.014075, 0.010735, -0.010179, 0.038329, -0.136777, 0.025606, -6.5e-05, -0.021471, 0.026878, -0.001918, -0.000142, 0.057892, 0.004314, 0.022107, 0.008509, -0.063299, 0.059164, 0.012644, -0.087155, -0.022902, -0.032763, 0.022584, 0.011769, -0.038329, 0.075704, 0.010179, 0.08143, -0.075704, 0.065526, 0.003181, 0.058528, 0.052166, 0.110694, 0.018051, 0.016222, -0.056937, 0.020994, -0.090336, -0.018369, -0.039125, -0.054075, 0.012803, 0.160315, 0.052166, -0.040715, 0.040715, 0.02163, -0.076977, 0.007276, 0.04326, -0.087155, -0.048031, 0.091609, 0.148864, -0.048985, -0.034035, -0.055347, -0.020357, 0.043896, 0.00672, 0.027992, -0.053756, 0.005765, 0.048031, 0.032604, 0.02163, 0.016143, -0.034194, -0.064571, 0.006799, -0.161587, 0.000634, -0.108785, 0.06807], 'apple': [-0.054797, -0.107925, 0.025034, 0.023365, -0.08456, -0.019332, -0.044505, -0.028789, -0.096799, 0.007163, -0.058135, -0.010083, 0.037829, -0.034492, -0.049234, 0.080666, 0.097911, -0.014047, -0.025312, 0.030458, -0.034352, 0.1057, 0.079553, -0.086785, -0.087898, 0.009805, 0.060638, 0.16912, -0.011544, -0.024617, 0.031293, 0.021835, -0.038108, -0.002156, 0.024895, -0.070652, -0.024061, 0.044783, 0.125171, -0.004659, -0.072877, -0.053406, -0.038386, -0.045896, 0.017733, 0.055631, -0.006363, 0.115714, -0.019332, 0.026286, -0.076215, 0.003912, 0.067036, 0.076771, 0.010987, 0.09513, -0.053684, -0.081222, 0.003651, 0.019193, 0.002008, 0.062307, -0.064533, 0.010014, -0.061195, -0.064811, 0.080666, -0.094017, -0.066758, -0.018219, -0.030458, 0.004763, -0.067592, -0.033657, -0.084004, 0.068427, 0.034909, -0.009875, -0.030736, 0.017315, -0.109038, 0.00186, -0.040611, -0.073434, -0.059804, 0.019332, 0.063976, 0.07093, -0.03922, -0.016064, -0.168007, -0.014951, 0.019332, -0.123502, 0.047843, -0.06342, -0.043393, 0.053684, 0.062307, 0.026564, 0.048956, 0.066201, 0.042002, -0.013212, 0.035048, 0.038386, -0.03171, 0.056466, 0.023504, -0.107369, 0.017107, -0.048399, 0.050625, 0.018497, 0.006224, 0.043949, -0.065923, -0.07399, 0.019332, -0.072321, -0.031015, 0.024478, -0.007267, -0.037829, 0.023922, 0.077328, -0.115714, 0.100693, -0.031432, -0.000648, -0.002086, -0.050903, -0.0009, -0.053963, -0.038108, 0.075659, 0.006606, 0.008692, -0.036717, -0.047287, 0.03616, -0.069539, -0.018219, -0.101806, -0.034074, -0.021696, 0.022531, 0.033935, -0.018776, -0.021418, 0.052294, -0.160219, -0.036995, 0.081778, -0.069539, 0.034074, -0.029624, -0.037551, -0.053406, 0.013282, -0.154656, 0.035048, 0.03922, -0.05591, -0.037273, -0.008762, 0.072877, 0.010153, -0.052572, 0.04673, -0.009179, 0.024339, -0.145755, -0.007893, -0.002538, 0.021835, 0.045896, 0.010709, 0.024061, 0.040611, -0.076771, -0.018219, 0.076215, 0.020166, 0.056466, 0.020305, -0.026564, -0.063698, -0.033657, -0.013282, -0.033379, 0.08901, -0.11015, -0.025173, -0.057022, -0.006154, -0.069261, -0.043949, 0.095686, 0.01203, 0.004207, -0.045896, 0.029624, -0.005598, -0.151318, 0.028511, 0.04979, -0.062585, -0.014742, -0.028372, 0.081778, 0.043393, 0.0573, 0.011265, 0.032405, 0.061473, 0.056188, -0.008449, 0.008414, -0.035882, -0.071765, -0.005285, 0.077328, 0.090679, -0.015438, 0.047287, -0.083447, -0.038942, 0.074546, 0.000398, -0.051459, 0.038386, 0.00043, -0.036995, 0.076771, 0.023365, 0.060082, -0.000123, -0.013143, -0.012517, -0.07399, -0.012448, -0.003303, 0.076771, -0.025869, -0.111263, -0.007754, 0.022253, -0.062307, -0.00911, -0.073434, 0.069818, 0.094017, -0.007545, 0.055353, 0.040611, 0.086785, 0.041724, -0.011404, 0.087341, 0.014256, -0.015507, 0.026147, 0.001669, -0.010779, 0.03477, -0.045062, 0.05591, -0.043671, -0.059804, -0.042836, 0.06787, 0.047287, -0.014951, 0.102362, 0.008032, 0.083447, -0.042558, -0.081222, 0.022253, -0.038386, 0.108481, -0.038108, 0.033518, 0.067592, -0.058413, 0.125727, -0.026008, 0.145755, -0.010292, 0.071765, -0.029207, -0.019332, 0.016481, 0.029346, -0.096242, -0.144642, 0.009527, -0.00911, 0.041167]}
ideal_v = [item + .0001 for item in embeddings.get('hi')] # alter "hi" vector to get an ideal vector close to hi
print(getClue(embeddings, ideal_v))

def improve_vector(ideal, their_vects, assassin_vect):
    for their_vect in their_vects:
        ideal = np.subtract(ideal, np.array(their_vect) * (1 / distance(ideal, their_vect)))
    ideal = np.subtract(ideal, np.array(assassin_vect) * (8 / distance(ideal, assassin_vect)))
    return ideal

#randomly generates input dictionary 
def generate_inputDict():
    f = open(r"C:\Users\Hset Hset Naing\Documents\Natural Language Processing Block 1\musical-chainsaw-NPL\codenames_default.txt")
    codenames = []
    line = f.readline()
    while line != "":
        #print(line)
        line = line.strip('\n')
        codenames.append(line)
        line = f.readline() #go to next line
    
    our_words = [] #7 words
    their_words = [] #8 words
    neutral_words = [] #9 words
    assassin_word = [] #1 word
    
    random.shuffle(codenames) #shuffle codenames
   
    our_words = codenames[:7]
    their_words = codenames[67:75]
    neutral_words = codenames[300:309]
    assassin_word = codenames[399]
    

    print(our_words)
    print(their_words)
    print(neutral_words)
    print(assassin_word)

    return { 'our_words':our_words, 'their_words': their_words, 'neutral_words': neutral_words, 'assassin_word': assassin_word } 
