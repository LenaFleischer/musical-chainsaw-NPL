#key: word, value: list of numbers (each word has about 300 numbers)
#returns dictionary/list of possible words for the game
def load_embeddings(filename):
    f = open(filename)
    line = f.readline()
    size = (int(line.split()[0]))
    embeddings = {}
    for i in range(size):
        line = f.readline().split()
        word = line[0].split('_')[0].lower()
        if ':' in word or line[0].split('_')[1] == 'PROPN':
            continue
        embeddings[word] = [float(x) for x in line[1:]]
    return embeddings