import numpy as np
from scipy.spatial.distance import cosine

class DocumentRetriever:
    def __init__(self, path):
        self.path = path
        arr = []
        lines = open(path, "r").read().splitlines()
        for line in lines:
            arr.append(line.split()[2:])
        self.matrix = np.array(arr, dtype=np.float64)
    
    def n_most_similar(self, doc, n=3, metric='cosine'):
        result = []
        if (metric=='cosine'):
            distances = []
            i = 0;
            for item in self.matrix:
                distances.append([i,cosine(doc, item)])
                i+=1
            distances.sort(key=lambda x: x[1])  
            for rank in distances[:n]:
                result.append(rank[0])
        return result        