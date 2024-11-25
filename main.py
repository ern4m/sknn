from math import sqrt
import numpy as np
import matplotlib.pyplot as plot

# k: int
# item: values[]
# neighbour: {group: (X,Y)[]}
# neighbours: neighbour[]
#
# euclidean_distance(item, neighbours) -> distance
# distance: int | int[] (obs: check how to calculate distances in multidimentional classes)
# distances: (distance, class)[]
#
# find_nearests(distances, k) -> knn: sorted(distances)[0..2]
# classify(knn) -> class: sort_by_class(knn)[0]

def euclidean_distance(item, neighbour):
    return sqrt((item[0]-neighbour[0])**2 + (item[1]-neighbour[1])**2)

class knn:
    def __init__(self, neighbours, item, k=1):
        self.neighbours = neighbours
        self.item = item
        self.k = k
        self.distances = []
        self.classification = ''
        self.frequencies = [0, 0]
    
    def calculate_distances(self):
        # getting distances for each neighbour in each class
        for group in self.neighbours:
            for neighbour in self.neighbours[group]:
                self.distances.append((euclidean_distance(self.item, neighbour), group))

        # sorting distances and getting 'k' firsts
        self.distances = sorted(self.distances)

    def classificate(self):
        # making classification based on distance
        for neighbour in self.distances[:self.k]:
            if neighbour[1] == 0:
                self.frequencies[0] += 1
            else:
                self.frequencies[1] += 1

        if self.frequencies[0] > self.frequencies[1]:
            self.classification = '0'
        else: 
            self.classification = '1'
        
        return self.classification
    
    
    # resolve this function to distinct given classes and item
    def plot(self):
        _n = list(self.neighbours.values())
        print(_n)
        plot.plot(_n[0], 'o', linewidth=0, markersize=5)
        plot.plot(_n[1], 'v', linewidth=0, markersize=5)
        plot.plot(self.item[0], self.item[1], 'o', linewidth=0, markersize=5)
        plot.show()

def main():
    points = {0:[(1,12),(2,5),(3,6),(3,10),(3.5,8),(2,11),(2,9),(1,7)], 1:[(5,3),(3,2),(1.5,9),(7,2),(6,1),(3.8,1),(5.6,4),(4,2),(2,5)]}
        
    item = (2, 3)
    
    _knn = knn(points, item)
    _knn.calculate_distances()
    
    k = [1,3,7]
    for n in k:
        _knn.k = n
        print(f"For k={n} we have that item ({item} is classified as {_knn.classificate()})")

    _knn.plot()

if __name__=='__main__':
    main()