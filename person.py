class Person:
    def __init__(self, person_id, centroid, histograms=None):
        self.person_id = person_id
        self.centroids = [centroid]
        self.histograms = histograms