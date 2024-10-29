import os
import numpy as np

class TransitionVector:
    """Contains the probabilities for a single token"""

    _probabilityVec = np.array(0)

    _transition_count = 0

    _id = 0

    _wordCount = 0

    def __init__(self, transitionMat, wordDict, id):
        self._probabilityVec = np.array(transitionMat[id]) # We copy to keep data being worked on thread local
        self._transition_count = len(wordDict[list(wordDict)[id]][2])
        self._id = id
        self._wordCount = len(wordDict)

    def _probability_to_ranges(self, prob_vector):
    
        range_vector = np.zeros(len(prob_vector))
    
        # Track cumulative probability
        # All non-zero values cover a number range (iterate with if p <= P)
        # Zero values are converted to -1 so are never picked
        cumulative_prob = 0.0
        for i in range(0, len(prob_vector)):
            if prob_vector[i] == 0:
                range_vector[i] = -1.0
            else:
                range_vector[i] = prob_vector[i] + cumulative_prob
                cumulative_prob = cumulative_prob + prob_vector[i]

        return range_vector

    def calculate_transition(self):
        for j in range(0, self._wordCount):
            if (self._transition_count == 0):
                self._probabilityVec[j] = 0
            else:
                self._probabilityVec[j] = self._probabilityVec[j] / self._transition_count
        self._probabilityVec = self._probability_to_ranges(self._probabilityVec)

    def probabilities(self):
        return self._probabilityVec