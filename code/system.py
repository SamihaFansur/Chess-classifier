"""Baseline classification system.

Solution outline for the COM2004/3004 assignment.

This solution will run but the dimensionality reduction and
the classifier are not doing anything useful, so it will
produce a very poor result.

version: v1.0
"""
from typing import List
import scipy.linalg
import collections as coll

import numpy as np

N_DIMENSIONS = 10


def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> List[str]:
    """Classify a set of feature vectors using a training set.

    This dummy implementation simply returns the empty square label ('.')
    for every input feature vector in the test data.

    Note, this produces a surprisingly high score because most squares are empty.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """
    
    k = 6
    labels = findKnn(train, test, train_labels, k)
    
#     labels = nn(train, test, train_labels)
    
    return labels

#Function that uses nearest neighbour to classify
def nn(train, test, train_labels):
    # Super compact implementation of nearest neighbour
    x = np.dot(test, train.transpose())
    modtest = np.sqrt(np.sum(test * test, axis=1))
    modtrain = np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose())  # cosine distance
    nearest = np.argmax(dist, axis=1)
    label = train_labels[nearest]
    
    return label

#The function used to find k nearest neighbours
def findKnn(train, test, train_labels, k):
    x = np.dot(test, train.transpose())
    modtest = np.sqrt(np.sum(test * test, axis=1))
    modtrain = np.sqrt(np.sum(train * train, axis=1))
    
    dist = x / np.outer(modtest, modtrain.transpose())  # cosine distance
    neighbours = np.argsort(-dist, axis=1)
    
    labels = []
#     scores = [] #unimplemented code
    for label in range(dist.shape[0]):
        k_neighbours_list = coll.Counter(train_labels[neighbours[label][:k]]).most_common(1)
        newKNNList = [ kn[0] for kn in k_neighbours_list ]
#         knnScores = [ kn[1] for kn in k_neighbours_list ]

        for n in newKNNList:
            labels.append(n)
            
#         for n in knnScores: #unimplemented code
#             scores.append(n)
    
    return labels #, scores #unimplemented code


# #The function used to find k nearest neighbours during reclassification
# def findKnnNew(fvectorTrain, fvectorTrain_labels, fvectorTest, train_labels, k, indexes):
#     newTrain = [fvectorTrain[x] for x in indexes]
#     newTrainArr = np.array(newTrain)
    
#     newTest = [fvectorTest[x] for x in indexes]
#     newTestArr = np.array(newTest)

#     newTrain_labels = [fvectorTrain_labels[i] for i in indexes]
#     new_train_labels = np.array(newTrain_labels)
    
#     orgTrain_labels = [train_labels[i] for i in indexes]
#     org_train_labels = np.array(orgTrain_labels)
 
    
#     x = np.dot(newTestArr, newTrainArr.transpose())
#     modtest = np.sqrt(np.sum(newTestArr * newTestArr, axis=1))
#     modtrain = np.sqrt(np.sum(newTrainArr * newTrainArr, axis=1))
    
#     dist = x / np.outer(modtest, modtrain.transpose())  # cosine distance
#     neighbours = np.argsort(-dist, axis=1)
    
#     scoresNew = []
#     labels = []
#     for label in range(dist.shape[0]):
#         knn_neighbour_list = coll.Counter(new_train_labels[neighbours[label][:k]]).most_common(1)
#         knnLables = [ kn[0] for kn in knn_neighbour_list ]
#         knnScores = [ kn[1] for kn in knn_neighbour_list ]
        
#         for s in knnScores:
#             scoresNew.append(s)

#         for l in knnLables:
#             labels.append(l)
        
#     scoresOld = []
#     for label in range(dist.shape[0]):
#         knn_neighbour_list_og = coll.Counter(org_train_labels[neighbours[label][:k]]).most_common(1)
#         knnScores = [ kn[1] for kn in knn_neighbour_list_og ]
    
#         for s in knnScores:
#             scoresOld.append(s)
                
#     return labels
    
# The functions below must all be provided in your solution. Think of them
# as an API that it used by the train.py and evaluate.py programs.
# If you don't provide them, then the train.py and evaluate.py programs will not run.
#
# The contents of these functions are up to you but their signatures (i.e., their names,
# list of parameters and return types) must not be changed. The trivial implementations
# below are provided as examples and will produce a result, but the score will be low.


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    The feature vectors are stored in the rows of 2-D array data, (i.e., a data matrix).
    The dummy implementation below simply returns the first N_DIMENSIONS columns.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
    vNpArr = np.array(model["v"])
    
    pcatrain_data = np.dot((data - np.mean(data)), vNpArr)
    return pcatrain_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Note, the contents of the dictionary are up to you, and it can contain any serializable
    data types stored under any keys. This dictionary will be passed to the classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    # The design of this is entirely up to you.
    # Note, if you are using an instance based approach, e.g. a nearest neighbour,
    # then the model will need to store the dimensionally-reduced training data and labels.
    model = {}
    
    covx = np.cov(fvectors_train, rowvar=0)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(N - N_DIMENSIONS, N - 1))
    v = np.fliplr(v)
    
    model["v"] = v.tolist()
    
    model["labels_train"] = labels_train.tolist()
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    
    return model


def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D feature vector array.

    In the feature vector array, each row corresponds to an image in the input list.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.

    Returns:
        np.ndarray: An 2-D array in which the rows represent feature vectors.
    """
    h, w = images[0].shape
    n_features = h * w
    fvectors = np.empty((len(images), n_features))
    for i, image in enumerate(images):
        fvectors[i, :] = image.reshape(1, n_features)

    return fvectors


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    Note, the feature vectors stored in the rows of fvectors_test represent squares
    to be classified. The ordering of the feature vectors is arbitrary, i.e., no information
    about the position of the squares within the board is available.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Get some data out of the model. It's up to you what you've stored in here
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    # Call the classify function.
    labels = classify(fvectors_train, labels_train, fvectors_test)

    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    The feature vectors for each square are guaranteed to be in 'board order', i.e.
    you can infer the position on the board from the position of the feature vector
    in the feature vector array.

    In the dummy code below, we just re-use the simple classify_squares function,
    i.e. we ignore the ordering.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    noOfBoards = fvectors_train.shape[0]//64
    noOfBoardsTest = fvectors_test.shape[0]//64

    boards = np.array_split(labels_train, noOfBoards)
    
    topRowForABoard = []
    middle6RowsForABoard = []
    bottomRowForABoard = []
    
    numRangeArr = np.arange(64)
    
    selectorTopEight = np.zeros(64, bool)
    selectorTopEight[numRangeArr%64 < 8] = True
    
    selectorBottomRow = np.zeros(64, bool)
    selectorBottomRow[numRangeArr%64 > 55] = True
    
    selectorMiddle6Rows = np.ones(64, bool)
    selectorMiddle6Rows[selectorTopEight] = False
    selectorMiddle6Rows[selectorBottomRow] = False
            
    top1RowForAllBoardsTrain = fvectors_train[np.tile(selectorTopEight, noOfBoards)]
    middle6RowsForAllBoardsTrain = fvectors_train[np.tile(selectorMiddle6Rows, noOfBoards)]
    bottomRowForAllBoardsTrainn = fvectors_train[np.tile(selectorBottomRow, noOfBoards)]
    
    top1RowForAllBoardTest = fvectors_test[np.tile(selectorTopEight, noOfBoardsTest)]
    middle6RowsForAllBoardLabelsTest = fvectors_test[np.tile(selectorMiddle6Rows, noOfBoardsTest)]
    bottomRowForAllBoardLabelsTest = fvectors_test[np.tile(selectorBottomRow, noOfBoardsTest)]
    
    top1RowForAllBoardLabels = labels_train[np.tile(selectorTopEight, noOfBoards)]
    middle6RowsForAllBoardLabels = labels_train[np.tile(selectorMiddle6Rows, noOfBoards)]
    bottomRowForAllBoardLabels = labels_train[np.tile(selectorBottomRow, noOfBoards)]
    
    k = 6
    #top rows
    knnTopRows = findKnn(top1RowForAllBoardsTrain, top1RowForAllBoardTest, top1RowForAllBoardLabels, k)
#     knnTopRows = nn(top1RowForAllBoardsTrain, top1RowForAllBoardTest, top1RowForAllBoardLabels)
    
    #middle 6 rows
    knnMiddle6Rows = findKnn(middle6RowsForAllBoardsTrain, middle6RowsForAllBoardLabelsTest, middle6RowsForAllBoardLabels, k)
#     knnMiddle6Rows = nn(middle6RowsForAllBoardsTrain, middle6RowsForAllBoardLabelsTest, middle6RowsForAllBoardLabels)
    
    #bottom rows
    knnBottomRows = findKnn(bottomRowForAllBoardsTrainn, bottomRowForAllBoardLabelsTest, bottomRowForAllBoardLabels, k)
#     knnBottomRows = nn(bottomRowForAllBoardsTrainn, bottomRowForAllBoardLabelsTest, bottomRowForAllBoardLabels)
    
    #combining labels back
    combiningLabels = np.zeros(fvectors_test.shape[0], str)
    combiningLabels[np.tile(selectorTopEight, noOfBoardsTest)] = knnTopRows
    combiningLabels[np.tile(selectorMiddle6Rows, noOfBoardsTest)] = knnMiddle6Rows
    combiningLabels[np.tile(selectorBottomRow, noOfBoardsTest)] = knnBottomRows
    
    labels = combiningLabels.tolist()

    return labels

#The following function is unimplemented since it returned the wrong scores during reclassification, which is why I did not proceed to use it
# def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
#     """Run classifier on a array of image feature vectors presented in 'board order'.

#     The feature vectors for each square are guaranteed to be in 'board order', i.e.
#     you can infer the position on the board from the position of the feature vector
#     in the feature vector array.

#     In the dummy code below, we just re-use the simple classify_squares function,
#     i.e. we ignore the ordering.

#     Args:
#         fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
#         model (dict): A dictionary storing the model data.

#     Returns:
#         list[str]: A list of one-character strings representing the labels for each square.
#     """
#     fvectors_train = np.array(model["fvectors_train"])
#     labels_train = np.array(model["labels_train"])

#     noOfBoards = fvectors_test.shape[0]/64 #25 boards
    
#     fvectorLabels = classify(fvectors_train, labels_train, fvectors_test)[0]
#     fvectorScores = classify(fvectors_train, labels_train, fvectors_test)[1]
    
#     #get the lowest scores from fvectorScores
#     minScoreValueInFvectorScores = min(fvectorScores)

#     indexesOfLeastScoreLabelsAfterClassify = []
#     #get the indexes of the scores returned
#     for index in range(len(fvectorScores)):
#         if fvectorScores[index] == minScoreValueInFvectorScores:
#             indexesOfLeastScoreLabelsAfterClassify.append(index)
            
#     print(indexesOfLeastScoreLabelsAfterClassify)
            
#     #for only those indexes on the boards check if rooks exist

#     boards = np.array_split(fvectorLabels, noOfBoards)
#     boards_length = len(boards)
    
#     rookLabels = []
#     #getting rook indexes
#     for board_number in range(boards_length):
#         board=boards[board_number]
#         rookCount = 0
#         rookIndexesOnBoard = []
#         for index,squareOnBoard in enumerate(board):
#             if(squareOnBoard == 'r' or squareOnBoard == 'R'): #ignoring the color of the pieces
#                 rookIndexesOnBoard.append((64*board_number)+index)
#                 rookCount = rookCount + 1
                
        
#         indexesOfLeastScoreLabelsAfterClassify_as_set = set(indexesOfLeastScoreLabelsAfterClassify)
#         commonIndexesRook = indexesOfLeastScoreLabelsAfterClassify_as_set.intersection(rookIndexesOnBoard)
#         commonIndexesRookList = list(commonIndexesRook)


#         if( rookCount > 4 ):
#             if commonIndexesRookList != []:
#                 knnNewRooks = findKnnNew(fvectors_train, fvectorLabels, fvectors_test, labels_train, 10, commonIndexesRookList)

#     return classify_squares(fvectors_test, model)