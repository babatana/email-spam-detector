"""
This file contains a couple of 'helper' printing functions,
and an example showing how to read in a datasets, create a decision
tree, and use the two to compute gains and divide the dataset."""


from corpus import *
from DTree2015 import *
import math 





def printInstances(instances):
    """Print instances"""
    for inst in instances:
        print(inst)


def printGains(attribList, instList, decisTree):
    """Given a list of attributes of interest, and a list
    of Instances, and a decision tree, uses the decision tree's gain
    calculator to determine the gain for each listed attribute, and
    prints them."""
    formStr = "Gain for attribute {0:8s} = {1:6.3f}"
    for attr in attribList:
        g = decisTree.gain(attr, instList)
        currStr = formStr.format(attr, g)
        print(currStr)


# ========================================================================

# First, read in the dataset and build the dTree object (which
# can't actually build the tree, but does have functions for
# computing gain and splitting up datasets).
corp = Corpus("PythonDatasets/cars.dat")
dTree = DecisionTree(corp)
dTree.buildTree()


# # Next, shows how to get the attributes and the list of Instances
# # out of the corpus, and how to print the gains across all attributes
# # NOTE: the attribute list does not include the category!
# print("============================================")
# attrs = corp.getAttributes()
# instances = corp.getInstances()
# printGains(attrs, instances, dTree)


# # Next, an example of how to get a subset of the data, how to print
# # a list of instances in a way that is quasi-readable, and how to 
# # check if a list of instances has a single value
# print("============================================")
# smallDogs = dTree.getSamplesWithValue("passengers", '5', instances)
# printInstances(smallDogs)
# allSame = dTree.areConsistent(smallDogs)
# print("All small dogs are the same category:", allSame)
# print("============================================")

# attrs.remove("passengers")
# printGains(attrs, smallDogs, dTree)

# print("============================================")

# smallDogs = dTree.getSamplesWithValue("safety", 'high', smallDogs)
# printInstances(smallDogs)
# allSame = dTree.areConsistent(smallDogs)
# print("All small dogs are the same category:", allSame)

# print("============================================")

# attrs.remove("safety")
# printGains(attrs, smallDogs, dTree)

# print("============================================")

# smallDogs = dTree.getSamplesWithValue("maintPrice", 'high', smallDogs)
# printInstances(smallDogs)
# allSame = dTree.areConsistent(smallDogs)
# print("All small dogs are the same category:", allSame)

# print("============================================")

# attrs.remove("maintPrice")
# printGains(attrs, smallDogs, dTree)

# print("============================================")

# smallDogs = dTree.getSamplesWithValue("purchasePrice", 'moderate', smallDogs)
# printInstances(smallDogs)
# allSame = dTree.areConsistent(smallDogs)
# print("All small dogs are the same category:", allSame)

# print("============================================")




