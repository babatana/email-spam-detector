""" ==============================================================
Defines a decision tree class that builds and uses a decision tree.
It has a hole in it: no definition of ID3 or anything fancier.
The overall decision tree is a wrapper of sorts that handles the
whole-tree operations, and lets the basic stuff be handled by
the treeNode class
    ==============================================================
"""

from corpus import *
import math 

class DecisionTree:
    """Represents a decision tree, and various utilities for accessing
    its data. Missing an implementation of ID3."""
    
    
    def __init__(self, dataset):
        """Given a Dataset object, it sets up all the internal information
        so that it is ready to build the tree."""
        self.treeNode = None
        self.dataset = dataset
        self.instances = dataset.getInstances()
        self.attributes = dataset.getAttributes()
        self.attributeMap = dataset.getAttributeMap()
        self.categoryValues = dataset.getCategoryValues()
        self.category = dataset.getCategory()



    def searchTree(self, newInstance):
        """Once the tree is built, this will take a new Instance object
        and will determine the tree's classification of that instance.
        Returns False if no tree exists yet."""
        currTree = self.treeNode
        while currTree != None:
            if currTree.isLeaf():
                return currTree.getNodeValue()
            currFeat = currTree.getNodeValue()
            currTree = currTree.getChild(newInstance.getFeatureValue(currFeat)) 
        return False



    def buildTree(self):
        """Takes in no inputs, and uses ID3 to build the tree. Not implemented
        yet."""

        yes = 0
        no = 0

        for inst in self.instances:
            if inst.getCatValue() == "yes":
                yes += 1
            else:
                no += 1

        aTree = self.ID3(self.instances, self.attributes, max([yes, no]))
        aTree.doPrintTree(1)

    def ID3(self, instances, attributes, defaultValue):

        if self.areConsistent(instances):
            val = ""
            for inst in instances:
                val = inst.getCatValue()
            return TreeNode(val)

        elif not attributes:

            yes = 0
            no = 0

            for inst in instances:
                if inst.getCatValue() == "yes":
                    yes += 1
                else:
                    no += 1

            # return max([yes, no])

            if yes > no:
                return TreeNode("yes")
            elif yes == no:
                return TreeNode("???")
            else:
                return TreeNode("no")
        else:
            temp = 0 # highest gain
            best = "" # best attribute of string type
            for attr in attributes:
                g = self.gain(attr, instances)
                if temp < g:
                    temp = g
                    best = attr

            qNode = TreeNode(best, self.attributeMap[best])

            yes = 0
            no = 0
            for inst in instances:
                if inst.getCatValue() == "yes":
                    yes += 1
                else:
                    no += 1
            maj = max([yes, no])

            for v in self.attributeMap[best]:
                newExamp = self.getSamplesWithValue(best, v, instances)
                if best in attributes:
                    attributes.remove(best)
                newAttr = attributes.copy()
                subtree = self.ID3(newExamp, newAttr, maj)
                qNode.addChild(v, subtree)
            return qNode

    def getSamplesWithValue(self, attribute, attrValue, dataset):
        """This function takes a string that is an attribute, a specific value
        for that attribute, and a list of instances and it returns those instances
        that have the given value for that attribute"""
        samples = []
        for sample in dataset:
            if sample.getFeatureValue(attribute) == attrValue:
                samples.append(sample)
        return samples
    


    def areConsistent(self, dataset):
        """This takes in a dataset and returns True if all instances have the
    same value for the category we want to learn"""
        if dataset == []:
            return True
        cat = dataset[0].getCatValue()
        for inst in dataset:
            if not (inst.getCatValue() == cat):
                return False
        return True
    

    # ----------------------------------------------
    def gain(self, attribute, instances):
        """Give this a string which is an attribute and a dataset of instances we are
        asking about (which is a list of dictionaries), this will calculate and return
        the gain associated with choosing this particular attribute for the root of
        the current tree"""
        entropyProbs = self.genEntropyProbs(instances)
        wholeEntropy = self.entropy(entropyProbs)
        remain = self.remainder(attribute, instances)
        return wholeEntropy - remain



    def genEntropyProbs(self, dataset):
        """A helper for gain, this calculates a list of entropy probabilities"""
        entropyProbs = []
        for cat in self.categoryValues:
            catset = self.getSamplesWithValue(self.category, cat, dataset)
            entropyProbs.append(len(catset) / len(dataset))
        return entropyProbs


      
    def entropy(self, probList):
        """Given a list of probabilities, compute the entropy value for them"""
        sum = 0
        for prob in probList:
            if prob != 0:
                sum += -1 * prob * math.log(prob, 2)
        return sum


    def remainder(self, attribute, dataset):
        """A helper for gain... computes the remainder for a given attribute"""
        total = 0
        for attrVal in self.attributeMap[attribute]:
            sampleset = self.getSamplesWithValue(attribute, attrVal, dataset)
            if len(sampleset) > 0:
                term1 = len(sampleset) / len(dataset)
                entropyProbs = self.genEntropyProbs(sampleset)
                total += term1 * self.entropy(entropyProbs)
        return total
    
    
    
# end class decisionTree


class TreeNode:
    """Here a tree has a value, which should be a string and represents
    a feature type -- the question, or the ultimate category answer 
    if the tree is a leaf.  It also contains a dictionary organized by 
    the feature values that are assigned to that feature type.  The
    value associated with each feature value is another decision tree node."""

    def __init__(self, featName, featValues = []):
        """takes in the name of the feature and a list of values, and it initializes
        any answers for this node"""
        self.nodeValue = featName
        self.featureValues = featValues[:]
        self.answers = {} # dictionary built from featValues
        for fv in featValues:
            self.answers[fv] = None

    def getNodeValue(self):
        """Access the node's value"""
        return self.nodeValue


    def getAnswerValues(self):
        """Returns a list of the answer values"""
        return self.featureValues

    def getChild(self, featValue):
        """Returns a specific subtree given the feature value for it"""
        return self.answers[featValue]

    def addChild(self, featValue, newKid):
        """Adds a new subtree for a given feature value"""
        if featValue != self.featureValues:
            self.featureValues.append(featValue)
        self.answers[featValue] = newKid

    def isLeaf(self):
        """A node is a leaf if there are no answers/feature values associated with it"""
        return self.answers == {}

    # ----------------------------------------
    def printTree(self):
        self.doPrintTree(0)
 
    def doPrintTree(self, indentSize):
        indent = ' ' * indentSize
        if self.isLeaf():
            print(indent, 'Answer:', self.getNodeValue())
        else:
            print (indent, "Question:", self.getNodeValue(), list(self.answers.keys()))
            for a in self.answers:
                print (indent, "-----Value:",  a)
                subtree = self.getChild(a)
                subtree.doPrintTree(indentSize + 5)

         
# end class treeNode

    

           

    