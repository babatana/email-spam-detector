"""=========================================================================
A corpus is just a list of instances.  Each instance is a 
object containing a dictionary keyed by the attribute names and containing the
attribute  values.  The category name and membership value are kept
separately. All this is packaged up in a class with helpful methods.

Author: Susan Fox
Date: Fall 2015 (latest modifications)
"""



class Instance:
    """An instance represents one exemplar from a dataset (one row of the table). It
    has a fixed attribute for categorization: category, and its category value, and
    then a set of attributes and their values. Really this is just a shell to 
    allow meaningful method names to access the parts of an instance."""
    
    
    def __init__(self, catName, catValue, attrNames, attrs):
        """Given the category name, and value, a list of attribute names
        and associated values, it stores the data away"""
        self.category = catName
        self.value = catValue
        self.attrNames = attrNames
        self.attributes = attrs

    def getCategory(self):
        """Return the category for the classification task for this instance."""
        return self.category

    def getCatValue(self):
        """Return the value of classification attribute for this instance."""
        return self.value

    def getFeatureValue(self, featName):
        """Return the value of a given feature/attribute of this instance."""
        if featName == self.category:
            return self.value
        else:
            return self.attributes[featName]


    def getAttribs(self):
        """Returns a copy of the attribute dictionary."""
        return self.attributes.copy()

    def __str__(self):
        """Overrides a built-in method and generates a meaningful printable form of this data."""
        descrString = "Category value: " + str(self.value)
        descrString += " [ "
        for a in self.attrNames:
            descrString += str(self.attributes[a]) + ", "
        descrString = descrString[0:-2]  # cut off last two, the ", " that isn't followed by another value
        descrString += ']'
        return descrString




class Corpus:
    """Represents a dataset read from a file."""
    
    def __init__(self, fileName):
        """Given a filename containing a properly formatted dataset, this reads
        in the dataset and represents it. It has methods for ..."""
        
        self.sourceFile = fileName
        self.corpus = []
        self.category = ""
        self.categValues = []
        self.attributeInfo = {}
        self._readData(fileName)


    def _readData(self, fileName):
        """Private method that reads in the dataset from the file. The first
        line contains the category and its possible values. The next line ..."""
        fPort = open(fileName, 'r')
        # read in main category and its values
        categLine = fPort.readline()
        categList = categLine.split()
        self.category = categList[0]
        self.categValues = categList[1:]
        
        # read in attribute names and attribute values (must be nominal)
        self.attrNames = []
        for nextLine in fPort:
            if nextLine.strip() == "":  # if next line is blank, then break
                break
            nextList = nextLine.split()
            self.attrNames.append(nextList[0])
            self.attributeInfo[nextList[0]] = nextList[1:]

        # read in instances until line contains -----
        for nextLine in fPort:
            if nextLine.strip() == '-----':
                break
            nextList = nextLine.split()
            instanceCateg = nextList[len(nextList) - 1]
            instDict = {}
            for i in range(0,len(nextList) - 1):
                instDict[self.attrNames[i]] = nextList[i]
            nextInstance = Instance(self.category, instanceCateg, self.attrNames,instDict)
            
            self.corpus.append(nextInstance)
        fPort.close()


    def printData(self):
        """Prints out the dataset in a simple but readable way."""
        print ("Category", self.category)
        print ('  values:', self.categValues)
        print (self.attributeInfo)
        for inst in self.corpus:
            print(inst)
            

    def getCategory(self):
        """Returns the attribute used for classification in this dataset."""
        return self.category

    def getCategoryValues(self):
        """Returns a copy of the category values list."""
        return self.categValues[:]

    def getAttributes(self):
        """Returns a copy of the list of attribute names."""
        return self.attrNames[:]

    def getAttributeValues(self, attr):
        """Given an attribute, returns a copy of the list of values associated with that attribute."""
        attrVals = self.attributeInfo[attr]
        return attrVals[:]

    def getAttributeMap(self):
        """Returns a copy of the mapping of attribute names to their potential values."""
        return self.attributeInfo.copy()

    def getInstances(self):
        """Returns a copy of the corpus list. Does NOT copy the Instances that make it up."""
        return self.corpus[:]
# end class Corpus


        


if __name__ == '__main__':
    newCorp = Corpus("PythonDatasets/dog1.dat")
    newCorp.printData()
    
    corp2 = Corpus("PythonDatasets/people2.dat")
    corp2.printData()
    