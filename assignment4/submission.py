
import numpy as np
from collections import Counter
import time


class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a decision function to select between left and right nodes.
        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.
        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        # """Get a child node based on the decision function.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
        # Args:͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
        #     feature (list(int)): vector for feature.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
        # Return:͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
        #     Class label if a leaf node, otherwise a child node.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
        # """͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
        """Determine recursively the class of an input array by testing a value
           against a feature's attributes values based on the decision function.

        Args:
            feature: (numpy array(value)): input vector for sample.

        Returns:
            Class label if a leaf node, otherwise a child node.
        """
        #print("here")
        if self.class_label is not None:
            
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if(class_index == -1):
        classes= out[:,class_index]
        features = out[:,:class_index]
        return features, classes
    elif(class_index == 0):
        classes= out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the sample data.
    Tree is built fully starting from the root.

    Returns:
        The root node of the decision tree.
    """

    dt_root = None
    # TODO: finish this.
    
    #add the function nodes 
    
    #A0
    func0 = lambda feature : feature[0] == 1
    #A2
    func2 = lambda feature : feature[2] == 1
    func3 = lambda feature : feature[3] == 1
    
    
    
    
    #build the nodes
    dt_root = DecisionNode(None, None, func0, None)
    dt_A3 = DecisionNode(None, None, func2, None)
    dt_A4 = DecisionNode(None, None, func3, None)
    dt_A4L = DecisionNode(None, None, func3, None)
    
    #build the tree
    dt_root.left = DecisionNode(None, None, None, 1)
    dt_root.right = dt_A3
    
    dt_A3.left = dt_A4L
    dt_A3.right = dt_A4
    
    dt_A4.left = DecisionNode(None, None, None, 0)
    dt_A4.right = DecisionNode(None, None, None, 1)
    
    dt_A4L.left = DecisionNode(None, None, None, 1)
    dt_A4L.right = DecisionNode(None, None, None, 0)
    
    
    #raise NotImplemented()
    return dt_root


def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.

    Classifier output vs true labels, which is equal to:
    Predicted  vs  Actual Values.

    Output will in the format:

                        |Predicted|
    |T|                
    |R|    [[true_positive, false_negative],
    |U|    [false_positive, true_negative]]
    |E|

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        A two dimensional array representing the confusion matrix.
    """

    c_matrix = np.zeros([2, 2])
    # TODO: finish this.
    
    #range is how you do a while loop, we loop len(true_labels) times since thats how long the outputs are
    for i in range(len(true_labels)):
        #take the value, and increment it in the correct spot, since all other values will be 0
        c_matrix[classifier_output[i]][true_labels[i]] +=1
        
    #since this matrix wants the top right to be the 1,1 coordinate, we must flip the diagonal since we have:
    #[(0,0) (0,1)]
    # (1,0) (1,1)
    #i originally did the grad assignment so I just reused the code, which was set up this way
    
    temp = c_matrix[0][0]
    c_matrix[0][0] = c_matrix[1][1]
    c_matrix[1][1] = temp
    
    #print(c_matrix)
    return c_matrix


def precision(true_labels, classifier_output):
    """Get the precision of a classifier compared to the correct values.
    Precision is measured as:
        true_positive/ (true_positive + false_positive)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The precision of the classifier output.
    """

    # TODO: finish this.
    #raise NotImplemented()
    
    
    matrix = confusion_matrix(classifier_output, true_labels)
    #print(true_labels)
    #print(classifier_output)
    #print("matrix", matrix)
    #print(n_classes)
    
    #print("output:", matrix[0][0]/(matrix[0][0] + matrix[1][0]))
    return matrix[0][0]/(matrix[0][0] + matrix[0][1])


def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.
    Recall is measured as:
        true_positive/ (true_positive + false_negative)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The recall of the classifier output.
    """

    matrix = confusion_matrix(classifier_output, true_labels)
    #print(true_labels)
    #print(classifier_output)
    #print("matrix", matrix)
    #print(n_classes)
    
    #print("output:", matrix[0][0]/(matrix[0][0] + matrix[1][0]))
    return matrix[0][0]/(matrix[0][0] + matrix[0][1])


def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.
    Accuracy is measured as:
        correct_classifications / total_number_examples
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The accuracy of the classifier output.
    """

    # TODO: finish this.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
    matrix = confusion_matrix(classifier_output, true_labels)
    #print(true_labels)
    #print(classifier_output)
    #print("matrix", matrix)
    #print(n_classes)
    
    #print("output:", matrix[0][0]/(matrix[0][0] + matrix[1][0]))
    return (matrix[0][0] + matrix[1][1])/(matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1])


def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.
    Returns:
        Floating point number representing the gini impurity.
    """
    #raise NotImplemented()
    #print(class_vector)
    counts = Counter(class_vector)
    #print(counts)
    impurity = 0 
    for key in counts:
        count = counts[key]
        p_i = count/len(class_vector)
        #print(p_i)
        p_squared = p_i * p_i
        
        impurity+=p_squared
        
    #print(prob)
    return 1 - impurity
    
    


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    
    parent_impurity = gini_impurity(previous_classes)
    #print("parent:", previous_classes, "impurity", parent_impurity)
    #print("child:", current_classes)
    
    total = 0
    for list in current_classes:
        impurity = gini_impurity(list)
        total += impurity * (len(list)/len(previous_classes))
    #print(total)
        
    return parent_impurity - total
    


class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        #print("hello")
        self.root = self.__build_tree__(features, classes)
        #print(self.root)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """

        # TODO: finish this.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
        #raise NotImplemented()
        #print(features)
        #print("---------------------------------------------------------------------------------------------------")
        #print(classes)
        
        #print("depth", depth)
        #print("depth_limit", self.depth_limit)
        if np.all(np.array(classes)) == 1:
            #raise NotImplemented()
            #print("CASE 1")
            #print("--------------------------------------------------------------------------------------------------------------------------")
            #case where all the values are the same 
            if classes[0] == 1:
                return DecisionNode(None, None, None, 1)
            else:
                return DecisionNode(None, None, None, 0)
        
        if depth >= self.depth_limit:
            #print("CASE 2")
            #print("--------------------------------------------------------------------------------------------------------------------------")
            count = Counter(classes)
            #print("MOST COMMON", count.most_common(1)[0][0])
            return DecisionNode(None, None, None, count.most_common(1)[0][0])
        if len(classes) == 0 or len(classes) == 1:
            #print("CASE 3")
            #print("--------------------------------------------------------------------------------------------------------------------------")
            return DecisionNode(None, None, None, None)
        
        alpha_best = None
        alpha_best_value = -1
        left_list = []
        right_list = []
        threshold = 0

        #for each attribute, we want to split by attriute (grab each column)
        #since classes has the same amount of cols, we want to iterate from 0 -> len(classes)
        for i in range(len(features[0])):
            
            #grab a column alpha
            alpha = features[:, i]
            
            #threshold = len(alpha)//2
            #print("threshold", threshold)
            #print(alpha)
            #we need the two lists we will split into (positive and negative since threshold is 0)
            alpha_neg = []
            alpha_pos = []
            
            # I will use the threshold of below zero 
            for j in range(len(alpha)):
                if alpha[j] < threshold:
                    #negative case
                    #print("neg", alpha[j])
                    alpha_neg.append(j)
                elif alpha[j] > threshold:
                    #greater than or equal to 0 (to account for edge case of alpha = threshold)
                    #print("pos", alpha[j])
                    alpha_pos.append(j)
            
            #now we need to evaluate ginigain for alpha 
            #we must convert our list of indices to the sublists
            child_1 = []
            child_2 = []
            for j in range(len(alpha_neg)):
                child_1.append(classes[alpha_neg[j]])
            for j in range(len(alpha_pos)):
                child_2.append(classes[alpha_pos[j]])
            
            gain = gini_gain(classes, [child_1, child_2])
            
            #update alpha_best and value if they are better than the current ones
            #print("gain", gain)
            if gain > alpha_best_value:
                #print("here")
                alpha_best = i
                alpha_best_value = gain
                left_list = alpha_neg
                right_list = alpha_pos
                exact_feature = alpha
        
        #print(alpha_best)
        ##print(alpha_best_value)
        #print("left list: ", left_list)
        #print("right list: ", right_list)
        #return
        
        if len(left_list) == 0 or len(right_list) == 0:
            #print("CASE 4")

            count = Counter(classes)
            #print("MOST COMMON", count.most_common(1)[0][0])
            #print("MOST COMMON", count.most_common(1)[0][0])
            #print("--------------------------------------------------------------------------------------------------------------------------")
            return DecisionNode(None, None, None, count.most_common(1)[0][0])
        
        #after that giant loop, alpha_best should hold the index of the best attribute to split by, with the left and right lists being in indexes of the split
        #now we must actually perform the split on both features and classes, and perform recursion on the sublists
        left_features = []
        left_classes = []
        for index in left_list:
            #print("iteration")
            left_classes.append(classes[index])
            x = features[index, :]
            #print(x)
            if len(left_features) == 0:
                left_features = x
            else:
                left_features = np.vstack((left_features, x))
        
        right_features = []
        right_classes = []
        for index in right_list:
            right_classes.append(classes[index])
            x = features[index, :]
            #right_features = np.vstack((right_features, x))
            if len(right_features) == 0:
                right_features = x
            else:
                right_features = np.vstack((right_features, x))
        
        
        #(left_classes)
        #(len(left_classes))
        #(right_classes)
        #(len(right_classes))
        #("--------------------------------------------------------------------------------------------------------------------------")
        #return
        #now that we have the sublists and features, we want to perform recursion on their respective sides to build a node
        l_node = self.__build_tree__(left_features, left_classes, depth+1)
        r_node = self.__build_tree__(right_features, right_classes, depth+1)
        
        #("final threshold", threshold)
        #when recursions return we actually build the tree
        
        #("alpha", alpha_best)
        #("L", l_node)
        #("R", r_node)
        #("feature", exact_feature[alpha_best])
        #("threshold", threshold)
        #go left if below 0
        return DecisionNode(l_node, r_node, lambda feature: feature[alpha_best] < 0)
        
            
        
    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """
        #print(features)
        class_labels = []
        for feature in features:
            #print(feature)
            class_labels.append(self.root.decide(feature))
        # TODO: finish this.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
        #raise NotImplemented()
        return class_labels


class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """

        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        # TODO: finish this.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
        #raise NotImplemented()
        #print(self.trees)
        #this represents the amount that we are going to resample
        example_subsample_amount = int(len(features[0]) * self.example_subsample_rate)
        attr_subsample_amount = int(len(features[0]) * self.attr_subsample_rate)
        
        
        #print(attr_subsample_amount)
        for i in range(self.num_trees):
            #first sample the features, where the generated number is the amount of rows
            feature_index = np.random.choice(len(features))
            
            for i in range(example_subsample_amount):
                
                #attr_choice = np.random.choice(len(classes))
                new = features[feature_index]
                #print(new)
                for i in range(attr_subsample_amount):
                    attr_choice = np.random.choice(len(features[feature_index]))
                    index_choice = np.random.choice(len(features[feature_index]))
                    
                    new[index_choice] = features[feature_index][attr_choice]
            
            
            features[feature_index] = new
            #now sampling is donw, we must generate the tree 
            d_tree = DecisionTree(self.depth_limit)
            d_tree.fit(features, classes)
            self.trees.append(d_tree)
                

    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (m x n): m examples with n features.
        """

        #print(features)
        class_labels = []
        for tree in self.trees:
            #print(feature)
            class_labels.append(tree.classify(features))
        
        index = len(class_labels)//2
        #middle value will be the most ideal, even though any of them will be valid in the list
        return class_labels[index]

class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be added to array.
        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Element wise array arithmetic using vectorization.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be sliced and summed.
        Returns:
            Numpy array of data.
        """

        # TODO: finish this.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
        #numpy dot function takese in a matrix and does the dot product with the second (does matrix mult if they are larger than 2d)
        #numpy_data = np.matrix(data)
        #print(numpy_data)
        vectorized = (data * data) + data
        #print(vectorized)
        return vectorized

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be added to array.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be sliced and summed.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        # TODO: finish this.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
        #first we want to grab the first 100 rows and put it into a smaller matrix
        rows = data[:100, :]
        #next we want the sum of each of these rows, we need the axis to be 1 so that it adds horizontally
        row_sums = np.sum(rows, axis=1)
        #now we want to grab the max from the row_sums array (the output will be the index of the row with the highest sum)
        max_sum_index = np.argmax(row_sums)
        #now we return the max row sum and its index
        #print(max_sum_index)
        return tuple((row_sums[max_sum_index], max_sum_index))

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        # TODO: finish this.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
        data.flatten()
        
        data = data[data>0]
        output, count = np.unique(data.astype(int), return_counts=True)
        solution = dict(zip(output, count))
        #print("vector", solution)
        return solution.items()
    
    
    def non_vectorized_glue(self, data, vector, dimension='c'):
        """Element wise array arithmetic with loops.
        This function takes a multi-dimensional array and a vector, and then combines
        both of them into a new multi-dimensional array. It must be capable of handling
        both column and row-wise additions.
        Args:
            data: multi-dimensional array.
            vector: either column or row vector
            dimension: either c for column or r for row
        Returns:
            Numpy array of data.
        """
        if dimension == 'c' and len(vector) == data.shape[0]:
            non_vectorized = np.ones((data.shape[0],data.shape[1]+1), dtype=float)
            non_vectorized[:, -1] *= vector
        elif dimension == 'r' and len(vector) == data.shape[1]:
            non_vectorized = np.ones((data.shape[0]+1,data.shape[1]), dtype=float)
            non_vectorized[-1, :] *= vector
        else:
            raise ValueError('This parameter must be either c for column or r for row')
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row, col] = data[row, col]
        return non_vectorized

    def vectorized_glue(self, data, vector, dimension='c'):
        """Array arithmetic without loops.
        This function takes a multi-dimensional array and a vector, and then combines
        both of them into a new multi-dimensional array. It must be capable of handling
        both column and row-wise additions.
        Args:
            data: multi-dimensional array.
            vector: either column or row vector
            dimension: either c for column or r for row
        Returns:
            Numpy array of data.
            
        """
        # TODO: finish this.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
        #print(vector)
        vectorized = None
        
        if dimension == 'c' and len(vector) == data.shape[0]:
            #vector.reshape(1, -1)
            vectorized = np.hstack((data, vector.reshape(len(vector),-1)))
        elif dimension == 'r' and len(vector) == data.shape[1]:
            #vector.reshape(-1, 1)
            vectorized = np.vstack((data, vector.reshape(-1,len(vector))))
        
        return vectorized

    def non_vectorized_mask(self, data, threshold):
        """Element wise array evaluation with loops.
        This function takes a multi-dimensional array and then populates a new
        multi-dimensional array. If the value in data is below threshold it
        will be squared.
        Args:
            data: multi-dimensional array.
            threshold: evaluation value for the array if a value is below it, it is squared
        Returns:
            Numpy array of data.
        """
        non_vectorized = np.zeros_like(data, dtype=float)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                val = data[row, col]
                if val >= threshold:
                    non_vectorized[row, col] = val
                    continue
                non_vectorized[row, col] = val**2

        return non_vectorized

    def vectorized_mask(self, data, threshold):
        """Array evaluation without loops.
        This function takes a multi-dimensional array and then populates a new
        multi-dimensional array. If the value in data is below threshold it
        will be squared. You are required to use a binary mask for this problem
        Args:
            data: multi-dimensional array.
            threshold: evaluation value for the array if a value is below it, it is squared
        Returns:
            Numpy array of data.
        """
        vectorized = None
        #raise NotImplemented()
        new_list = np.copy(data)
        new_list[data < threshold] = np.square(new_list[data < threshold])
        return new_list

def return_your_name():
    # return your name͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
    # TODO: finish this͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏️͏︌͏󠄃
    return "Shreshta Yadav"