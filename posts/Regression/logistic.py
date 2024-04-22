import torch
import numpy as np 

class LinearModel:

    def __init__(self):
        self.w = None 
        self.prevWeight = None

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))
        s = torch.matmul(X, self.w)
        #or like this 
        # scores = (X @ self.w)
        s = s.float()
        return s

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """

        threshold = 0 

        #get the scores from prev funct
        scores = self.score(X)

        #assigns 1 if score>=0.5, 0 if <0.5
        y_hat = torch.where(scores >= threshold, 1.0, 0.0)

        return y_hat
    

# then implement logistic regression 
class LogisticRegression(LinearModel):
     
    def loss(self, X, y):
            """
            Compute the misclassification rate. In the perceptron algorithm, the target vector y is assumed to have labels in {-1, 1}. A point i is classified correctly if its score s_i has the same sign as y_i. 

            ARGUMENTS: 
                X, torch.Tensor: the feature matrix. X.size() == (n, p), 
                where n is the number of data points and p is the 
                number of features. This implementation always assumes 
                that the final column of X is a constant column of 1s. 

                y, torch.Tensor: the target vector.  y.size() = (n,). In the perceptron algorithm, the possible labels for y are assumed to be {-1, 1}
            """
            scores = self.score(X) 
            
            sigmoid = torch.sigmoid(scores) 

            loss = -y * torch.log(sigmoid) - (1-y) * torch.log(1-sigmoid)
            finalLoss = torch.mean(loss)

            return finalLoss 
    
    def grad(self, X, y):
        """
        Should compute the gradient of the empirical risk
        
        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 
            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}

        RETURNS:
            grad: float: the gradient of the empirical risk of the model
        """
 
        scores = self.score(X)
        sigmoid = torch.sigmoid(scores)
        #convert tensor shape to avoid loops 
        grad = (sigmoid -y)[:, None] * X
        finalGrad = torch.mean(grad, dim =0)
        
        return finalGrad
    

class GradientDescentOptimizer: 

    def __init__(self, model):
        self.model = model
    
    def step(self, X, y, alphaRate, betaRate):

        loss = self.model.loss(X, y)
        grad = self.model.grad(X, y)

        #save current weights 
        currW = self.model.w.clone()
        
        #first update, no previous w 
        if self.model.prevWeight == None: 
            self.model.w -= alphaRate * grad
        else: 
            self.model.w += -1*alphaRate*grad + betaRate*(self.model.w - self.model.prevWeight.clone())

        self.model.prevWeight = currW.clone()
        
        return loss
   