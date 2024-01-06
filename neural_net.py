from loss import *

class NeuralNetwork:
    def __init__(self, layers, alpha, epochs):
        self.layers = layers
        self.num_layers = len(layers)
        self.alpha = alpha
        self.epochs = epochs
        self.W = []
        
    def initialize(self):
        for x,y in self.layers:
            w = np.random.randn(x+1, y)* np.sqrt(2.0 / (x + 1))

            self.W.append(w)
            
    def forward_prop(self, image):
        prediction = []
        curr_output = image
        for i in range(self.num_layers):
            curr_output = np.dot(curr_output, self.W[i][:-1]) + self.W[i][-1]
            if(i != self.num_layers - 1):
                curr_output = sigmoid(curr_output)
            else:
                curr_output = sigmoid(curr_output)
            prediction.append(curr_output)
        return prediction

    def back_prop(self, prediction, y):
        target = [0]*10
        target[int(y)] = 1
        error = prediction[-1] - target
        deltas = []
        deltas.append(error * sigmoid_derivative(prediction[-1]))
        for i in reversed(range(1, self.num_layers)):
            delta = deltas[-1].dot(self.W[i][:-1].T) * sigmoid_derivative(prediction[i-1])
            deltas.append(delta)
        deltas.reverse()
        for i in range(self.num_layers):
            self.W[i] += -self.alpha * prediction[i].T.dot(deltas[i])
            
    
    def loss(self, X, y):
        target = [0]*10
        target[int(y)] = 1
        predictions = self.forward_prop(X)[-1]
        # loss = 0.5 * np.sum((predictions - target) ** 2)
        loss = cross_entropy_loss(predictions, target)
        # print(predictions, target, loss)
        return loss

    def calculate_loss(self, test_X, test_y):
        loss = 0
        for i in range(len(test_X)):
            loss += self.loss(test_X[i], test_y[i])
        return loss
        
    def predict(self, image):
        return self.forward_prop(image)[-1].argmax()

    
    def train(self, train_X, train_y, test_X, test_y):
        losses = []
        train_len = len(train_X)
        for epoch in range(self.epochs):
            train_num = 0 
            for i in range(train_len):
                prediction = self.forward_prop(train_X[i])
                self.back_prop(prediction, train_y[i])
                if(train_num == 2000):
                    losses.append(self.calculate_loss(test_X, test_y))
                    print(f'train_num: {i}/{train_len}, loss: {losses[-1]}')
                    train_num = 0
                train_num +=1
            print(f'epoch: {epoch}, loss: {losses[-1]}')
        return losses
    
    def test(self, test_set, train_sol):
        total = len(test_set)
        correct = 0
        d = {}
        for i in range(len(test_set)):
            prediction = self.predict(test_set[i])
            if prediction not in d:
                d[prediction] = 0
            d[prediction] += 1
            if self.predict(test_set[i]) == int(train_sol[i]):
                correct += 1
        return correct/total, d
            
        
            