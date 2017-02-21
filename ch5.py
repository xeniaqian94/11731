import dynet as dy #automatically initializes the gloabl dynet parameters
import random
# Parameters of the model and training
HIDDEN_SIZE = 20
NUM_EPOCHS = 20
# Define the model and SGD optimizer
model = dy.Model()
W_xh_p = model.add_parameters((HIDDEN_SIZE, 2))
b_h_p = model.add_parameters(HIDDEN_SIZE)
W_hy_p = model.add_parameters((1, HIDDEN_SIZE))
b_y_p = model.add_parameters(1)
trainer = dy.SimpleSGDTrainer(model)
# Define the training data, consisting of (x,y) tuples
data = [([1,1],1), ([-1,1],-1), ([1,-1],-1), ([-1,-1],1)]
# Define the function we would like to calculate
def calc_function(x):
   dy.renew_cg() #reset the computation graph to a new state by calling renew_cg().
   w_xh = dy.parameter(W_xh_p)
   b_h = dy.parameter(b_h_p)
   W_hy = dy.parameter(W_hy_p)
   b_y = dy.parameter(b_y_p)
   x_val = dy.inputVector(x)
   h_val = dy.tanh(w_xh * x_val + b_h) 
   y_val = W_hy * h_val + b_y
   return y_val
#Perform training
for epoch in range(NUM_EPOCHS):
   epoch_loss = 0
   random.shuffle(data)
   for x, ystar in data:
      y = calc_function(x)
      print("%r -> %f" % (x, y.value()))
      loss = dy.squared_distance(y, dy.scalarInput(ystar))
      epoch_loss += loss.value()
      loss.backward()
      trainer.update()
   print("Epoch %d: loss=%f" % (epoch, epoch_loss))
for x, ystar in data:
   y = calc_function(x)
   print("%r -> %f" % (x, y.value()))