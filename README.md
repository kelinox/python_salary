# Linear regression algorithm to predict salary from experience
This simple example shows how to solve a simple problem using the machine learning model of linear regression

## Getting started
Pull the repository and then run the ml.py script to see the result
```
python3 ml.py
```

## Cost function
The cost function used here is the mean squared error
This function calculate the difference between the expected result and the current result square it, do the sum of all the lines and then calculate the average squared error.
It allows us to see how efficient is our model, in this simple example you will see that at the beginning it is not efficient and then after some iterations it is almost 0 which means it became really efficient

## Update our bias and weight
To increase our bias and weight we need to use partial derivative to find the function that will allow us to increase our bias and weight
Once we found our equation we modify our bias and weight at every iteration of the training

## Predict
And at the end we are doing a single prediction to check if the model is correct
