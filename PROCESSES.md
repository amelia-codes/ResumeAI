#Results and Processes
###Token-Based Neural Network
- This network started off with quite a few issues. The data (text) had to be tokenized with a spacy nlp (I used the medium pipeline here) and then converted into a vector which was finally converted into a tensor.
- The resulting tensors had to be padded before inputting into the neural network to be size-compatible (I did not cut any data for risk of losing important information)
- The value for size1 had to be 300 instead of 384 to fit the results of transforming the data which is different from the other model and I suspect is a cause for the some loss in the results of the neural network
- There was also an extra dimension to the array which was aggregated to fit
- Using (almost) all of the same initial conditions as the other model, the final loss was 0.693943. The loss between each epoch decreased by about .000010 each round and the initial loss was 0.694058.
- The final results included:
    - Accuracy: 63.4%
    - Avg loss: 0.688499
