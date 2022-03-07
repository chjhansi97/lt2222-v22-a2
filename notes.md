Part 1:
I used gzip function to open the zip file and random.sample to randomly select certain number of lines from the file.

Part 2:
The sampled sentences are processed producing a list of processed_sentences.The sentences are tokenized, tagged and removed stop words.

Part 3:
The processed sentences are used to produce five_gram lists using ngrams(). From the list features are created. 

Part 4:
Vectors are created for rows and columns to create a dataframe. 

Part 5:
train_test_split() is used to split the data.
train() is used to get a model for linear and rbf kernels.

Part 6:
The eval_model() is used to calculate precision, recall and f-measure for the model on the test data. The model couldn't produce any values as well as no errors either. I ran it for almost 5 hours but there were no results for precision, recall.
