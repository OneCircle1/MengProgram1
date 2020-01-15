"""
This is a program using fastText to do text classification.

Question: In test result, precision and recall are often the same.
"""

import fasttext

# Training the file
training_file = "dbpedia.train"
model = fasttext.train_supervised(input = training_file)

# Save the model
model.save_model("dbpedia.bin")

# Testing the file
testing_file = "dbpedia.test"
test_result = model.test(testing_file)

# The test_result is a tuple
# [number of samples, precision, recall]
print(test_result)		

# Calculate the F1 value
F1 = 2*(test_result[1]*test_result[2])/(test_result[1]+test_result[2])

# Print the result
print("The precision is %f." % test_result[1])
print("The recall is %f." % test_result[2])
print("The F1 value is %f." % F1)
