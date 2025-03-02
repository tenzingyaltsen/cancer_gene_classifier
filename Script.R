# Load packages.
library(keras)
library(ggplot2)

# Import data and view structure.
genes <- read.csv("pancan_data.csv")
dim(genes)
str(genes)
# Import labels and view structure.
classes <- read.csv("pancan_labels.csv")
dim(classes)
str(classes)

# Pre-process data and labels by removing unnecessary columns.
genes <- genes[,-1:-2]
str(genes)
classes <- classes[,-1]
str(classes)

# Process data into matrix type (already numerical) for use with keras.
genes <- as.matrix(genes, nrow = 801, ncol = 20530, byrow = TRUE)
# One hot encode labels. Keras requires integer type to use to_categorical() function.
classes_labels <- as.factor(classes)
# Levels are BRCA COAD KIRC LUAD PRAD.
classes_indices <- as.numeric(classes_labels) - 1
# Now the levels are 0, 1, 2, 3, 4 (same order as above). Can apply to_categorical().
classes <- to_categorical(classes_indices)

# Assign training (50%), validation (25%) and test indices (25%).
set.seed(123)
# Create pool of available indices.
indices <- 1:nrow(genes)
# Assign training indices and remove those indices from available pool.
train_indices <- sample(indices, size = 0.5 * length(indices))
indices <- setdiff(indices, train_indices)
# Check length. 400 observations in train_indices and 401 remaining in pool.
length(train_indices)
length(indices)
# Assign validation indices and assign the remaining indices in pool to test indices.
val_indices <- sample(indices, size = 0.5 * length(indices))
test_indices <- setdiff(indices,val_indices)
# Check length. 200 observations in val_indices and 201 in test_indices.
length(val_indices)
length(test_indices)

# Assign training, validation and test data and labels.
train_data <- genes[train_indices,]
train_labels <- classes[train_indices,]
val_data <- genes[val_indices,]
val_labels <- classes[val_indices,]
test_data <- genes[test_indices,]
test_labels <- classes[test_indices,]

#'Scale the training, validation and test data, using the mean and std of training data.
# mean <- apply(train_data, 2, mean)
# std <- apply(train_data, 2, sd)
# train_data <- scale(train_data, center = mean, scale = std)
#'Each feature in training data will have a mean of 0 and standard deviation of 1.
# val_data <- scale(val_data, center = mean, scale = std)
# test_data <- scale(test_data, center = mean, scale = std)
#'NOTE: Scaling NOT performed due to stubbornly low accuracy of model, even with tuning.

# Define function to build/compile initial network with 3 hidden layers.
# Variables to change include number of units, optimizer, batch size and epoch number.
train_model <- function(seed, units, optimizer_type, batches, epoch_number) {
  # Ensure reproducibility by setting seed for both R and keras via tensorflow.
  set.seed(seed)
  tensorflow::set_random_seed(seed)
  model <- keras_model_sequential() %>%
    # ReLU activation function picked.
    layer_dense(units = units, activation = "relu", input_shape = c(20530)) %>%
    layer_dense(units = units, activation = "relu") %>%
    layer_dense(units = units, activation = "relu") %>%
    # 5 possible classes, so 5 units in output layer.
    # Softmax activation function used due to classification task. 
    layer_dense(units = 5, activation = "softmax")
  
  # Compile model, using parameters that would fit a multi-class classification problem.
  model %>% compile(
    optimizer = optimizer_type,
    loss = "categorical_crossentropy",
    metrics = "accuracy"
  )
  
  # Train model, including calculation of validation metrics and loss.
  history <- model %>% fit(
    train_data,
    train_labels,
    batch_size = batches,
    epochs = epoch_number,
    validation_data = list(val_data,val_labels)
  )
  
  # Plot training and validation loss over epochs.
  plot(history$metrics$loss, type = "l", col = "black", xlab = "Epochs", ylab = "Loss")
  lines(history$metrics$val_loss, type = "l", col = "red")
  legend("topright", legend = c("Training Loss", "Validation Loss"), lwd = 1,
         col = c("black", "red"), cex = 0.5)
  # Plot training and validation accuracy over epochs.
  plot(history$metrics$accuracy, type = "l", col = "black", xlab = "Epochs", ylab = "Accuracy")
  lines(history$metrics$val_accuracy, type = "l", col = "red")
  legend("bottomright", legend = c("Training Accuracy", "Validation Accuracy"), lwd = 1,
         col = c("black", "red"), cex = 0.5)
  
  # Print last epoch accuracy and loss values.
  cat("Hyperparameters: ", units, optimizer_type, batches, epoch_number, "\n")
  cat("Training Loss: ", tail(history$metrics$loss,1), "\n")
  cat("Training Accuracy: ", tail(history$metrics$accuracy,1), "\n")
  cat("Validation Loss: ", tail(history$metrics$val_loss,1), "\n")
  cat("Validation Accuracy: ", tail(history$metrics$val_accuracy,1), "\n")
  # 
  # Print maximum validation accuracy and minimum loss values encountered.
  cat("Max Validation Accuracy: ", max(history$metrics$val_accuracy), "\n")
  cat("Min Validation Loss: ", min(history$metrics$val_loss), "\n")
  
  # Return model so that it can be assigned to an object outside of this function.
  return(model)
}

#'Starting parameters are 1024 units, 'rmsprop' optimizer, batch size of 256 and epochs
#' of 50. Let's start by tuning the number of units in the 3 hidden layers by assessing
#' validation loss and accuracy. We want to minimize loss and maximize accuracy.
train_model(123,1024,"rmsprop",256,50)
#'Started with 1024 units, between number of input features and output classes, and 
#' considering that data appears not too sparse. Accuracy is quite low.
train_model(123,2048,"rmsprop",256,50)
# Increased to 2048. Arrived at the same accuracy but had higher loss. Try lower units.
train_model(123,512,"rmsprop",256,50)
# Decreased to 512. Arrived at the same accuracy but had lower loss. Try even less units.
train_model(123,256,"rmsprop",256,50)
# Decreased to 256. Arrived at same accuracy but higher loss. Proceed with 512 units.

# Now let's tune for the optimizer, considering 2 other optimizers (3 total).
train_model(123,512,"sgd",256,50)
# Same accuracy, but lower loss. Better, but need to consider the last optimizer.
train_model(123,512,"adam",256,50)
# Works great! Much higher accuracy and much lower loss. Proceed with "adam" optimizer.

# Now let's tune the batch size.
train_model(123,512,"adam",512,50)
# Lower accuracy (did not reach 1) and higher loss within 50 epochs. Try smaller batches.
train_model(123,512,"adam",128,50)
#'Reached maximum accuracy of 1 and lower minimum loss faster than larger batch size. Try
#' even smaller batch size.
train_model(123,512, "adam", 64,50)
train_model(123,512, "adam", 32,50)
#'With smaller batch sizes, the model reaches maximum validation accuracy quicker. This
#' reduces the number of epochs needed and is favourable, but if batch size is too small, 
#' this can lead to more instability with weight updates (each batch sees fewer samples).
# Let's proceed with 64 as batch size since 32 might be too small.

#'It looks like with pre-specified hyperparameters, the validation loss begins to 
#' increase/level off around 5 epochs while the validation accuracy appears to reach 
#' its maximum at around 5 epochs as well. 
train_model(123,512, "adam", 64,5)
#'However, we can't guarantee that this number of epochs will be optimal in other data
#' sets, so let's increase the epochs to give us some wiggle room. Let's add only 5 epochs
#' since we do not want to overfit to the training data.
train_model(123,512, "adam", 64,10)
# We have our tuned model!

# Merge training and validation data into a new training data set.
train_indices <- append(train_indices,val_indices)
train_data <- genes[train_indices,]
train_labels <- classes[train_indices,]
# Train model on "full" training set. Ignore validation loss/accuracy values from function.
model <- train_model(123,512, "adam", 64,10)

# Evaluate model on test set.
results <- model %>% evaluate(test_data, test_labels)
results

# Create vector of predicted probabilities for each class for the test data.
preds <- predict(model, test_data)
# Create vector with classes with highest probability for each test patient.
preds.cl <- max.col(preds)
# Recall that the classes BRCA COAD KIRC LUAD PRAD correspond to labels 0, 1, 2, 3, 4.
# Create confusion matrix of actual vs predicted labels.
table(max.col(test_labels),preds.cl)
#'The model almost classified all test data to correctly match test labels. 

# Define function for splitting the data and labels based on random inputted seed.
split_data <- function(seed) {
  set.seed(seed)
  # Split indices into training and test indices.
  indices <- 1:nrow(genes)
  train_indices <- sample(indices, size = 0.75 * length(indices))
  test_indices <- setdiff(indices, train_indices)
  
  # Assign data.
  train_data <- genes[train_indices,]
  train_labels <- classes[train_indices,]
  test_data <- genes[test_indices,]
  test_labels <- classes[test_indices,]
}

# Train and evaluate model 5 times using previous defined function.
# Create an empty results table.
results_table <- data.frame(Seed = integer(), Test_Loss = numeric(), Test_Accuracy = numeric())
for (i in c(123,234,345,456,567,678)) {
  # Split data randomly based on given seed.
  split_data(i)
  # Train and evaluate model, adding results to results table.
  model <- train_model(i, 512, "adam", 64,10)
  results <- model %>% evaluate(test_data, test_labels)
  print(results)
  results_table <- rbind(results_table, data.frame(Seed = i, Test_Loss = results[1], 
                                                   Test_Accuracy = results[2]))
  
  # Generate confusion matrix.
  preds <- predict(model, test_data)
  preds.cl <- max.col(preds)
  print(table(max.col(test_labels),preds.cl))
}
# View final results for all seeds.
print(results_table)
# View average test loss of model across all seeds.
print(mean(results_table$Test_Loss))
# View average test accuracy of model across all seeds.
print(mean(results_table$Test_Accuracy))
# Not bad results! But there is area for improvement.