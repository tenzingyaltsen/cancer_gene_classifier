# Load packages.
library(keras)
library(ggplot2)

# Import data and view structure.
genes <- read.csv("pancan_data.csv")
dim(genes)
str(genes)
classes <- read.csv("pancan_labels.csv")
dim(classes)
str(classes)

# Pre-process data by removing unnecessary columns.
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

# Assign training, validation and test data.
train_data <- genes[train_indices,]
train_labels <- classes[train_indices,]
val_data <- genes[val_indices,]
val_labels <- classes[val_indices,]
test_data <- genes[test_indices,]
test_labels <- classes[test_indices,]

# Define function to build/compile initial network with 3 hidden layers.
# Variables to change include number of units, optimizer, batch size and epoch number.
train_model <- function(units, optimizer_type, batches, epoch_number) {
  # Ensure reproducibility by setting seed for both R and keras via tensorflow.
  set.seed(123)
  tensorflow::set_random_seed(123)
  model <- keras_model_sequential() %>%
    # ReLU activation function picked.
    layer_dense(units = units, activation = "relu", input_shape = c(20530)) %>%
    layer_dense(units = units, activation = "relu") %>%
    layer_dense(units = units, activation = "relu") %>%
    # 5 possible classes, so 5 units in output layer.
    # Softmax activation function used due to classification task. 
    layer_dense(units = 5, activation = "softmax")
  model %>% compile(
    optimizer = optimizer_type,
    loss = "categorical_crossentropy",
    metrics = "accuracy"
  )
  history <- model %>% fit(
    train_data,
    train_labels,
    batch_size = batches,
    epochs = epoch_number
  )
  plot(history)
  cat("Hyperparameters: ", units, optimizer_type, batches, epoch_number, "\n")
  cat("Loss: ", tail(history$metrics$loss,1), "\n")
  cat("Accuracy: ", tail(history$metrics$accuracy,1))
}

#'Starting parameters are 128 units, 'rmsprop' optimizer, batch size of 2048 and epochs
#' of 50. Let's start by tuning the number of units in the 3 hidden layers. 
train_model(1024,"rmsprop",2048,50)
#'Started with 1024 units, between number of input features and output classes, and 
#' considering that data appears not too sparse.
train_model(2048,"rmsprop",2048,50)
# Increased to 2048. Arrived at the same accuracy but had lower loss. Try lower units.
train_model(512,"rmsprop",2048,50)
# Decreased to 512. Arrived at the same accuracy but had higher loss. Try even higher units.
train_model(4096,"rmsprop",2048,50)
# Again, same accuracy, but lower loss. Let's try even higher units!
train_model(8192,"rmsprop",2048,50)
#'Again, same accuracy, but lower loss, though with this 2X increase the loss decrease is 
#' less. Let's select previous unit number of 4096 since loss was only marginally worse 
#' but training was faster.

# Now let's tune for the optimizer, considering 2 other optimizers.
train_model(4096,"...",2048,50)
