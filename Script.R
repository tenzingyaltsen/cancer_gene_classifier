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
classes <- classes[,-1]
str(genes)

# Process data into matrix type (already numerical) for use with keras.
genes <- as.matrix(genes)
# One hot encode labels. Keras requires integer type to use to_categorical() function.
classes_labels <- as.factor(classes)
# Levels are BRCA COAD KIRC LUAD PRAD.
classes_indices <- as.numeric(classes_labels) - 1
# Now the levels are 0, 1, 2, 3, 4 (same order as above). Can apply to_categorical().
classes <- to_categorical(classes_indices)

# Assign training (50%), validation (25%) and test indices (25%).
set.seed(123)
indices <- 1:nrow(genes)
train_indices <- sample(indices, size = 0.5 * length(indices))
indices <- setdiff(indices, train_indices)
length(train_indices)
length(indices)
val_indices <- sample(indices, size = 0.5 * length(indices))
length(val_indices)
test_indices <- setdiff(indices,val_indices)
length(test_indices)

