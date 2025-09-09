# Install needed packages if not already installed
if(!require(bnlearn)) install.packages("bnlearn")
if(!require(gRain)) install.packages("gRain")
if(!require(caret)) install.packages("caret")
suppressPackageStartupMessages({
      library(bnlearn)
      library(gRain)
      library(caret)
})

#' ## Load and Partition Data 
set.seed(42)

data("asia") 

# Partition data into training and test
n <- nrow(asia)
train_idx <- sample(1:n, size = 0.8*n)
asia_train <- asia[train_idx, ]     # takes all indexes in asia 
asia_test  <- asia[-train_idx, ]    # Removes all indexes in asia


#' ## Learn Structure and Parameters
#' hc() - Hill-climbing algorithm for structure learning, find DAG that 
#'        maximizes score function)
#' bn.fit() - Estimates parameters using maximum likelihood
#' as.grain() - Converts bn.fit object to gRain format for exact inference.
#'              The posterior distributions are calculated on-demand when
#'              querying.
bn <- hc(asia_train) # Here we use default score and start.
fitted_bn <- bn.fit(bn, asia_train) # Again, default here

# Plot learned BN
graphviz.plot(bn, main = "Learned BN (Hill-climbing on train set)")

# Convert to gRain object for inference
grain_bn <- as.grain(fitted_bn)

# Classification of S on test data 
predictions_bn <- sapply(1:nrow(asia_test), function(i) {
  ev <- asia_test[i, !(names(asia_test) %in% "S"), drop=FALSE]
  ev <- lapply(ev, as.character)
  q  <- querygrain(setEvidence(grain_bn, evidence = ev), nodes="S")$S
  ifelse(q["yes"] > q["no"], "yes", "no") 
  # here there is room for changes, instead of criteria q(yes) > q(no) we could
  # instead have like q(yes) > X%
})

# Confusion matrix
bn_table <- table(True = asia_test$S, Pred = predictions_bn)

#' ## Construct True Graph
# From instructions
dag_true = model2network("[A][S][T|A][L|S][B|S][D|B:E][E|T:L][X|E]") 
fitted_true = bn.fit(dag_true, data = asia)
grain_true = as.grain(fitted_true)

graphviz.plot(dag_true, main = "True BN (Given in Assignment)")


# Classification of S on test data 
predictions_true <- sapply(1:nrow(asia_test), function(i) {
  ev <- asia_test[i, !(names(asia_test) %in% "S"), drop=FALSE]
  ev <- lapply(ev, as.character)
  # Calculate posterios given evidence
  q  <- querygrain(setEvidence(grain_true, evidence = ev), nodes="S")$S
  ifelse(q["yes"] > q["no"], "yes", "no")
})


# Confusion matrix
true_table <- table(True = asia_test$S, Pred = predictions_true)

#' ## Comparison
#' Here we simply note that they are more or less identical implying very 
#' similar predicative performance.
true_table
bn_table


#' ## Discussion
#' 
#' ### Question:
#' 
#' Learn a BN from 80 % of the Asia data set. The data set is included in the 
#' bnlearn package. To load the data, run data("asia"). Learn both the structure 
#' and the parameters. Use any learning algorithm and settings that you consider 
#' appropriate. Use the BN learned to classify the remaining 20 % of the Asia 
#' data set in two classes: S = yes and S = no. In other words, compute the
#' posterior probability distribution of S for each case and classify it in the
#' most likely class. To do so, you have to use exact or approximate inference 
#' with the help of the bnlearn and gRain packages, i.e. you are not allowed to
#' use functions such as predict. Report the confusion matrix, i.e. true/false 
#' positives/negatives. Compare your results with those of the true Asia BN, 
#' which can be obtained by running 
#' dag = model2network("[A][S][T|A][L|S][B|S][D|B:E][E|T:L][X|E]")
#' 
#' ### Answer
#' If we compare the confusion matrices we see that the trained BN produce the 
#' same results as the true dag ( in the confusion matrix metric ). However,
#' when we compare the structure of the graphs we see that they differ. Upon 
#' inspection of the plots of the graphs this is obvious. 
#' 
#' This is interesting because we here see that networks that are different may
#' have the same predictive performance. 
