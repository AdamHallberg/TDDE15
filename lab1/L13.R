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

# First run task 2 to get the work space
# Our fitted BN is called "fitted_bn"

#' ## Get the Markov Blankets of S in the Network
mb_vars <- mb(fitted_bn, "S")
mb_vars
#' So we see that the Markov blankets of S are "L" and "B".
#' This we can verify by stating the definition and observe the graph
#' Markov blankets are defined as the minimal set of nodes that separates a 
#' given node from the rest. 
graphviz.plot(bn, main = "Learned BN (Hill-climbing on train set)")
#' Here it is easy to see that L and B satisfy this.
#' 
#' Now we classify using only Markov blankets.
# Classify using only Markov blanket variables
predictions_mb <- sapply(1:nrow(asia_test), function(i) {
  # Get only the Markov blanket variables from test case
  ev <- asia_test[i, mb_vars, drop = FALSE]
  ev <- lapply(ev, as.character)
  
  # Query probability of S given only Markov blanket evidence
  q <- querygrain(setEvidence(grain_bn, evidence = ev), nodes = "S")$S
  ifelse(q["yes"] > q["no"], "yes", "no")
})

# Confusion matrix for Markov blanket classification
mb_table <- table(True = asia_test$S, Pred = predictions_mb)

#' ## Comparison of Results
cat("\n=== COMPARISON OF CLASSIFICATION RESULTS ===\n")

#' Using ALL variables (except S)
print(bn_table)
#' This had accuracy:
print(round(sum(diag(bn_table)) / sum(bn_table), 4))

#' Using only MARKOV BLANKET variables
print(mb_table)
#' This had accuracy:
print(round(sum(diag(mb_table)) / sum(mb_table), 4))

#' Using the True BN structure 
print(true_table)
#' This has accuracy:
round(sum(diag(true_table)) / sum(true_table), 4)

#' ## Performance Metrics Comparison
calculate_metrics <- function(conf_matrix, label) {
  tp <- conf_matrix["yes", "yes"]
  fp <- conf_matrix["yes", "no"] 
  fn <- conf_matrix["no", "yes"]
  tn <- conf_matrix["no", "no"]
  
  accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
  precision <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
  recall <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
  f1 <- ifelse((precision + recall) > 0, 2 * (precision * recall) / (precision + recall), 0)
  
  cat(sprintf("\n%s Metrics:", label))
  cat(sprintf("\n  Accuracy:  %.4f", accuracy))
  cat(sprintf("\n  Precision: %.4f", precision)) 
  cat(sprintf("\n  Recall:    %.4f", recall))
  cat(sprintf("\n  F1-score:  %.4f", f1))
}

calculate_metrics(bn_table, "All Variables")
calculate_metrics(mb_table, "Markov Blanket Only")
calculate_metrics(true_table, "True BN Structure")

#' ## Discussion
#' 
#' ### Question:
#' 
#' In the previous exercise, you classified the variable S given observations 
#' for all the rest of the variables. Now, you are asked to classify S given 
#' observations only for the so-called Markov blanket of S, i.e. its parents 
#' plus its children plus the parents of its children minus S itself. Report 
#' again the confusion matrix.
#' 
#' ### Answer
#' 
#' The Markov blanket of a variable contains all variables that can provide
#' information about it which is parents, children, and parents of 
#' children (called spouses?). According to the Markov property, the variable S
#' is conditionally independent of all other variables given its Markov blanket.
#' 
#' So we expect the predicative preformance of the leaned BN useing all nodes
#' and only the Markov blanket nodes to have the same predicative preformance of
#' S. So all other variables are conditionally independent given the Markov
#' Blanket. That is,
#' 
#' $S \perp \{T, E, X, D\} | \{L, B\}$
#' 
#' And this is supported by the results.
