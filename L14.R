# Install needed packages if not already installed
if(!require(bnlearn)) install.packages("bnlearn")
if(!require(gRain)) install.packages("gRain")
if(!require(caret)) install.packages("caret")
suppressPackageStartupMessages({
  library(bnlearn)
  library(gRain)
  library(caret)
})

# Create the naive Bayes DAG structure manually
# S is the parent of all other variables
nb_dag <- model2network("[S][A|S][T|S][L|S][B|S][E|S][X|S][D|S]")
graphviz.plot(nb_dag, main="Naive Bayes BN Structure")

# Fit the parameters using the training data
nb_fitted <- bn.fit(nb_dag, data = asia_train)

# Convert to gRain object for inference
nb_grain <- as.grain(nb_fitted)

#' ## Classification on test set using Naive Bayes
predictions_nb <- sapply(1:nrow(asia_test), function(i) {
  case <- asia_test[i, ]
  
  # Evidence: all variables except S (all predictive features)
  ev <- as.list(case[ , !(names(case) %in% "S"), drop = FALSE])
  
  # Ensure character values (not factors)
  ev <- lapply(ev, as.character)
  
  # Calculates posterior distributions given evidence
  nb_evid <- setEvidence(nb_grain, evidence = ev)
  
  # Get posterior probability for S
  prob_S <- querygrain(nb_evid, nodes = "S")$S
  
  # Pick most probable class
  if (prob_S["yes"] > prob_S["no"]) "yes" else "no"
})

#' ## Confusion matrix for Naive Bayes
conf_matrix_nb <- table(Actual = asia_test$S, Predicted = predictions_nb)
print("Confusion Matrix for Naive Bayes Classifier:")
print(conf_matrix_nb)

#' ## Compare with previous results
print("Confusion Matrix using Full BN (from exercise 1):")
print(bn_table)

calculate_metrics(bn_table, "With HC")
calculate_metrics(conf_matrix_nb, "With naîve Bay")


#' ## Discussion
#' 
#' ### Question:
#' Repeat the exercise (2) using a naive Bayes classifier, i.e. the predictive 
#' variables are independent given the class variable. See p. 380 in Bishop’s 
#' book or Wikipedia for more information on the naive Bayes classifier. Model 
#' the naive Bayes classifier as a BN. You have to create the BN by hand, i.e. 
#' you are not allowed to use the function naive.bayes from the bnlearn package.
#' 
#' ### Answer
#' We state that S depends on all variables, then given this structure we 
#' calculate all posteriors.
#' 
#' In previous tasks we did this, 
#' Find suitable structure, THEN GIVEN this structure we fit parameters using
#' maximum likelihood, then use grain to make queries.
#' 
#' Now we do this, 
#' Assume a structure, THEN GIVEN this structure we fit parameters using
#' maximum likelihood, then use grain to make queries.
#' 
#' This is reasonable since to find the optimal structure is O(N!), HC is a 
#' heuristic that finds a "Good" structure. And in this task we see that we can 
#' get similar performance by naively chosing a structure.
#' 
#' What is important here is the fact that the structure with Naive bay is O(1).
