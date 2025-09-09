# Install needed packages if not already installed
if(!require(bnlearn)) install.packages("bnlearn")
if(!require(gRain)) install.packages("gRain")
if(!require(Rgraphviz)) BiocManager::install("Rgraphviz")

library(bnlearn)
library(gRain)
library(Rgraphviz)

# ---- Load Asia dataset ----
data("asia") # From bnlearn

# ---- Hill-climbing algorithms ----
#' Multiple runs of the hill-climbing algorithm can return non-equivalent BN structures.
#' We demonstrate this by running hc twice with different seeds.

set.seed(123)
bn1 <- hc(asia, score = "bic")

set.seed(456)
bn2 <- hc(asia, score = "bic")

# Compare the DAGs
arcs(bn1)
arcs(bn2)

# Are they equivalent? Compare equivalence classes
all.equal(cpdag(bn1), cpdag(bn2))   # TRUE means equivalent, FALSE means non-equivalent

# Plot both DAGs
graphviz.plot(bn1, main = "Hill-climbing Run 1")
graphviz.plot(bn2, main = "Hill-climbing Run 2")

# ---- LEARN BN ----
#' Next we learn a BN from 80% of the Asia dataset (train set),
#' fit its parameters, and then classify the remaining 20% (test set).

# Partition data into training and test
n <- nrow(asia)
train_idx <- sample(1:n, size = 0.8*n)
asia_train <- asia[train_idx, ]
asia_test  <- asia[-train_idx, ]

# Learn structure and parameters
bn <- hc(asia_train)
fitted_bn <- bn.fit(bn, asia_train)

# Plot learned BN
graphviz.plot(bn, main = "Learned BN (Hill-climbing on train set)")

# Convert to gRain object for inference
grain_bn <- as.grain(fitted_bn)

# ---- Classification of S on test data ----
predictions <- sapply(1:nrow(asia_test), function(i) {
  ev <- asia_test[i, !(names(asia_test) %in% "S"), drop=FALSE]
  ev <- lapply(ev, as.character)
  q  <- querygrain(setEvidence(grain_bn, evidence = ev), nodes="S")$S
  ifelse(q["yes"] > q["no"], "yes", "no")
})

# Confusion matrix
table(True = asia_test$S, Pred = predictions)

# ---- Markov blanket classification ----
mb_S <- mb(bn, "S")  # get Markov blanket of S
mb_S

pred_mb <- sapply(1:nrow(asia_test), function(i) {
  ev <- asia_test[i, mb_S, drop=FALSE]
  ev <- lapply(ev, as.character)
  
  q <- querygrain(setEvidence(grain_bn, evidence = ev), nodes="S")$S
  ifelse(q["yes"] > q["no"], "yes", "no")
})

# Confusion matrix
table(True = asia_test$S, Pred = pred_mb)

# ---- Naive Bayes classifier ----
#' Build a Naive Bayes BN with S as the class variable and all others as independent children.

nb_dag <- model2network("[S][A|S][T|S][L|S][B|S][D|S][E|S][X|S]")

# Plot Naive Bayes structure
graphviz.plot(nb_dag, main = "Naive Bayes BN Structure")

# Fit parameters
nb_fit <- bn.fit(nb_dag, asia_train)

# Convert to gRain object
grain_nb <- as.grain(nb_fit)

# Classify test data with Naive Bayes BN
pred_nb <- sapply(1:nrow(asia_test), function(i) {
  ev <- asia_test[i, !(names(asia_test) %in% "S"), drop=FALSE]
  ev <- lapply(ev, as.character)
  q <- querygrain(setEvidence(grain_nb, evidence = ev), nodes="S")$S
  ifelse(q["yes"] > q["no"], "yes", "no")
})

# Confusion matrix
table(True = asia_test$S, Pred = pred_nb)

# ---- Comparison of results ----
#' The accuracy of the three approaches can now be compared:
#' 1. Full BN learned with hill-climbing.
#' 2. BN restricted to Markov blanket of S.
#' 3. Naive Bayes classifier.
#' Typically:
#' - (1) performs best since it uses full learned dependencies,
#' - (2) is almost as good since Markov blanket is sufficient for S,
#' - (3) may perform worse due to independence assumptions.
