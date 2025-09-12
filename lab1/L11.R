# Load data set
data("asia")

#' ## Hill-Climbing Algorithms
#' Multiple runs of the hill-climbing algorithm can return non-equivalent BN structures.
#' We demonstrate this by running hc twice with different seeds.
#'
#' Hill-climbing is a form of stochastic optimization. One might assume that the
#' algorithm should always produce equivalent graphs, but this is not guaranteed
#' because it can converge to different local optima depending on the starting point.
#' HC is not asymptotically correct under faithfulness, i.e. it may get trapped
#' in local optima.
#'
#' The choice of score also matters:
#' - BIC: log-likelihood – (penalty * number_of_params). This is an asymptotic
#'        approximation that penalizes complexity, and different runs can still lead
#'        to different solutions.
#'
#' - BDe: Bayesian Dirichlet equivalent uniform prior. This evaluates P(structure | data).
#'        The parameter `iss` (imaginary sample size) controls the weight of the prior.
#'        Different values of `iss` can lead to different structures.

#' ## BIC comparison 
set.seed(42)
random_init <- random.graph(nodes = names(asia))
bn1_bic <- hc(asia, score = "bic", start = random_init)


random_init <- random.graph(nodes = names(asia))
bn2_bic <- hc(asia, score = "bic", start = random_init)

# Compare the DAGs
arcs(bn1_bic)
arcs(bn2_bic)

# Are they equivalent? Compare equivalence classes
all.equal(cpdag(bn1_bic), cpdag(bn2_bic))   # TRUE means equivalent, FALSE means non-equivalent

# Plot both DAGs
graphviz.plot(bn1_bic, main = "Hill-climbing Run 1, BIC")
graphviz.plot(bn2_bic, main = "Hill-climbing Run 2, BIC")

#' ## BDe comparison
set.seed(42)
random_init <- random.graph(nodes = names(asia))
bn1_bde <- hc(asia, score = "bde", iss = 10, start = random_init)

set.seed(24)
random_init <- random.graph(nodes = names(asia))
bn2_bde <- hc(asia, score = "bde", iss = 10, start = random_init)

# Compare the DAGs
arcs(bn1_bde)
arcs(bn2_bde)

# Are they equivalent? Compare equivalence classes
all.equal(cpdag(bn1_bde), cpdag(bn2_bde))   # TRUE means equivalent, FALSE means non-equivalent

# Plot both DAGs
graphviz.plot(bn1_bde, main = "Hill-climbing Run 1, BDE")
graphviz.plot(bn2_bde, main = "Hill-climbing Run 2, BDE")

#' ## Discussion
#' The question was:
#'
#' Show that multiple runs of the hill-climbing algorithm can return non-equivalent 
#' Bayesian network (BN) structures. Explain why this happens. Use the Asia 
#' data set which is included in the bnlearn package. To load the data, 
#' run data("asia"). Recall from the lectures that the concept of non-equivalent
#' BN structures has a precise meaning.
#' 
#' Answer:
#' When we use different seeds we see that we produce non-equivalent graphs. 
#' In our code the use of seeds does not affect how we partition the data, it 
#' only affects the initial graph structure. Since the final graph depends on 
#' the initial graph. However the algorithm is deterministic given the same 
#' initial graph. So given the same data we see that we converge to different
#'solutions given different initial graphs.

