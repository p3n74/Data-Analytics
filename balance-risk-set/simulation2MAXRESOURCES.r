# Load necessary libraries
library(MatchIt)
library(dplyr)
library(boot)
library(MASS)
library(optmatch)
library(parallel)
library(pbapply)  # For progress bar

# Set seed for reproducibility
set.seed(69)

# Simulation parameters
n_treated <- 500 
n_control <- 500 
p_cov <- 5
beta <- 2
sigma <- 1
n_sim <- 1000 

# Function to perform a single simulation
simulate_iteration <- function(i) {
  # Data generation
  cov_matrix <- matrix(0.3, nrow = p_cov, ncol = p_cov)
  diag(cov_matrix) <- 1
  X <- mvrnorm(n = n_treated + n_control, mu = rep(0, p_cov), Sigma = cov_matrix)
  X <- scale(X)
  colnames(X) <- paste0("X", 1:p_cov)
  data <- as.data.frame(X)

  # Treatment assignment
  beta_ps <- rep(0.2, p_cov)
  propensity_logit <- X %*% matrix(beta_ps, ncol = 1)
  propensity_scores <- plogis(propensity_logit)
  propensity_scores <- pmin(pmax(propensity_scores, 0.05), 0.95)
  data$treated <- rbinom(n_treated + n_control, 1, propensity_scores)

  if (length(unique(data$treated)) < 2) {
    cat("Iteration", i, "No two levels, skipping\n")
    return(list(est = NA, bias = NA, lower_ci = NA, upper_ci = NA, coverage = NA))
  }

  # Outcome generation
  data$Y <- X %*% matrix(rep(1, p_cov), ncol = 1) + beta * data$treated + rnorm(n_treated + n_control, mean = 0, sd = sigma)

  # Propensity score matching
  ps_model <- glm(treated ~ ., data = data, family = binomial)
  data$ps <- predict(ps_model, type = "response")
  n_strata <- 5
  data$strata <- ntile(data$ps, n_strata)
  matched_data <- NULL

  for (s in 1:n_strata) {
    strata_data <- data %>% filter(strata == s)
    n_treated_strata <- sum(strata_data$treated == 1)
    n_control_strata <- sum(strata_data$treated == 0)

    cat("Iteration", i, "- Stratum", s, 
        ": Treated =", n_treated_strata, 
        ", Control =", n_control_strata, "\n")

    if (n_treated_strata < 1 || n_control_strata < 1) {
      next
    }

    desired_ratio <- floor(n_control_strata / n_treated_strata)
    desired_ratio <- ifelse(desired_ratio < 1, 1, desired_ratio)

    ##if (n_control_strata < n_treated_strata) {
    #  cat("Warning: Fewer control units than treated units; not all treated units will get a match.\n")
    #}

    m.out <- tryCatch({
      matchit(treated ~ X1 + X2 + X3 + X4 + X5,
              data = strata_data,
              method = "nearest",
              distance = "mahalanobis",
              ratio = desired_ratio)
    }, error = function(e) {
      cat("Error in matching at Iteration", i, "Stratum", s, ":", conditionMessage(e), "\n")
      return(NULL)
    })

    if (!is.null(m.out)) {
      strata_matched <- match.data(m.out)
      matched_data <- bind_rows(matched_data, strata_matched)
    }
  }

  if (is.null(matched_data) || nrow(matched_data) == 0) {
    return(list(est = NA, bias = NA, lower_ci = NA, upper_ci = NA, coverage = NA))
  }

  model <- lm(Y ~ treated, data = matched_data)
  est <- coef(model)["treated"]

  coef_fun <- function(data, indices) {
    d <- data[indices, ]
    fit <- lm(Y ~ treated, data = d)
    return(coef(fit)["treated"])
  }

  boot_results <- tryCatch({
    boot(data = matched_data, statistic = coef_fun, R = 500)
  }, error = function(e) {
    cat("Error in bootstrapping at Iteration", i, ":", conditionMessage(e), "\n")
    return(NULL)
  })

  ci <- boot.ci(boot_results, type = "perc")
  lower_ci <- if (!is.null(ci$percent)) ci$percent[4] else NA
  upper_ci <- if (!is.null(ci$percent)) ci$percent[5] else NA

  bias <- est - beta
  coverage <- (lower_ci <= beta) & (upper_ci >= beta)

  return(list(est = est, bias = bias, lower_ci = lower_ci, upper_ci = upper_ci, coverage = coverage))
}

# Use parallel processing with a progress bar
num_cores <- detectCores() - 1  # Use all but one core
results <- pblapply(1:n_sim, simulate_iteration, cl = num_cores)

# Extract results
estimates <- sapply(results, function(res) res$est)
biases <- sapply(results, function(res) res$bias)
lower_ci <- sapply(results, function(res) res$lower_ci)
upper_ci <- sapply(results, function(res) res$upper_ci)
coverage <- sapply(results, function(res) res$coverage)

# Evaluation metrics
valid_indices <- !is.na(estimates)
valid_estimates <- estimates[valid_indices]
valid_biases <- biases[valid_indices]
valid_lower_ci <- lower_ci[valid_indices]
valid_upper_ci <- upper_ci[valid_indices]
valid_coverage <- coverage[valid_indices]

mean_estimate <- mean(valid_estimates)
bias_mean <- mean(valid_biases)
rmse <- sqrt(mean(valid_biases^2))
coverage_prob <- mean(valid_coverage)

# Display results
cat("\n")
cat("Valid Simulations: ", length(valid_estimates), "/", n_sim, "\n")
cat("True Treatment Effect: ", beta, "\n")
cat("Bias: ", round(bias_mean, 4), "\n")
cat("RMS Error: ", round(rmse, 4), "\n")
cat("Coverage 95% CI: ", round(coverage_prob, 4), "\n")
