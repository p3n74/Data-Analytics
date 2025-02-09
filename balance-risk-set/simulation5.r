# Load necessary libraries
library(dplyr)
library(survival)
library(ROI)
library(ROI.plugin.glpk)
library(ggplot2)
library(parallel)
library(cobalt)
library(sensitivitymw)

# Set seed for reproducibility
set.seed(6969)

# Simulation Parameters
n_patients <- 800 # Total number of patients
n_treated_pairs <- 400 # Number of matched pairs
n_covariates <- 6     # Number of covariates

# Function to generate covariate values
generate_covariates <- function(n) {
  # Generate continuous covariates for Pain, Urgency, Frequency at baseline
  baseline_pain <- rnorm(n, mean = 5, sd = 2)
  baseline_urgency <- rnorm(n, mean = 5, sd = 2)
  baseline_frequency <- rnorm(n, mean = 5, sd = 2)
  
  # Generate change in covariates until treatment time
  delta_pain <- rnorm(n, mean = 0, sd = 1)
  delta_urgency <- rnorm(n, mean = 0, sd = 1)
  delta_frequency <- rnorm(n, mean = 0, sd = 1)
  
  # Covariates at treatment time
  treatment_pain <- baseline_pain + delta_pain
  treatment_urgency <- baseline_urgency + delta_urgency
  treatment_frequency <- baseline_frequency + delta_frequency
  
  # Combine into a data frame
  covariates <- data.frame(
    baseline_pain,
    baseline_urgency,
    baseline_frequency,
    treatment_pain,
    treatment_urgency,
    treatment_frequency
  )
  
  return(covariates)
}

# Simulate Patient Data
patients <- data.frame(ID = 1:n_patients)
patients <- cbind(patients, generate_covariates(n_patients))

# Simulate Treatment Assignment
# Patients who eventually receive treatment
treated_patients <- sample(patients$ID, size = n_treated_pairs)
patients$eventually_treated <- ifelse(patients$ID %in% treated_patients, 1, 0)

# Assume that treated patients receive treatment at random times
patients$treatment_time <- NA
patients$treatment_time[patients$eventually_treated == 1] <- sample(1:24, size = n_treated_pairs, replace = TRUE)
patients$treatment_time[patients$eventually_treated == 0] <- Inf  # Never treated

# Generate Unobserved Frailty Term
patients$unobserved <- rnorm(n_patients, mean = 0, sd = 1)

# Adjusted treatment effect
treatment_effect <- -4  # Adjusted from -10 to -2 for a moderate effect size

# Adjust the linear predictor to include the treatment effect
patients$lin_pred <- with(patients,
                          0.5 * baseline_pain +
                          0.5 * baseline_urgency +
                          0.5 * baseline_frequency +
                          0.5 * treatment_pain +
                          0.5 * treatment_urgency +
                          0.5 * treatment_frequency +
                          unobserved +
                          treatment_effect * eventually_treated)

# Simulate event times (time to symptom improvement)
baseline_hazard <- 0.0002 # Adjust as needed
patients$event_time <- rexp(n_patients, rate = baseline_hazard * exp(patients$lin_pred))

# Observe patients up to a fixed censoring time
censoring_time <- 180 
patients$time <- pmin(patients$event_time, censoring_time)
patients$event <- as.numeric(patients$event_time <= censoring_time)

# Define covariate names
covariate_names <- c("baseline_pain", "baseline_urgency", "baseline_frequency",
                     "treatment_pain", "treatment_urgency", "treatment_frequency")

# Calculate quantiles for each covariate
quantiles <- lapply(patients[, covariate_names], function(x) quantile(x, probs = c(1/3, 2/3)))

# Create binary variables for quantiles
create_binary_variables <- function(x, q) {
  # x: vector of covariate values
  # q: vector of quantile cutoffs (1/3 and 2/3)
  binary_vars <- matrix(0, nrow = length(x), ncol = 2)
  binary_vars[, 1] <- as.numeric(x > q[1])
  binary_vars[, 2] <- as.numeric(x > q[2])
  return(binary_vars)
}

# Initialize an empty list to store binary variables
binary_vars_list <- list()

# Generate binary variables for each covariate
for (i in seq_along(covariate_names)) {
  covariate_name <- covariate_names[i]
  x <- patients[[covariate_name]]
  q <- quantiles[[i]]
  
  # Ensure that quantiles and covariate values are valid
  if (length(q) == 2 && !is.null(x)) {
    binary_vars <- create_binary_variables(x, q)
    colnames(binary_vars) <- paste0(c("B1_", "B2_"), covariate_name)
    binary_vars_list[[i]] <- binary_vars
  } else {
    stop(paste("Error in processing covariate:", covariate_name))
  }
}

# Combine all binary variables into a data frame
binary_covariates <- do.call(cbind, binary_vars_list)

# Add binary variables to patients data frame
patients <- cbind(patients, binary_covariates)

# Formulate and Solve the Integer Programming Problem
# Initialize Matched Pairs List
matched_pairs <- list()

# List of treated patients and their treatment times
treated_info <- patients %>%
  filter(eventually_treated == 1) %>%
  select(ID, treatment_time)

# Create cluster for parallel processing
num_cores <- detectCores() - 1  # Number of cores for parallel processing
cl <- makeCluster(num_cores)
clusterExport(cl, varlist = c("patients", "treated_info", "covariate_names", "binary_covariates"), envir = environment())
clusterEvalQ(cl, {
  library(dplyr)
  library(ROI)
  library(ROI.plugin.glpk)
})

# Function to perform matching for each treated patient
match_treated_patient <- function(i) {
  library(dplyr)
  library(ROI)
  library(ROI.plugin.glpk)
  
  # Get treated patient information
  treated_patient <- treated_info[i,]
  p <- treated_patient$ID
  Tp <- treated_patient$treatment_time
  
  # Identify potential controls: patients not yet treated at time Tp
  potential_controls <- patients %>%
    filter((eventually_treated == 0) | (treatment_time > Tp)) %>%
    filter(ID != p)
  
  if (nrow(potential_controls) == 0) {
    return(NULL)
  }
  
  # Compute Mahalanobis distance between treated patient and potential controls
  # Covariate matrices
  X_p <- as.numeric(patients[patients$ID == p, covariate_names])
  X_q <- as.matrix(potential_controls[, covariate_names])
  
  # Compute Mahalanobis distance
  cov_matrix <- cov(patients[, covariate_names])
  inv_cov_matrix <- solve(cov_matrix)
  diff <- X_q - matrix(X_p, nrow = nrow(X_q), ncol = length(X_p), byrow = TRUE)
  d_e <- sqrt(rowSums((diff %*% inv_cov_matrix) * diff))
  
  # Scale distances
  d_e <- d_e / max(d_e + 1e-8)
  
  # Prepare binary variables for balance constraints
  Bpk_names <- colnames(binary_covariates)
  Bpk <- as.numeric(patients[patients$ID == p, Bpk_names])
  Bek <- as.matrix(potential_controls[, Bpk_names])
  
  K <- length(Bpk)  # Total number of binary variables (should be 12)
  
  # Number of variables:
  # - N_controls flow variables (one per potential control)
  # - 2 * K gap variables
  
  N_controls <- nrow(potential_controls)
  num_variables <- N_controls + 2 * K
  
  # Objective function
  # Minimize total distance + penalties for imbalance
  # Set penalties to a large value
  Lambda_k <- rep(max(d_e) + 1, K)  # Penalties larger than any possible distance
  obj <- c(d_e, Lambda_k, Lambda_k)  # [flow variables, gk+, gk-]
  
  # Variable types
  var_types <- c(rep("B", N_controls), rep("C", 2 * K))
  
  # Constraints
  
  # 1. Matching Constraint: Treated patient matched to exactly one control
  mat_matching <- matrix(0, nrow = 1, ncol = num_variables)
  mat_matching[1, 1:N_controls] <- 1
  dir_matching <- "=="
  rhs_matching <- 1
  
  # 2. Balance Constraints
  mat_balance <- matrix(0, nrow = 2 * K, ncol = num_variables)
  rhs_balance <- numeric(2 * K)
  dir_balance <- rep("==", 2 * K)
  
  for (k in 1:K) {
    Bpk_k <- Bpk[k]
    Bek_k <- Bek[, k]
    
    # Positive Gap Constraint
    mat_balance[k, 1:N_controls] <- Bek_k
    mat_balance[k, N_controls + k] <- 1  # gk+
    mat_balance[k, N_controls + K + k] <- 0  # gk-
    rhs_balance[k] <- Bpk_k
    
    # Negative Gap Constraint
    mat_balance[K + k, 1:N_controls] <- -Bek_k
    mat_balance[K + k, N_controls + k] <- 0  # gk+
    mat_balance[K + k, N_controls + K + k] <- 1  # gk-
    rhs_balance[K + k] <- -Bpk_k
  }
  
  # Variable Bounds
  lb <- c(rep(0, num_variables))
  ub <- c(rep(1, N_controls), rep(Inf, 2 * K))
  
  # Combine Constraints
  mat <- rbind(mat_matching, mat_balance)
  dir <- c(dir_matching, dir_balance)
  rhs <- c(rhs_matching, rhs_balance)
  
  # Define the Optimization Problem
  opt_problem <- OP(objective = obj,
                    constraints = L_constraint(L = mat, dir = dir, rhs = rhs),
                    types = var_types,
                    bounds = V_bound(li = 1:num_variables, ui = 1:num_variables, lb = lb, ub = ub),
                    maximum = FALSE)
  
  # Solve the Integer Program
  result <- ROI_solve(opt_problem, solver = "glpk")
  
  # Check if an optimal solution was found
  if (result$status$code == 0) {
    # Extract solution
    solution <- result$solution
    matched_control_index <- which(solution[1:N_controls] == 1)
    
    if (length(matched_control_index) == 1) {
      matched_control <- potential_controls[matched_control_index, ]
      
      # Store Matched Pair
      matched_pair <- data.frame(
        TreatedID = p,
        ControlID = matched_control$ID
      )
      return(matched_pair)
    } else {
      return(NULL)
    }
  } else {
    # No feasible solution found
    return(NULL)
  }
}

# Export variables and functions to cluster
clusterExport(cl, varlist = c("patients", "treated_info", "covariate_names",
                              "binary_covariates", "match_treated_patient"), envir = environment())

# Perform matching in parallel
matched_pairs_list <- parLapply(cl, 1:n_treated_pairs, match_treated_patient)

# Stop cluster
stopCluster(cl)

# Combine matched pairs
matched_pairs <- do.call(rbind, matched_pairs_list)

# Remove any NULL entries
matched_pairs <- matched_pairs[complete.cases(matched_pairs), ]

# Assign pair IDs
matched_pairs$pair_id <- 1:nrow(matched_pairs)

# Retrieve matched data
matched_data_treated <- patients[patients$ID %in% matched_pairs$TreatedID, ]
matched_data_control <- patients[patients$ID %in% matched_pairs$ControlID, ]

# Assign treatment indicator
matched_data_treated$treatment <- 1
matched_data_control$treatment <- 0

# Merge pair IDs
matched_data_treated <- merge(matched_data_treated,
                              matched_pairs[, c("TreatedID", "pair_id")],
                              by.x = "ID", by.y = "TreatedID", all.x = TRUE)
matched_data_control <- merge(matched_data_control,
                              matched_pairs[, c("ControlID", "pair_id")],
                              by.x = "ID", by.y = "ControlID", all.x = TRUE)

# Combine matched data
matched_data <- rbind(matched_data_treated, matched_data_control)

# Sort matched_data by pair_id and treatment
matched_data <- matched_data[order(matched_data$pair_id, matched_data$treatment), ]

# Check covariate balance using cobalt
balance <- bal.tab(treatment ~ baseline_pain + baseline_urgency + baseline_frequency +
                     treatment_pain + treatment_urgency + treatment_frequency,
                   data = matched_data,
                   estimand = "ATT",
                   subclass = matched_data$pair_id)

# Extract the balance table and add the Covariate names
balance_table <- as.data.frame(balance$Balance)
balance_table$Covariate <- rownames(balance_table)

# Adjust the selection of columns based on actual column names
balance_table <- balance_table[, c("Covariate", "Type", "Diff.Adj")]

# Rename columns for clarity
colnames(balance_table) <- c("Covariate", "Type", "Std.Diff.Matched")

# Display the balance table
cat("\nCovariate Balance Table After Matching:\n")
print(balance_table)

# Outcome Analysis using Cox Proportional Hazards Model

# Check the event distribution
event_table <- table(Treatment = matched_data$treatment, Event = matched_data$event)
cat("\nEvent Table:\n")
print(event_table)

# Fit the Cox proportional hazards model
cox_model <- coxph(Surv(time, event) ~ treatment, data = matched_data)
cox_summary <- summary(cox_model)

# Display Cox model results
cat("\n\n**Cox Proportional Hazards Model Results**\n")
cox_results <- data.frame(
  Coefficient = cox_summary$coefficients[, "coef"],
  Exp_Coefficient = cox_summary$coefficients[, "exp(coef)"],
  Std_Error = cox_summary$coefficients[, "se(coef)"],
  z_value = cox_summary$coefficients[, "z"],
  p_value = cox_summary$coefficients[, "Pr(>|z|)"]
)
rownames(cox_results) <- rownames(cox_summary$coefficients)
print(round(cox_results, 4))

# Global test statistics
global_tests <- data.frame(
  Test = c("Likelihood ratio test", "Wald test", "Score (logrank) test"),
  Statistic = c(cox_summary$logtest["test"],
                cox_summary$waldtest["test"],
                cox_summary$sctest["test"]),
  df = c(cox_summary$logtest["df"],
         cox_summary$waldtest["df"],
         cox_summary$sctest["df"]),
  p_value = c(cox_summary$logtest["pvalue"],
              cox_summary$waldtest["pvalue"],
              cox_summary$sctest["pvalue"])
)

# Round numeric columns only
global_tests[, c('Statistic', 'df', 'p_value')] <- round(global_tests[, c('Statistic', 'df', 'p_value')], 4)

# Display the global test statistics
cat("\n\n**Global Test Statistics**\n")
print(global_tests)

# Sensitivity Analysis
# Prepare data for sensitivity analysis
sens_data <- matched_data %>%
  group_by(pair_id) %>%
  summarise(
    treated_event = event[treatment == 1],
    control_event = event[treatment == 0]
  ) %>%
  ungroup()

# Remove pairs with missing event data
sens_data <- sens_data %>%
  filter(!is.na(treated_event) & !is.na(control_event))

# Calculate differences
sens_data <- sens_data %>%
  mutate(
    diff = treated_event - control_event
  )

# Remove pairs where diff is NA or zero
discordant_pairs <- sens_data %>%
  filter(!is.na(diff) & diff != 0)

# Differences vector
y <- discordant_pairs$diff

# Check if y is non-empty
if (length(y) == 0) {
  stop("No discordant pairs available for sensitivity analysis.")
}

# Ensure y is numeric
y <- as.numeric(y)

# Perform sensitivity analysis using senmw()
gamma_values <- seq(1, 2, by = 0.1)
sensitivity_results <- data.frame(Gamma = gamma_values, p_value = NA)

for (i in seq_along(gamma_values)) {
  gamma <- gamma_values[i]
  
  # Perform sensitivity analysis
  senmw_result <- senmw(y = y, gamma = gamma)
  
  # Extract the two-sided p-value
  p_val_two_sided <- senmw_result$pval
  
  # Adjust for one-sided test
  p_val_one_sided <- p_val_two_sided / 2
  
  # Adjust p-value based on the direction of the effect
  if (mean(y) < 0) {
    p_val_one_sided <- 1 - p_val_one_sided
  }
  
  # Store the p-value
  sensitivity_results$p_value[i] <- p_val_one_sided
}

# Display sensitivity analysis results
cat("\n\n**Sensitivity Analysis Results**\n")
print(sensitivity_results)

# Plot the sensitivity analysis results
ggplot(sensitivity_results, aes(x = Gamma, y = p_value)) +
  geom_line(color = "blue") +
  geom_point(color = "red") +
  geom_hline(yintercept = 0.05, linetype = "dashed", color = "darkgreen") +
  labs(
    title = "Sensitivity Analysis using senmw()",
    x = expression(Gamma),
    y = "One-sided P-value"
  ) +
  theme_minimal()
