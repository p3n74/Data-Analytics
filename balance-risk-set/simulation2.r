# 2nd Iteration 

# Things to address


# high negative bias
# Incorrect treatment assignment, possibly due to
# being not properly balanced or may not be a proper binary
# BSRM methods may not be effective 
# Possible model misspecification


# high RSME
# substantial deviation between estimated effects and true effects
# High variability in estimates due to inconsistent matching
# large bias inflates RSME

# Zero Coverage reliability
# Confidence intervals are completely unreliable
# CI's are too narrow and may not be reflective of the true variability in esitmates
# The negative bias pulls the CI away from the true effect


# install.packages(c("MatchIt", "dplyr"))

library(MatchIt)
library(dplyr)
library(boot)
library(MASS)
library(optmatch)

# Simulate dataset based on the paper

# Simulation Parameters

set.seed(69) # seed for reproductibility

n_treated <- 2500 
n_control <- 2500 

p_cov <- 5	# Number of Covariates
beta <- 2	# True Treatment Effect
sigma <- 1	# STD dev of error term
n_sim <- 5000 # Number of simulation runs

# Storage for estimates

estimates <- numeric(n_sim)		# Estimated treatment effects
biases <- numeric(n_sim)		# Biases
lower_ci <- numeric(n_sim)		# Lowebounds, 95% CI
upper_ci <- numeric(n_sim)		# Upperbounds 95% CI
coverage <- logical(n_sim)		# Coverage Indicator

# Simulation loop to perform data generation, matching and treatment
# effect estimation and aswell as metric calculation

for (i in 1:n_sim){

	# = = = Data Generation = = = 

	# Generate covariates from a multivariate normal distribution

	# Assume independence

	# Adjust covariance structure if needed
	
	# V2 -> introduced covariate correlation


	cov_matrix <- matrix(0.3, nrow = p_cov, ncol = p_cov)
	diag(cov_matrix) <- 1
	X <- mvrnorm(n = n_treated + n_control, mu = rep(0, p_cov), Sigma = cov_matrix)

	# X <- matrix(rnorm((n_treated + n_control) * p_cov), ncol = p_cov)

	# Standardize

	X <- scale(X)
	
	colnames(X) <- paste0("X", 1:p_cov)

	# Visualize

	data <- as.data.frame(X)




	# = = = Treatement Assignment = = =

	# caculate the propensity score using a logistical model
	# use a linear combination of covariates
	# adjust the model as per the specifications in the paper

	# using fixed coefficients due to extreme propensity scores
	beta_ps <- rep(0.2, p_cov)
	propensity_logit <- X %*% matrix(beta_ps, ncol = 1)
	# randomize coefficients

	propensity_scores <- plogis(propensity_logit)

	propensity_scores <- pmin(pmax(propensity_scores, 0.05), 0.95)


	# assign treatement based on propensity scores
	data$treated <- rbinom(n_treated + n_control, 1, propensity_scores)

	unique_treated_values <- unique(data$treated)
	if(length(unique_treated_values) < 2){
		cat("Iteration ", i, " No two levels, skipping")
		next
	}

	# = = = Outcome Generation = = = 

	# Outcomes are generated and influenced by treatment and covariates

	# Y = X-Contribution + treatment effect + noise

	data$Y <- X %*% matrix(rep(1, p_cov), ncol = 1) +
			beta * data$treated +
			rnorm(n_treated + n_control, mean = 0, sd = sigma)

	# = = = Propensity Score Matching to Risky Sets

	#giggity

	# Estimate propensity score using logistical regression
	
	ps_model <- glm(treated ~ ., data = data, family = binomial)
	data$ps <- predict(ps_model, type = "response")

	# define how much risky sets

	#giggity

	# basically quintiles
	n_strata <- 5
	data$strata <- ntile(data$ps, n_strata)

	# Init matched dataset

	# Strata = stratum, a subset of the population

	matched_data <- NULL

	# perfrom matching with each strata

	for(s in 1:n_strata){

		# Data subset for the current strata

		strata_data <- data %>% filter(strata == s)

		# Check for the presence of treated and control units
		n_treated_strata <- sum(strata_data$treated == 1)
		n_control_strata <- sum(strata_data$treated == 0)

		# Debugging
		cat("Iteration", i, "- Stratum", s, 
			": Treated =", n_treated_strata, 
			", Control =", n_control_strata, "\n")

		if (n_treated_strata < 1 || n_control_strata < 1) {
		# Skip strata without both treated and control units
			next
		}
		# here, apply matching within the strata to the nearest
		# neighbor

		# match the treated units to the control units
		# By replacing the nearest with a specific method
		# as per mentioned in the paper

		# it had this algorithm
		
	
		# Calculate desired_ratio
		desired_ratio <- floor(n_control_strata / n_treated_strata)
		# Ensure at least one control per treated
		desired_ratio <- ifelse(desired_ratio < 1, 1, desired_ratio)
		# Optionally -> set a maximum ratio to prevent extensive matching
		# max_ratio <- 4 # Example maximum ratio
		# desired_ratio <- ifelse(desired_ratio > max_ratio, max_ratio, desired_ratio)

		m.out <- tryCatch({
			matchit(treated ~ X1 + X2 + X3 + X4 + X5,
							data = strata_data,
							method = "optimal",
							distance = "logit",
							ratio = desired_ratio
							# caliper = 0.2
							)
						# added caliper to restrict matches to control units within a range of propensity score
					# Param: 1 Control per treated, adjust if need
		}, error = function(e) {
			cat("Error in matching at Iteration", i, "Stratum", s, ":", conditionMessage(e), "\n")
			return(NULL)
		})

		if (!is.null(m.out)) {
			# Extract matched data
			strata_matched <- match.data(m.out)
    
			# Combine matched strata
			matched_data <- bind_rows(matched_data, strata_matched)
		}
	}


	# = = = Estimation of the treatment effect = = = 

	# check nulls

	if (is.null(matched_data) || nrow(matched_data) == 0){
			estimates[i] <- NA
			biases[i] <- NA
			lower_ci[i] <- NA
			upper_ci[i] <- NA
			coverage[i] <- NA
			next
		}

	# Use linear regression on the matched data to estimate treatment offect

	model <- lm(Y ~ treated, data = matched_data)
	est <- coef(model)["treated"]
	estimates[i] <- est

	# calculate the confidence interval


	# ext treatement effect
		
	coef_fun <- function(data, indices){
			d <- data[indices, ]
			fit <- lm(Y ~ treated, data = d)
			return(coef(fit)["treated"])
		}

	# Boot the confidence intervals

	# Underestimated variability
	boot_results <- tryCatch({
		boot(data = matched_data, statistic = coef_fun, R = 500)  # Increased R
	}, error = function(e) {
		cat("Error in bootstrapping at Iteration", i, ":", conditionMessage(e), "\n")
		return(NULL)
	})	
	ci <- boot.ci(boot_results, type = "perc")

	if(!is.null(ci$percent)){
			lower_ci[i] <- ci$percent[4]
			upper_ci[i] <- ci$percent[5]
		} else {
			lower_ci[i] <- NA
			upper_ci[i] <- NA
		}

	# = = = Bias Calculation = = =

	biases[i] <- est - beta
	coverage[i] <- (lower_ci[i] <= beta) & (upper_ci[i] >= beta)


	# progress bar to make things cooler

	if (i %% 100 == 0){
			cat("Complete simulation: ", i, "\n")
	}
}




# = = = Evaluation metrics = = = 

	# clean by removing skipped data

	valid_indices <- !is.na(estimates)
	valid_estimates <- estimates[valid_indices]
	valid_biases <- biases[valid_indices]
	valid_lower_ci <- lower_ci[valid_indices]
	valid_upper_ci <- upper_ci[valid_indices]
	valid_coverage <- coverage[valid_indices]


	# Summary Stats

	mean_estimate <- mean(valid_estimates)
	bias_mean <- mean(valid_biases)
	rmse <- sqrt(mean(valid_biases^2))
	coverage_prob <- mean(valid_coverage)



	# Display

	cat("\n")
	cat("Valid Simulations: ", length(valid_estimates), "/", n_sim, "\n")
	cat("True Treatment Effect: ", beta, "\n")
	cat("Bias: ", round(bias_mean, 4), "\n")
	cat("RMS Error: ", round (rmse, 4), "\n")
	cat("Coverage 95% CI: ", round(coverage_prob, 4), "\n")


