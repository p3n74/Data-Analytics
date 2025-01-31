# Simulation Output Recap

## Valid Simulations:  1000 / 1000
**True Treatment Effect:**  2  
**Bias:**  -0.6411  
**RMS Error:**  0.7229  
**Coverage 95% CI:**  0.506  

---

### 1. Valid Simulations: 1000 / 1000
**Meaning:** All 1,000 simulation iterations were successfully completed without being skipped due to errors or insufficient data in any stratum.

**Implication:** Your simulation setup is robust in handling different data scenarios, allowing for comprehensive analysis across all intended iterations.

---

### 2. True Treatment Effect: 2
**Meaning:** In your simulation design, the actual or ground truth effect of the treatment is set to 2. This serves as the benchmark against which all estimated treatment effects from the simulation are compared.

---

### 3. Bias: -0.6411
**Definition:** Bias measures the average difference between the estimated treatment effects and the true treatment effect across all simulations.

\[\text{Bias} = \text{Mean} (\text{Estimated Treatment Effects}) - \text{True Treatment Effect} \]

**Interpretation:**
- **Negative Bias (-0.6411):** On average, the BRSM method underestimates the true treatment effect by approximately 0.6411 units.
- **Severity:** A bias of -0.6411 is substantial, considering the true effect is 2. This indicates a systematic underestimation of the treatment effect by the method.

**Implications:**
- **Systematic Error:** The method consistently produces estimates lower than the actual effect.

**Potential Causes:**
- **Incomplete Covariate Balance:** Despite matching, covariates may not be fully balanced between treated and control groups.
- **Model Misspecification:** The outcome model (e.g., linear regression) may not appropriately capture the relationship between treatment and outcome.
- **Matching Algorithm Limitations:** The chosen matching parameters or methods might not be optimal for the data structure.

---

### 4. Root Mean Squared Error (RMSE): 0.7229
**Definition:** RMSE combines both the bias and the variance of the estimated treatment effects, providing an overall measure of estimation accuracy.

\[\text{RMSE} = \sqrt{\text{Mean} ((\text{Estimated Treatment Effects} - \text{True Treatment Effect})^2)} \]

**Interpretation:**
- **Value (0.7229):** Indicates that, on average, the estimated treatment effects deviate from the true effect by approximately ±0.7229 units.
- **Context:** Considering the true treatment effect is 2, an RMSE of ~0.723 reflects a moderate level of error.

**Implications:**
- **Combined Effect of Bias and Variance:** RMSE accounts for both systematic errors (bias) and random errors (variance). A high RMSE suggests room for improvement in both areas.
- **Relative Performance:** While better than RMSE values nearing the magnitude of the treatment effect itself, there's still significant error to address.

---

### 5. Coverage Probability (95% CI): 0.506
**Definition:** Coverage Probability assesses the proportion of simulated 95% confidence intervals that successfully contain the true treatment effect.

**Interpretation:**
- **Value (0.506):** Approximately 50.6% of the 95% confidence intervals included the true treatment effect of 2.
- **Desired Standard:** Ideally, around 95% of the confidence intervals should capture the true effect, aligning with the nominal confidence level.

**Implications:**
- **Under-Coverage:** A coverage of ~50.6% is significantly below the expected 95%, indicating that the confidence intervals are too narrow or the estimates are biased.
- **Reliability of Inference:** The confidence intervals generated are unreliable, meaning that statistical inferences based on them may be misleading.

---

## Interpreting the Results
Your simulation results reveal several critical issues with the current implementation of the Balanced Risk Set Matching (BRSM) method:

- **Significant Negative Bias (-0.6411):**
  - The method consistently underestimates the true treatment effect.
  - This systematic bias suggests that there's a fundamental issue in how matching or estimation is being conducted.

- **Moderate RMSE (0.7229):**
  - Reflects both the bias and variability in the estimates.
  - While not excessively high, it's indicative of notable estimation errors.

- **Poor Coverage Probability (0.506):**
  - Confidence intervals are failing to capture the true effect in approximately half of the simulations.
  - This undermines the validity of any statistical conclusions drawn from these intervals.

---

## Potential Causes and Recommendations

### 1. Treatment Assignment and Propensity Score Calibration
#### Cause:
- The treatment assignment process may still be skewed, leading to imbalanced groups even after matching.
- Extreme propensity scores might result in poor overlap between treated and control groups.

#### Recommendations:
- **Re-examine Propensity Score Model:**
  - Ensure that the propensity score model accurately captures the relationship between covariates and treatment assignment.
  - Incorporate interaction terms or non-linear relationships if necessary.

- **Propensity Score Distribution:**
  - Plot the distribution of propensity scores for treated and control groups to assess overlap.
  - Use techniques like caliper matching to restrict matches within a certain propensity score range.

```r
# Example: Adding a caliper
m.out <- matchit(treated ~ X1 + X2 + X3 + X4 + X5, 
                data = strata_data, 
                method = "nearest", 
                distance = "logit",
                ratio = desired_ratio,
                caliper = 0.1)
```

### 2. Matching Quality and Methodology
#### Cause:
- The matching algorithm or parameters may not be effectively balancing covariates between treated and control groups within each risk set.

#### Recommendations:
- **Assess Covariate Balance Post-Matching:**

```r
summary(m.out)
```

- **Use visual tools like love plots to assess balance:**

```r
library(cobalt)
love.plot(m.out, binary = "std")
```

- **Experiment with Different Matching Methods:**

```r
# Optimal Matching
m.out <- matchit(treated ~ X1 + X2 + X3 + X4 + X5, 
                data = strata_data, 
                method = "optimal", 
                distance = "logit",
                ratio = desired_ratio)
```

- **Adjust Matching Ratio:**

```r
# Limiting to a maximum of 2 controls per treated unit
m.out <- matchit(treated ~ X1 + X2 + X3 + X4 + X5, 
                data = strata_data, 
                method = "nearest", 
                distance = "logit",
                ratio = 2,
                caliper = 0.1)
```

---

### 3. Outcome Model Specification
#### Cause:
- The outcome model may not accurately reflect the true relationship between treatment, covariates, and outcomes.

#### Recommendations:
- **Include Relevant Covariates:**

```r
model <- lm(Y ~ treated + X1 + X2 + X3 + X4 + X5, data = matched_data)
```

- **Check for Model Misspecification:**

```r
model <- lm(Y ~ treated + X1 + I(X1^2) + X2 + I(X2^2), data = matched_data)
```

---

