# Simulation Results with Optimal Matching

## Valid Simulations:  1000 / 1000  
**True Treatment Effect:**  2  
**Bias:**  0.0819  
**RMS Error:**  0.2682  
**Coverage 95% CI:**  0.965  

---

## Interpretation of the Results  

### **Bias: 0.0819**  
The estimated treatment effect was, on average, 0.0819 units higher than the true effect of 2.  

**Percentage Bias Calculation:**  
\[
\text{Percentage Bias} = \left( \frac{0.0819}{2} \right) \times 100\% \approx 4.1\%
\]  

This indicates a minimal positive bias, suggesting that the estimation method produced results very close to the true treatment effect.  

### **Root Mean Squared Error (RMSE): 0.2682**  
The RMSE, which accounts for both bias and variability, was 0.2682. This low value suggests that the overall estimation error was minimal.  

These findings indicate that the estimates were both accurate and precise, demonstrating that the matching method effectively reduced both systematic and random errors.  

### **Coverage Probability (95% CI): 0.965**  
The 95% confidence intervals contained the true treatment effect of 2 in 96.5% of the simulations.  

This suggests that the confidence intervals were well-calibrated, providing a reliable measure of uncertainty in the estimates.  

---

## Comparison with Previous Results Using Nearest Neighbor Matching  

For comparison, the simulation results using nearest neighbor matching were as follows:  

- **Valid Simulations:**  1000 / 1000  
- **True Treatment Effect:**  2  
- **Bias:**  -1.7678  
- **RMS Error:**  1.7928  
- **Coverage 95% CI:**  0  

### **Key Observations**  

#### **Reduction in Bias**  
- Nearest neighbor matching produced a substantial negative bias of -1.7678, leading to severe underestimation of the treatment effect.  
- In contrast, optimal matching reduced the bias to 0.0819, yielding estimates much closer to the true effect.  

#### **Improved RMSE**  
- The RMSE for nearest neighbor matching was 1.7928, indicating high estimation error.  
- Optimal matching significantly improved precision, reducing the RMSE to 0.2682.  

#### **Increased Coverage Probability**  
- The coverage probability for nearest neighbor matching was 0, meaning that none of the confidence intervals contained the true treatment effect.  
- With optimal matching, the coverage probability increased to 0.965, aligning closely with the expected 95% level.  

---

## Explanation for the Improved Performance of Optimal Matching  

### **1. Enhanced Covariate Balance**  
- Optimal matching selects matches by minimizing overall distance across all pairs, rather than making sequential choices.  
- This approach results in better covariate balance compared to nearest neighbor matching, which does not account for the global best set of matches.  

### **2. Reduced Bias**  
- The improved balance in covariates reduces confounding effects, leading to more accurate estimates of the treatment effect.  
- Optimal matching is particularly effective in cases where covariate distributions differ significantly between treated and control groups.  

### **3. Greater Efficiency**  
- Unlike nearest neighbor matching, which forces a fixed 1:1 ratio, optimal matching allows for variable matching ratios.  
- This flexibility maximizes the use of available control units, leading to greater precision in the estimates.  

### **4. Handling Sparse Strata**  
- Optimal matching performs better in cases where some treated units have fewer available control matches.  
- This flexibility ensures consistent performance across all simulations, even in scenarios with limited matching options.  

---

# Simulation Results with Large Sample and Increased Covariates

## Valid Simulations: 5000 / 5000  
**True Treatment Effect:** 2  
**Bias:** 0.075  
**RMS Error:** 0.1224  
**Coverage 95% CI:** 0.961  

---

## Analysis of the Simulation Results

The latest simulation results demonstrate significant improvements in estimating the treatment effect using the Balanced Risk Set Matching method, especially with a larger sample size and potentially more covariates. Here’s a detailed interpretation of the results:

### 1. Valid Simulations: 5000 / 5000  
**Meaning:** All 5,000 simulation iterations were successfully completed without errors or skipped iterations.  
**Implication:** The larger number of simulations enhances the reliability and robustness of the findings, providing more precise estimates of performance metrics.

### 2. True Treatment Effect: 2  
**Meaning:** The actual or true treatment effect is set to 2 in the simulation.  
**Purpose:** This known value acts as a benchmark against which the estimated treatment effects are compared.

### 3. Bias: 0.075  
**Definition:** Bias measures the average difference between the estimated treatment effects and the true treatment effect.  

\[
\text{Bias} = \text{Mean}(\hat{\beta}) - \beta_{\text{true}}
\]

**Interpretation:** A bias of 0.075 indicates that, on average, the estimated treatment effect is 0.075 units higher than the true effect of 2.

#### Percentage Bias:
\[
\text{Percentage Bias} = \left( \frac{0.075}{2} \right) \times 100\% = 3.75\%
\]

**Implication:** The small and acceptable bias suggests that the estimation method is nearly unbiased, accurately recovering the true treatment effect.

### 4. Root Mean Squared Error (RMSE): 0.1224  
**Definition:** RMSE combines both the bias and the variability (standard error) of the estimator.  

\[
\text{RMSE} = \text{Mean}\left( (\hat{\beta} - \beta_{\text{true}})^2 \right)
\]

**Interpretation:** An RMSE of 0.1224 means that the estimated treatment effects deviate from the true effect by about 0.1224 units on average.  
**Implication:** The low RMSE indicates that the estimates are both accurate (low bias) and precise (low variance).

### 5. Coverage Probability (95% CI): 0.961  
**Definition:** Coverage Probability is the proportion of times the 95% confidence intervals contain the true treatment effect.

\[
\text{Coverage Probability} = \frac{\text{Number of CIs containing the true effect}}{\text{Total number of simulations}}
\]

**Interpretation:** A coverage of 0.961 means that in 96.1% of the simulations, the 95% confidence intervals included the true treatment effect of 2.  
**Implication:** The coverage is very close to the nominal level of 95%, indicating that the confidence intervals are well-calibrated and the uncertainty in the estimates is accurately quantified.

---

## Key Observations and Implications

### Improved Estimation Accuracy:
- The minimal bias and low RMSE suggest that the Balanced Risk Set Matching method is effectively estimating the true treatment effect with high accuracy.
- The small positive bias suggests slight overestimation, which is acceptable in many practical applications.

### High Precision:
- The low RMSE reflects not only low bias but also low variability in the estimates.
- This precision is likely due to the larger sample size, which reduces estimation variance.

### Reliable Confidence Intervals:
- The high coverage probability demonstrates that the confidence intervals are valid and provide a reliable measure of the uncertainty around the estimates.
- Slightly exceeding the nominal level (96.1% vs. 95%) is acceptable and indicates a conservative estimation.

### Effect of Larger Sample Size:
- Increasing the sample size to 5,000 simulations enhances the statistical power of the study.
- Larger samples improve the matching process by providing more potential matches, leading to better covariate balance between treated and control groups.
- The Law of Large Numbers ensures that sample estimates converge closer to population parameters.

---

## Possible Reasons for Improved Performance

### Enhanced Matching Quality:
- **More Matches Available:** With a larger sample size, there are more control units available for matching, which improves the quality of matches.
- **Better Covariate Balance:** Increased data leads to better balancing of covariates between treated and control groups, reducing bias due to confounding.

### Reduced Variance:
- Larger samples result in more precise estimates due to the reduction in sampling variability.
- The standard errors of the estimates decrease with larger sample sizes, contributing to more accurate confidence intervals.

### Stabilized Estimates:
- With 5,000 simulations, the estimates of bias, RMSE, and coverage become more stable and less influenced by random fluctuations.

