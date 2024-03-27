rm(list = ls())

# Required Libraries
library(boot)
library(car)
library(statmod)
library(patchwork)
library(vip)
library(pdp)

# Setting the working directory
setwd("/Users/Harsh/Downloads/2023 Travelers Case Competition/2023-travelers-university-competition/")

# Importing the training and test data
training_data <- read.csv("InsNova_data_2023_train.csv")
test_data <- read.csv("InsNova_data_2023_vh.csv")

training_data$veh_age <- as.factor(training_data$veh_age)
training_data$agecat <- as.factor(training_data$agecat)
training_data$trm_len <- as.factor(training_data$trm_len)
training_data$e_bill <- as.factor(training_data$e_bill)
training_data$high_education_ind <- as.factor(training_data$high_education_ind)

test_data$veh_age <- as.factor(test_data$veh_age)
test_data$agecat <- as.factor(test_data$agecat)
test_data$trm_len <- as.factor(test_data$trm_len)
test_data$e_bill <- as.factor(test_data$e_bill)
test_data$high_education_ind <- as.factor(test_data$high_education_ind)

# Defining the Gini function to measure the Gini coefficient of the model (as defined on Kaggle)
calculate_gini <- function(actual_values, predicted_values) {
  gini_df <- data.frame(actual_values = actual_values, predicted_values = predicted_values)
  gini_df <- gini_df[order(gini_df$predicted_values, decreasing = TRUE), ]
  gini_df$random <- (1:nrow(gini_df)) / nrow(gini_df)
  total_positive <- sum(gini_df$actual_values)
  gini_df$cumulative_pos_found <- cumsum(gini_df$actual_values)
  gini_df$lorentz <- gini_df$cumulative_pos_found / total_positive
  gini_df$gini <- gini_df$lorentz - gini_df$random
  return(sum(gini_df$gini))
}

# Defining the Normalized Gini function
calculate_normalized_gini <- function(actual_values, predicted_values) {
  gini_model <- calculate_gini(actual_values, predicted_values)
  gini_baseline <- calculate_gini(actual_values, actual_values)
  return(gini_model / gini_baseline)
}

# Function for bootstrapping the data
boot_glm <- function(formula, input_data, glm_family, index) 
{
  boot_d <- input_data[index, ] # bootstrapped sample for fitting the required GLM  
  model_fit <- glm(formula, glm_family, data = boot_d)
  return(coef(model_fit)) 
}

# Histogram of the target variable - claimcst0
# The target variable is highly skewed towards the right
hist(training_data$claimcst0, main="Histogram of claimcst0", xlab="claimcst0", breaks = 15)

# Distribution of the 'clm' column and the 'numclaim' column
# Both tables confirm that the training data is highly imbalanced in favor of clm = 0 and numclaim = 0
table(training_data$clm)
table(training_data$numclaim)

# Fitting a linear regression model to the full training data 
full_ols <-  lm(claimcst0 ~ veh_value+exposure+veh_body+veh_age+gender+
             area+factor(agecat)+engine_type+max_power+driving_history_score +veh_color +marital_status +e_bill 
           +time_of_week_driven +time_driven +trm_len + credit_score + high_education_ind, data=training_data)

summary(full_ols)

calculate_normalized_gini(training_data$claimcst0, predict(full_ols))

# Looking at the ANOVA results for the full_ols model
Anova(full_ols)

# Fitting a linear regression model treating numclaims as the target variable
# This will help us in the first step of our two part model (Count * Severity) where we model numclaims
numclaim_ols <-  lm(numclaims ~ veh_value+exposure+veh_body+veh_age+gender+
              area+factor(agecat)+engine_type+max_power+driving_history_score +veh_color 
              +marital_status +e_bill +time_of_week_driven +time_driven +trm_len + credit_score 
              + high_education_ind, data=training_data)

summary(numclaim_ols)

# Fitting a tweedie distribution model to the target variable claimcst0 to get further insights (using full training data)
tweedie_full <- glm(claimcst0 ~ veh_value+exposure+veh_body+veh_age+gender+
               area+factor(agecat)+engine_type+max_power+driving_history_score +veh_color 
             +marital_status +e_bill +time_of_week_driven +time_driven +trm_len + credit_score + 
               high_education_ind, data=training_data,  family=tweedie(var.power=1.3,link.power=0))

summary(tweedie_full)

# Model 1: Modeling the count variable numclaims using possion regression with (offset(log(exposure))
numclaim_poisson <- glm(numclaims ~ 0 + offset(log(exposure))+factor(agecat)+area+veh_value+veh_age+
                veh_value:veh_age+area:veh_value, family = poisson, data = subset(training_data))

summary(numclaim_poisson)

pred_model1 <- predict(numclaim_poisson, newdata = test_data, type="response")

numclaim_boot <- boot(data = subset(training_data), statistic = boot_glm, R = 10, 
                     formula = formula(numclaim_poisson), 
                     glm_family = poisson, parallel="multicore")

cbind(coef(numclaim_poisson), colMeans(numclaim_boot$t))

numclaim_boot_final <- numclaim_poisson

numclaim_boot_final$coefficients <- colMeans(numclaim_boot$t)

pred_model1_boot <- predict(numclaim_boot_final , newdata = test_data, type="response")


# Model 2: Modeling claimcst0/numclaims (Severity model) using Inverse Gaussian model
inverse_g <- glm((claimcst0/numclaims) ~ gender + veh_age + agecat,
               family = inverse.gaussian(link="log"),
               data = subset(training_data, clm > 0))

inverse_g_pred <- predict(inverse_g, newdata = test_data, type = "response")

inverse_g_boot <- boot(data = subset(training_data, clm > 0 & veh_value > 0), 
               statistic = boot_glm, R = 10, formula = formula(inverse_g), 
               glm_family = inverse.gaussian(link="log"), parallel = "multicore")

cbind(coef(inverse_g), colMeans(inverse_g_boot$t))

inverse_g_final <- inverse_g
inverse_g_final$coefficients <- colMeans(inverse_g_boot$t)
inverse_g_final_pred <- predict(inverse_g_final, newdata = test_data, type="response")

y_hat_test = predict(numclaim_boot_final, newdata = test_data, type = "response")*predict(inverse_g_final, 
                                                                      newdata = test_data, type = "response")

y_hat_final = predict(numclaim_boot_final, newdata = training_data, type = "response")*predict(inverse_g_final, 
                                                                      newdata = training_data, type = "response")

calculate_normalized_gini(training_data$claimcst0, y_hat_final)
test_data$Pred_Claim = y_hat_test

# Some useful plots for variable selection  
boxplot(y_hat_test ~ test_data$gender)
boxplot(y_hat_test ~ test_data$veh_age)

ggplot(data = test_data, aes(x = exposure, y = Pred_Claim)) +
  geom_point() +
  labs(title = "Predicted claim cost vs exposure", x = "Exposure", y = "Predicted claim cost")

ggplot(data = test_data, aes(x = veh_value, y = Pred_Claim)) +
  geom_point() +
  labs(title = "Predicted claim cost vs Vehicle value", x = "Vehicle value", y = "Predicted claim cost")

plot1 <- ggplot(data = test_data, aes(x = gender, y = Pred_Claim)) +
  geom_boxplot() +
  labs(title = "Predicted claim cost vs Gender", x = "Gender", y = "Predicted claim cost")

plot2 <- ggplot(data = test_data, aes(x = veh_age, y = Pred_Claim)) +
  geom_boxplot() +
  labs(title = "Predicted claim cost vs Vehicle age", x = "Vehicle age", y = "Predicted claim cost")

plot3 <-ggplot(data = test_data, aes(x = agecat, y = Pred_Claim)) +
  geom_boxplot() +
  labs(title = "Predicted claim cost vs Age category", x = "Age category", y = "Predicted claim cost")

plot4 <-ggplot(data = test_data, aes(x = area, y = Pred_Claim)) +
  geom_boxplot() +
  labs(title = "Predicted claim cost vs Area", x = "Area", y = "Predicted claim cost")

# Arrange the plots into a grid
combined_plots <- plot1 + plot2 + plot3 + plot4

# Display the combined plot
combined_plots

training_data_clm = subset(training_data, clm > 0)
plot(training_data_clm$veh_value, training_data_clm$claimcst0)
plot(training_data_clm$exposure, training_data_clm$claimcst0)

vip(numclaim_poisson, method = "firm", train = training_data)
vip(inverse_g, method = "firm", train = training_data)
