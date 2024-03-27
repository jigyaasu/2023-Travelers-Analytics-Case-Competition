rm(list = ls())

# Required Libraries
library(boot)
library(car)
library(statmod)
library(ggplot2)
library(GGally)
library(vip)

# Setting the working directory
setwd("/Users/Harsh/Downloads/2023 Travelers Case Competition/2023-travelers-university-competition/")

# Importing the training and test data
training_data <- read.csv("InsNova_data_2023_train.csv")
test_data <- read.csv("InsNova_data_2023_vh.csv")

training_data$veh_age = as.factor(training_data$veh_age)
training_data$agecat = as.factor(training_data$agecat)
training_data$trm_len = as.factor(training_data$trm_len)
training_data$e_bill = as.factor(training_data$e_bill)
training_data$high_education_ind = as.factor(training_data$high_education_ind)

test_data$veh_age <- as.factor(test_data$veh_age)
test_data$agecat <- as.factor(test_data$agecat)
test_data$trm_len <- as.factor(test_data$trm_len)
test_data$e_bill <- as.factor(test_data$e_bill)
test_data$high_education_ind <- as.factor(test_data$high_education_ind)

cat_vars <- c("veh_body", "veh_age", "gender", "area", "agecat", "engine_type", "veh_color", 
              "marital_status", "e_bill" , "time_of_week_driven" ,"time_driven" , 
              "trm_len", "high_education_ind")

quant_vars <- c("claimcst0", "numclaims", "veh_value", "exposure", 
                "max_power", "driving_history_score", "credit_score")

training_data_cat <- training_data[, cat_vars]
training_data_quant <- training_data[, quant_vars]

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


# Function to conduct chi-square tests for all pairs of categorical variables
chi_sq_test_all_pairs <- function(data, cat_vars) {
  results <- data.frame(variable1 = character(), variable2 = character(), p_value = numeric())
  
  for (i in 1:(length(cat_vars) - 1)) 
    { 
    for (j in (i + 1):length(cat_vars)) 
      { 
      contingency_table <- table(data[[cat_vars[i]]], data[[cat_vars[j]]])
      chi_sq_result <- chisq.test(contingency_table)
      
      results <- rbind(results, data.frame(
        variable1 = cat_vars[i],
        variable2 = cat_vars[j],
        p_value = chi_sq_result$p.value
      ))
    }
  }
  
  return(results)
}

chi_sq_res = chi_sq_test_all_pairs(training_data, cat_vars)
chi_sq_res$reln = ifelse(chi_sq_res$p_value < 0.05, yes = "Yes", no = "No")
chi_sq_res_final = chi_sq_res[chi_sq_res$reln == "Yes", ]

boxplot(claimcst0~veh_body, data = training_data)
thsd <- TukeyHSD(aov(claimcst0~veh_body, data=subset(training_data, clm>0)))
thsd

boxplot(claimcst0 ~ gender, data = subset(training_data, clm>0))
boxplot(claimcst0 ~ area, data = subset(training_data, clm>0))
boxplot(claimcst0 ~ factor(agecat), data = subset(training_data, clm>0))

pairs(training_data[,c("claimcst0", "veh_value", "exposure", 
                       "max_power", "driving_history_score", "credit_score")])
pairs(subset(training_data,clm>0)[,c("claimcst0", "veh_value", "exposure", 
                                     "max_power", "driving_history_score", "credit_score")])


#scatterplotMatrix(~claimcst0 + veh_value + exposure + 
                    #max_power + driving_history_score + credit_score,
                  #data = training_data)

ggpairs(training_data[, c("claimcst0", "veh_value", "exposure", 
                          "max_power", "driving_history_score", "credit_score")])

hist(training_data$claimcst0, main = "Histogram of claimcst0", xlab = "claimcst0")

training_data_clm = subset(training_data, clm > 0)
hist(training_data_clm$claimcst0/training_data_clm$numclaims, 
     main = "Histogram of claimcst0/numclaims", xlab = "claimcst0/numclaims")

# Null Model
calculate_normalized_gini(training_data$claimcst0, mean(training_data$claimcst0))


# Model 1: Fitting a linear regression model to the full training data 
full_ols <-  lm(claimcst0 ~ veh_value+exposure+veh_body+veh_age+gender+
                  area+factor(agecat)+engine_type+max_power+driving_history_score 
                +veh_color +marital_status +e_bill 
                +time_of_week_driven +time_driven +factor(trm_len) + credit_score 
                + high_education_ind, data = training_data)

summary(full_ols)

calculate_normalized_gini(training_data$claimcst0, predict(full_ols))

# Model 2: GLM Tweedie distribution model
tweedie_model <- glm((claimcst0/exposure) ~ (veh_value+area+veh_age+gender+factor(agecat))^2, 
             data = training_data,  family = tweedie(var.power=1.3,link.power=0))
summary(tweedie_model)

calculate_normalized_gini(training_data$claimcst0, predict(tweedie_model, type="response")*training_data$exposure)

# Model 3: Modeling numclaims using ols with all variables

numclaim_ols <- lm(numclaims ~ veh_value+exposure+veh_body+veh_age+gender+
                  area+factor(agecat)+engine_type+max_power+driving_history_score +veh_color 
                  +marital_status +e_bill +time_of_week_driven +time_driven +factor(trm_len) + 
                    credit_score + high_education_ind, data=training_data)

summary(numclaim_ols)


# Model 4: Modeling claimcst0/numclaims using ols with all variables
claimcst0_ols <- lm((claimcst0/numclaims) ~ veh_value+exposure+veh_body+veh_age+gender+
                     area+factor(agecat)+engine_type+max_power+driving_history_score +veh_color 
                   +marital_status +e_bill +time_of_week_driven +time_driven +factor(trm_len) + 
                     credit_score + high_education_ind, data=training_data_clm)

summary(claimcst0_ols)

# Two part model
# Poisson Regression or Negative binomial?
print(paste("Mean of numclaim =", round(mean(training_data$numclaim), 6)))
print(paste("Var of numclaim = ", round(var(training_data$numclaim), 6)))

# model 1: count (offset(log(exposure)) - poisson regression model
pm.count = glm(numclaims ~ offset(log(exposure))+(veh_value+veh_age+gender+area+factor(agecat)+
                                                  +trm_len)^2, 
               family = poisson, data = training_data)

pm.count.sub = step(pm.count, test = "Chi")
dp <- sum(residuals(pm.count.sub,type="pearson")^2)/pm.count.sub$df.res
summary(pm.count.sub)

pm.small = glm(numclaims ~ offset(log(exposure))+veh_value+area+factor(agecat)+
                 veh_value:veh_age+veh_value:area+gender:area, 
               family = poisson, data = training_data)
summary(pm.small)

# model 2: claimcst0/numclaims - inverse gaussian model
ivg <- glm((claimcst0/numclaims) ~ veh_value+gender+area+factor(agecat)+factor(trm_len)+
             driving_history_score,
           family = inverse.gaussian(link = "log"), 
           data = subset(training_data, clm > 0))
ivg.sub = step(ivg)
summary(ivg.sub)

vip(pm.small, method = "firm", train = training_data)
vip(inverse_g, method = "firm", train = training_data)

