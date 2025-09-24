############################################################ 
# Project1: Linear Regression (Insurance Data)
############################################################

# ----------------------------------------------------------------------------- #
# 0. Set Java Environment (for H2O) -------------------------------------------
# ----------------------------------------------------------------------------- #
Sys.setenv(JAVA_HOME = "C:\\Program Files\\Java\\jdk-24") # Update path if needed
# ----------------------------------------------------------------------------- #
# 1. Load Libraries -----------------------------------------------------------
# ----------------------------------------------------------------------------- #
library(h2o)
library(MLmetrics)
library(recipes)
library(dplyr)
library(caTools)
library(caret)        
library(ggplot2)
library(lmtest)
library(gridExtra)
library(tidyr)
# ----------------------------------------------------------------------------- #
# 2. User Parameters ----------------------------------------------------------
# ----------------------------------------------------------------------------- #
seed <- 123          # Reproducibility
train_frac <- 0.7    # 70% training set
metric <- "rmse"     # Performance metric for grid search
folds <- 5           # Cross-validation folds (can change to 10)
# ----------------------------------------------------------------------------- #
# 3. Load & Inspect Data ------------------------------------------------------
# ----------------------------------------------------------------------------- #
df <- read.csv("C:\\Assignment\\insurance.csv")
any(is.na(df))
# Convert categorical variables to factors
df$sex <- as.factor(df$sex)
df$smoker <- as.factor(df$smoker)
df$region <- as.factor(df$region)
# Optional: check summary
summary(df)
# ----------------------------------------------------------------------------- #
# 3. Explore Target Distribution ---------------------------------------------- 
# -------------------------------------------------------------str---------------- #

# For this demonstration, we are interested in predicting the medical expenses. Let's first assess normality (a requirement for OLS):

hist <- ggplot(df, aes(x = expenses)) +
  geom_histogram(bins = 10, fill = "skyblue", color = "black") +
  theme_minimal() +
  ggtitle("Histogram of Target Variable (expenses)")

hist # some deviation from normality so let's transform it:
# apply transformation to make more normal
df$tranformed_target <- log(df$expenses)  # natural logarithm or try sqrt()

hist_transformed <- ggplot(df, aes(x = tranformed_target)) +
  geom_histogram(bins = 10, fill = "#C1FFC1", color = "black") +
  theme_minimal() +
  ggtitle("Histogram of Target Variable (log expenses)")

hist_transformed # showing less deviation from normality
summary(df)
# specify target and features

# Target: expenses, Features: all others - remove if not so
# remove original target:

df_final <- df %>% select(-expenses) # change this to transformed_target if not being used

target <- "tranformed_target" 
features <- setdiff(names(df_final), target)

# ----------------------------------------------------------------------------- #
# 4. Train/Test Split --------------------------------------------------------- 
# ----------------------------------------------------------------------------- #

set.seed(seed)  
split=sample.split(df_final[[target]],SplitRatio = train_frac) 

training_set=subset(df_final,split==TRUE) 
test_set=subset(df_final,split==FALSE)
# ------------------------------------------------------------------ #
# 5. Data preprocessing (outside H2O) -----------------------------------------
# ----------------------------------------------------------------------------- #

# Note: for this example, there are no categorical predictors, but we will include it in the process anyway.

# as.formula(paste(target, "~ .")) is a generic or dynamic way to create a formula in R, where the target variable name is stored in a variable (called 'target" specified above) instead of being hard-coded.

# 1. define the model so that the function knows what is the target (this defines the recipe)
rec <- recipe(as.formula(paste(target, "~ .")), data = training_set) %>%
  step_normalize(all_numeric_predictors()) %>%  # this must be done first
  step_dummy(all_nominal_predictors(), one_hot = FALSE) # the last category is dropped

# to avoid perfect linearity, one of the categories of the variable during the encoding is dropped. This speeds up the training and improves the stability of the ML model. This is done by setting one_hot = FALSE. 
# 2. Prep the recipe using the training data
rec_prep <- prep(rec, training = training_set)

# 3. Apply (bake) the prepped recipe on the scaled training set 
train_processed <- bake(rec_prep, new_data = NULL)

# 4. Apply (bake) the same transformations to the scaled test set
test_processed <- bake(rec_prep, new_data = test_set)

# define target and features AFTER baking
# ------------------------------
target <- "tranformed_target"
features <- setdiff(colnames(train_processed), target)


summary(train_processed)
summary(test_processed)
# we do not need to scale the target for these methods

# ----------------------------------------------------------------------------- #
# 6. OLS Regression (with inference) ------------------------------------------
# ----------------------------------------------------------------------------- #

ols_model <- lm(as.formula(paste(target, "~ .")), data = train_processed)
summary(ols_model)  # Shows coefficients, p-values


# Residual diagnostics
ols_residuals <- residuals(ols_model)
ols_fitted <- fitted(ols_model)

# Autocorrelation check (Durbin-Watson)
dwtest(ols_model)

# Residual plots
resid_plot <- ggplot(data.frame(fitted = ols_fitted, resid = ols_residuals),
                     aes(x = fitted, y = resid)) +
  geom_point() +
  geom_hline(yintercept = 0, color = "red") +
  theme_minimal() +
  ggtitle("Residuals vs Fitted")

qq_resid_plot <- ggplot(data.frame(resid = ols_residuals),
                        aes(sample = resid)) +
  stat_qq() +
  stat_qq_line(col = "red") +
  theme_minimal() +
  ggtitle("QQ Plot of Residuals")

grid.arrange(resid_plot, qq_resid_plot, ncol = 2)
# Performance of OLS model
preds_ols_train <- predict(ols_model, newdata = train_processed)
preds_ols_test  <- predict(ols_model, newdata = test_processed)
c# We will use the MLmetrics package for metrics

?MLmetrics

## Performance on training set:

# NOTE: train_processed[[target]] is also a generic code instead of using hard-coded targets

# the following functions require the predicted values followed by the actual values
MAE(preds_ols_train,train_processed[[target]])   # MAE - Mean Absolute Error 
RMSE(preds_ols_train,train_processed[[target]])  # RMSE - Root Mean Square Error
R2_Score(preds_ols_train,train_processed[[target]]) # R2_Score (unadjusted though)
## Performance on test set:

MAE(preds_ols_test,test_processed[[target]])
RMSE(preds_ols_test,test_processed[[target]])
R2_Score(preds_ols_test,test_processed[[target]])
# ----------------------------------------------------------------------------- #
# 7. Initialize H2O -----------------------------------------------------------
# ----------------------------------------------------------------------------- #

h2o.init()

# Convert scaled data to H2O dataframe
train_h2o <- as.h2o(train_processed)
test_h2o  <- as.h2o(test_processed)
# ----------------------------------------------------------------------------- #
# 8. Regularized Regression in H2O --------------------------------------------
# ----------------------------------------------------------------------------- #

# H2O uses a glm function to fit a generalized linear model with elastic net regularization, which combines Lasso and Ridge penalties through the alpha parameter. By setting alpha to 0 or 1, we can fit a Ridge or Lasso regression model, respectively. 

## Define hyperparameter grid for Ridge and Lasso ------------------------------
hyper_params <- list(
  lambda = 10^seq(-3, 5, length = 20)  # lambda values (regularization strength)
) 
# note for the above, instead of creating a vector with 20 values, we can use the 'by argument to generate values in a certain increment: 10^seq(-3, 5, by = 1), 
# or we can have set values in a vector: c(10^-3, 10^-1, 10^2)
## Ridge Regression: alpha = 0 -------------------------------------------------

# Step 1: Grid search for lambda with alpha=0 (Ridge)
ridge_grid <- h2o.grid(
  algorithm = "glm",
  grid_id = "ridge_grid",
  x = features,
  y = target, 
  training_frame = train_h2o,
  family = "gaussian",
  alpha = 0,  # ridge
  nfolds = folds,
  keep_cross_validation_predictions = TRUE,
  standardize = FALSE, # this has been done outside of H2O
  seed = seed,
  hyper_params = hyper_params,
  search_criteria = list(strategy = "Cartesian")
)
# Step 2: Get the best lambda based on chosen metric (e.g., RMSE)
grid_perf_ridge <- h2o.getGrid(grid_id = "ridge_grid", 
                               sort_by = metric, 
                               decreasing = TRUE)
print(grid_perf_ridge)

# Step 3: Extract the tuned hyperparameter(s)
best_model_id_ridge <- grid_perf_ridge@model_ids[[1]]
best_ridge_model <- h2o.getModel(best_model_id_ridge)
best_lambda_ridge <- best_ridge_model@parameters$lambda

# Step 4: Refit a single Ridge model with the best lambda 
ridge_final <- h2o.glm(
  x = features,
  y = target,
  training_frame = train_h2o,
  family = "gaussian",
  alpha = 0,
  lambda = best_lambda_ridge,
  standardize = FALSE,
  seed = seed
)
# Save predictions
preds_ridge_train <- h2o.predict(ridge_final, train_h2o)
preds_ridge_test <- h2o.predict(ridge_final, test_h2o)

# Convert predictions to R vector to extract from H2O environment:
preds_ridge_train <- as.vector(as.data.frame(preds_ridge_train)$predict)
preds_ridge_test <- as.vector(as.data.frame(preds_ridge_test)$predict)

# Look at performance:
MAE(preds_ridge_train,train_processed[[target]])  
RMSE(preds_ridge_train,train_processed[[target]])  
R2_Score(preds_ridge_train,train_processed[[target]]) 

## Performance on test set:

MAE(preds_ridge_test,test_processed[[target]])
RMSE(preds_ridge_test,test_processed[[target]])
R2_Score(preds_ridge_test,test_processed[[target]])

## Lasso: alpha = 1 -----------------------------------------------------------


# Step 1: Grid search for lambda with alpha=1 (Lasso)
lasso_grid <- h2o.grid(
  algorithm = "glm",
  grid_id = "lasso_grid",
  x = features,
  y = target, 
  training_frame = train_h2o,
  family = "gaussian",
  alpha = 1,  # lasso
  nfolds = folds,
  keep_cross_validation_predictions = TRUE,
  standardize = FALSE,
  seed = seed,
  hyper_params = hyper_params,
  search_criteria = list(strategy = "Cartesian")
)

# Step 2: Get the best lambda based on chosen metric (e.g., RMSE)
grid_perf_lasso <- h2o.getGrid(grid_id = "lasso_grid", 
                               sort_by = metric, 
                               decreasing = FALSE)
print(grid_perf_lasso)
# Step 3: Extract the tuned hyperparameter(s)
best_model_id_lasso <- grid_perf_lasso@model_ids[[1]]
best_lasso_model <- h2o.getModel(best_model_id_lasso)
best_lambda_lasso <- best_lasso_model@parameters$lambda

# Step 4: Refit a single Lasso model with the best lambda 
lasso_final <- h2o.glm(
  x = features,
  y = target,
  training_frame = train_h2o,
  family = "gaussian",
  alpha = 1,
  lambda = best_lambda_lasso,
  standardize = FALSE,
  seed = seed
)

# Save predictions
preds_lasso_train <- h2o.predict(lasso_final, train_h2o)
preds_lasso_test <- h2o.predict(lasso_final, test_h2o)

# Convert predictions to R vector to extract from H2O environment:
preds_lasso_train <- as.vector(as.data.frame(preds_lasso_train)$predict)
preds_lasso_test <- as.vector(as.data.frame(preds_lasso_test)$predict)
# Look at performance:
MAE(preds_lasso_train,train_processed[[target]])  
RMSE(preds_lasso_train,train_processed[[target]])  
R2_Score(preds_lasso_train,train_processed[[target]])

## Performance on test set:

MAE(preds_lasso_test,test_processed[[target]])
RMSE(preds_lasso_test,test_processed[[target]])
R2_Score(preds_lasso_test,test_processed[[target]])

## Elastic net (combination of ridge and lasso) --------------------------------

# define a new grid search to include alpha

hyper_params_elastic <- list(
  lambda = 10^seq(-3, 5, length = 20),
  alpha = seq(0, 1, by = 0.1)
) 
# Step 1: Grid search for lambda and alpha
elastic_grid <- h2o.grid(
  algorithm = "glm",
  grid_id = "elastic_grid",
  x = features,
  y = target, 
  training_frame = train_h2o,
  family = "gaussian",
  nfolds = folds,
  keep_cross_validation_predictions = TRUE,
  standardize = FALSE,
  seed = seed,
  hyper_params = hyper_params_elastic,
  search_criteria = list(strategy = "Cartesian")
)

# Step 2: Get the best lambda and alpha based on chosen metric (e.g., RMSE)
grid_perf_elastic <- h2o.getGrid(grid_id = "elastic_grid", 
                                 sort_by = metric, 
                                 decreasing = FALSE)
print(grid_perf_elastic)

# Step 3: Extract the tuned hyperparameter(s)
best_model_id_elastic <- grid_perf_elastic@model_ids[[1]]
best_elastic_model <- h2o.getModel(best_model_id_elastic)
best_lambda_elastic<- best_elastic_model@parameters$lambda
best_alpha_elastic <- best_elastic_model@parameters$alpha
# Step 4: Refit a single elastic model with the best lambda and alpha
elastic_final <- h2o.glm(
  x = features,
  y = target,
  training_frame = train_h2o,
  family = "gaussian",
  lambda = best_lambda_elastic,
  alpha = best_alpha_elastic,
  standardize = FALSE,
  seed = seed
)

# Save predictions
preds_elastic_train <- h2o.predict(elastic_final, train_h2o)
preds_elastic_test <- h2o.predict(elastic_final, test_h2o)

# Convert predictions to R vector to extract from H2O environment:
preds_elastic_train <- as.vector(as.data.frame(preds_elastic_train)$predict)
preds_elastic_test <- as.vector(as.data.frame(preds_elastic_test)$predict)
# Look at performance:
MAE(preds_elastic_train,train_processed[[target]])  
RMSE(preds_elastic_train,train_processed[[target]])  
R2_Score(preds_elastic_train,train_processed[[target]]) 

## Performance on test set:

MAE(preds_elastic_test,test_processed[[target]])
RMSE(preds_elastic_test,test_processed[[target]])
R2_Score(preds_elastic_test,test_processed[[target]])
# ----------------------------------------------------------------------------- #
# 10. Create a data frame of the results --------------------------------------
# ----------------------------------------------------------------------------- #

models <- c("ols", "elastic", "ridge", "lasso")
metrics <- c("MAE", "RMSE", "R2")

results <- data.frame()
for (model in models) {
  # Construct prediction object names dynamically
  preds_train <- get(paste0("preds_", model, "_train"))
  preds_test  <- get(paste0("preds_", model, "_test"))
  
  # Calculate metrics for train
  mae_train  <- round(MAE(preds_train, train_processed[[target]]), 4)
  rmse_train <- round(RMSE(preds_train, train_processed[[target]]), 4)
  r2_train   <- round(R2_Score(preds_train, train_processed[[target]]), 4)
  
  # Calculate metrics for test
  mae_test  <- round(MAE(preds_test, test_processed[[target]]), 4)
  rmse_test <- round(RMSE(preds_test, test_processed[[target]]), 4)
  r2_test   <- round(R2_Score(preds_test, test_processed[[target]]), 4)
  
  # Bind results into a dataframe
  temp <- data.frame(
    Model = model,
    Dataset = c("Train", "Test"),
    MAE = c(mae_train, mae_test),
    RMSE = c(rmse_train, rmse_test),
    R2 = c(r2_train, r2_test)
  )
  
  results <- rbind(results, temp)
}

print(results)
# graph the results

# change results to long form first
results_long <- results %>%
  pivot_longer(cols = c(MAE, RMSE, R2), names_to = "Metric", values_to = "Value")

results_long$Dataset <- factor(results_long$Dataset, levels = c("Train", "Test"))

# Plot
ggplot(results_long, aes(x = Model, y = Value, fill = Dataset)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  facet_wrap(~ Metric, scales = "free_y") +
  labs(title = "Model Performance Metrics by Dataset",
       y = "Metric Value",
       x = "Model") +
  theme_minimal() +
  scale_fill_manual(values = c("Train" = "#66CDAA", "Test" = "#FF6A6A"))



# ----------------------------------------------------------------------------- #
# 11. Shutdown H2O ------------------------------------------------------------
# ----------------------------------------------------------------------------- #

# If a mistake was made along the way or H2O model needed to be run again, shut H2o down and start again.

h2o.shutdown(prompt = FALSE)





