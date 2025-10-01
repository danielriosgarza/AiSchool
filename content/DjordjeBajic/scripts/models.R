###

# load data and rename columns
df <- read.csv('content/DjordjeBajic/data/full-factorial-construction_Diaz-Colunga2024.csv')
colnames(df) <- c(paste0('x', 1:8), 'y')

# what's the true functional maximum?
F_max <- max(df$y)

# split into training set and test set
N_obs <- 30 
set.seed(0)
df <- df[sample(nrow(df)), ]
training_set <- df[1:N_obs, ]
test_set <- df[(N_obs + 1):nrow(df), ]



### DIFFERENT MODELS ARE PROPOSED BELOW.
### CHOOSE YOUR FAVORITE TO GENERATE PREDICTIONS (y_pred).
### YOU CAN DELETE THE REST SO THE SCRIPT RUNS SMOOTHLY.
### ----------------------------------------------------------------------------

### 1. LINEAR MODEL
fit <- lm(y ~ ., data = training_set)
y_pred <- predict(fit, newdata = test_set[, 1:8])



### 2. MODEL WITH LOW-ORDER (PAIRWISE) INTERACTIONS
fit <- lm(y ~ (.)^2, data = training_set)
y_pred <- predict(fit, newdata = test_set[, 1:8])



### 2b. REGULARIZED PAIRWISE MODEL
# tip: ChatGPT wrote this
library(glmnet)

form <- as.formula("~ (.)^2")
X_model <- model.matrix(form, data = as.data.frame(training_set[, 1:8]))[, -1]  # remove intercept
cv_fit <- cv.glmnet(X_model, training_set$y, alpha = 0)  # alpha=1 -> lasso, alpha=0 -> ridge

fit_lasso <- glmnet(X_model, training_set$y, alpha = 1, lambda = cv_fit$lambda.min)

X_outofsample <- model.matrix(form, data = test_set[, 1:8])[, -1]
y_pred <- predict(fit_lasso, newx = X_outofsample, s = cv_fit$lambda.min)[, 1]



### 3. FACTORIZATION MACHINE (FM)
# thank you again, ChatGPT
library(rsparse)
library(Matrix)

X_obs_sp <- Matrix(as.matrix(training_set[, 1:8]), sparse = TRUE)
fm_model <- rsparse::FactorizationMachine$new(
  learning_rate_w = 0.2,
  rank = 2,         # small latent dimension, because data is small
  lambda_w = 0.01,  # regularization on w (linear terms)
  lambda_v = 0.01,  # regularization on v (latent interaction terms)
  family = "gaussian",  # since predicting scalar F
  intercept = TRUE,
  learning_rate_v = 0.2
)

fm_model$fit(X_obs_sp, training_set$y, n_iter = 100)

X_outofsample_sp <- as(as.matrix(test_set[, 1:8]), "RsparseMatrix")
y_pred <- fm_model$predict(X_outofsample_sp)



### 4. RANDOM FOREST
library(ranger)

rf_fit <- ranger(
  y ~ ., 
  data = training_set,
  num.trees = 500,          # number of trees
  mtry = 3,                 # features per split (tunable)
  min.node.size = 1,        # allow deep trees, since N is small
  importance = "impurity"   # compute variable importance
)

y_pred <- predict(rf_fit, data = test_set[, 1:8])$predictions



### 5. GRADIENT BOOSTING
library(xgboost)

dtrain <- xgb.DMatrix(data = as.matrix(training_set[, 1:8]), label = training_set$y)
xgb_fit <- xgboost(
  data = dtrain,
  nrounds = 50,           # number of boosting rounds
  max_depth = 3,          # tree depth
  eta = 0.1,              # learning rate
  objective = "reg:squarederror",
  verbose = 0
)

y_pred <- predict(xgb_fit, newdata = as.matrix(test_set[, 1:8]))


### 6. GAUSSIAN PROCESS
library(kernlab)

gp_fit <- gausspr(
  x = as.matrix(training_set[, 1:8]), 
  y = training_set$y, 
  kernel = "rbfdot",
  kpar = list(sigma = 0.5)  # tune this parameter
)

y_pred <- predict(gp_fit, as.matrix(test_set[, 1:8]))


### 7. NEURAL MODELS
library(nnet)

nn_fit <- nnet(
  y ~ ., 
  data = training_set,
  size = 4,       # hidden units
  linout = TRUE,  # regression instead of classification
  decay = 0.01,   # L2 penalty to reduce overfitting
  maxit = 500,    # iterations
  trace = FALSE
)

y_pred <- predict(nn_fit, newdata = as.matrix(test_set[, 1:8]))[, 1]


### 8. SUBMODULAR FUNCTION
library(nnls)

fit_nn <- nnls(as.matrix(training_set[, 1:8]), training_set$y)

X_outofsample <- as.matrix(test_set[, 1:8])
y_pred <- (X_outofsample %*% coef(fit_nn))[, 1]


### ----------------------------------------------------------------------------
### CHECK PERFORMANCE

# (optional) check: are predictions close to true values in the test set?
library(ggplot2)
ggplot(data.frame(true = test_set$y, pred = y_pred),
       aes(x = pred, y = true)) +
  geom_point() +
  geom_blank(aes(x = true, y = pred)) + # little trick for equal x and y scales

  geom_abline(slope = 1,
              intercept = 0,
              color = 'gray') +
  xlab('Predicted F') +
  ylab('True F') +
  theme(aspect.ratio = 1)

# combine predictions and observations
df_comb <- rbind(training_set,
                 cbind(test_set[, 1:8], y = y_pred))

# look for max. function and best consortium
# (may be in the training set, or a prediction)
which_max <- which.max(df_comb$y)
best_consortium <- df_comb[which_max, 1:8]

# check the TRUE function* of the consortium predicted to be best
# (* in the original, unmodified data set)
F_true_xopt <- df$y[which_max]

# how well did we do?
Q <- F_true_xopt / F_max
print(Q)
