library(rmarkdown)
library(ggforce)
library(tidyr)
library(combinat)
library(dplyr)
library(ggplot2)
library(glmnet)
library(rsparse)
library(Matrix)
library(ranger)
library(xgboost)
library(kernlab)
library(nnet)
library(nnls)

plotLandscape <- function(df, color = 'gray') {
  
  # Number of elements (we assume the last column is always the function, the rest represent composition)
  N <- ncol(df) - 1
  
  # Rename columns for tractability
  colnames(df) <- c(paste('x', 1:N, sep = ''), 'y')
  
  # Step 1: Compute x-coordinate (sum of binary vars)
  df$xsum <- rowSums(df[, 1:N])
  
  # Step 2: Create row identifiers
  df$id <- 1:nrow(df)
  
  # Step 3: Function to compute Hamming distance
  hamming <- function(a, b) sum(a != b)
  
  # Step 4: Build edge list (pairs of rows differing by exactly one flip)
  edges <- do.call(rbind,
                   combn(df$id, 2, function(pair) {
                     a <- df[pair[1], 1:N]
                     b <- df[pair[2], 1:N]
                     if (hamming(as.numeric(a), as.numeric(b)) == 1) {
                       data.frame(from = pair[1], to = pair[2])
                     }
                   }, simplify = FALSE)
  )
  
  # Step 5: Prepare data for plotting edges
  edges_df <- edges %>%
    left_join(df %>% select(id, xsum, y), by = c("from" = "id")) %>%
    rename(x1 = xsum, y1 = y) %>%
    left_join(df %>% select(id, xsum, y), by = c("to" = "id")) %>%
    rename(x2 = xsum, y2 = y)
  
  # Step 6: Plot
  myplot <- ggplot() +
    geom_segment(data = edges_df, aes(x = x1, y = y1, xend = x2, yend = y2), color = color) +
    geom_abline(slope = 0,
                intercept = 0,
                color = '#d1d3d4') +
    scale_x_continuous(name = '# of elements',
                       breaks = 0:N,
                       labels = as.character(0:N)) +
    scale_y_continuous(name = 'Function',
                       expand = c(0.05, 0.05)) +
    theme_bw() +
    theme(aspect.ratio = 0.6,
          panel.grid = element_blank(),
          panel.border = element_blank(),
          legend.position = 'none',
          axis.title = element_text(size = 18),
          axis.text = element_text(size = 16)) +
    annotate("segment", x=-Inf, xend=Inf, y=-Inf, yend=-Inf, linewidth=0.5) +
    annotate("segment", x=-Inf, xend=-Inf, y=-Inf, yend=Inf, linewidth=0.5)
  
  return(myplot)
  
}

getInterCoefficients <- function(df, mode = 'Taylor') {
  
  # This function takes an input data frame (df) (important: df should NOT have
  # been generated through structureData(), it should be a "raw" data frame) and
  # returns the interaction coefficients at all orders, meaning, if the function
  # F is expressed as:
  #    F = f_0 + f_1 x_1 + f_2 x_2 + ... + f_12 x_1 x_2 + ... + f_123 x_1 x_2 x_3 + ...
  # with x_i = 0,1 (Taylor) or x_i = -1,+1 (Fourier), then the function returns
  # the (fitted) values of the coefficients f_0, f_i, f_ij, etc.
  # This, in principle, can only be used reliably with combinatorially complete
  # data frames (otherwise it will return a bunch of NAs for some coefficients,
  # and provide unreliable estimates of the non-NA ones).
  # Important: in the input df, compositional values should take values 0 and 1,
  # even if the Fourier coefficients are to be returned. The conversion of 0's
  # into -1's is done internally by the function if 'mode' is set to 'Fourier'
  # (this conversion is not done if mode = 'Taylor').
  
  # Number of elements (we assume the last column is always the function, the rest represent composition)
  N <- ncol(df) - 1
  
  # Rename columns for tractability
  colnames(df) <- c(paste('x', 1:N, sep = ''), 'y')
  
  # Taylor or Fourier coefficients
  if (mode == 'Fourier') for (i in 1:N) df[df[, i] == 0, i] <- -1
  
  # fit linear model with interaction terms up to the N-th (with N being the
  # number of compositional variables)
  my_fit <- lm(as.formula(paste('y ~ ', paste(rep('.', N), collapse = ' * '))),
               data = df)
  coefs <- my_fit$coefficients
  names(coefs)[1] <- ''
  coefs <- data.frame(order = c(0, 1 + nchar(names(coefs[2:length(coefs)])) - nchar(gsub(':', '', names(coefs[2:length(coefs)])))),
                      index = names(coefs),
                      value = as.numeric(coefs))
  
  rownames(coefs) <- NULL
  return(coefs)
  
}

makeLandscapeFromInter <- function(coefs, mode = 'Taylor') {
  
  # extract num,ber of elements
  N <- log2(nrow(coefs))
  
  # make grid
  grid <- expand.grid(replicate(N, 0:1, simplify = FALSE)) %>%
    as.data.frame() %>% rename_with(~ paste0("x", seq_along(.)), everything())
  if (mode == 'Fourier') for (i in 1:N) grid[grid[, i] == 0, i] <- -1

  # full-saturated formula producing all interactions up to N
  mm <- model.matrix(as.formula(paste0("~ (", paste(names(grid), collapse = " + "), ")^", N)),
                     data = grid)
  
  # ensure coefficient vector matches model matrix columns (fill missing with 0)
  coefs$index[coefs$index == ''] <- '(Intercept)'
  coef_for_mm <- numeric(ncol(mm))
  names(coef_for_mm) <- colnames(mm)
  intersect_names <- intersect(colnames(mm), coefs$index)
  coef_for_mm[intersect_names] <- setNames(coefs$value, coefs$index)[intersect_names]
  
  # predicted y
  y <- as.vector(mm %*% coef_for_mm)
  
  # grid back to {0, 1}
  if (mode == 'Fourier') for (i in 1:N) grid[grid[, i] == -1, i] <- 0
  
  return(cbind(grid, y = y))
  
}

getQ <- function(df, method = 'LM', training_fraction = 0.1) {
  
  # 'method' argument must be one of the following:
  # 
  # 'LM'                  -> linear model (no interactions)
  # 'pairwise'            -> model with pairwise interaction terms
  # 'reg_pairwise'        -> pairwise model with regularization
  # 'FM'                  -> factorization machine
  # 'RF'                  -> random forest
  # 'gradient_boosting'   -> gradient boosting, obviously
  # 'GP'                  -> gaussian process
  # 'NN'                  -> neural model
  
  # format column names for consistency
  colnames(df) <- c(paste0('x', 1:(ncol(df) - 1)), 'y')
  
  # get maximum function
  F_max <- max(df$y)
  
  # split into training set and test set
  N_obs <- ceiling(training_fraction * nrow(df))
  
  df <- df[sample(nrow(df)), ]
  training_set <- df[1:N_obs, ]
  test_set <- df[(N_obs + 1):nrow(df), ]
  
  if (method == 'control') {
    
    F_max_insample <- max(training_set$y)
    Q <- F_max_insample / F_max
    
  } else {
    
    # FIT MODEL(S)
    if (method == 'LM') {
      
      ### 1. LINEAR MODEL
      
      fit <- lm(y ~ ., data = training_set)
      y_pred <- predict(fit, newdata = test_set[, 1:8])
      
    } else if (method == 'pairwise') {
      
      ### 2. MODEL WITH LOW-ORDER (PAIRWISE) INTERACTIONS
      
      fit <- lm(y ~ (.)^2, data = training_set)
      y_pred <- predict(fit, newdata = test_set[, 1:8])
      
    } else if (method == 'reg_pairwise') {
      
      ### 2b. REGULARIZED PAIRWISE MODEL
      
      form <- as.formula("~ (.)^2")
      X_model <- model.matrix(form, data = as.data.frame(training_set[, 1:8]))[, -1]  # remove intercept
      cv_fit <- cv.glmnet(X_model, training_set$y, alpha = 0)  # alpha=1 -> lasso, alpha=0 -> ridge
      
      fit_lasso <- glmnet(X_model, training_set$y, alpha = 1, lambda = cv_fit$lambda.min)
      
      X_outofsample <- model.matrix(form, data = test_set[, 1:8])[, -1]
      y_pred <- predict(fit_lasso, newx = X_outofsample, s = cv_fit$lambda.min)[, 1]
      
    } else if (method == 'FM') {
      
      ### 3. FACTORIZATION MACHINE (FM)
      
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
      
    } else if (method == 'RF') {
      
      ### 4. RANDOM FOREST
      
      rf_fit <- ranger(
        y ~ ., 
        data = training_set,
        num.trees = 500,          # number of trees
        mtry = 3,                 # features per split (tunable)
        min.node.size = 1,        # allow deep trees, since N is small
        importance = "impurity"   # compute variable importance
      )
      
      y_pred <- predict(rf_fit, data = test_set[, 1:8])$predictions
      
    } else if (method == 'gradient_boosting') {
      
      ### 5. GRADIENT BOOSTING
      
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
      
    } else if (method == 'GP') {
      
      ### 6. GAUSSIAN PROCESS
      
      gp_fit <- gausspr(
        x = as.matrix(training_set[, 1:8]), 
        y = training_set$y, 
        kernel = "rbfdot",
        kpar = list(sigma = 0.5)  # tune this parameter
      )
      
      y_pred <- predict(gp_fit, as.matrix(test_set[, 1:8]))
      
    } else if (method == 'NN') {
      
      ### 7. NEURAL MODELS
      
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
      
    } else if (method == 'submodular') {
      
      ### 8. SUBMODULAR FUNCTION
      
      fit_nn <- nnls(as.matrix(training_set[, 1:8]), training_set$y)
      
      X_outofsample <- as.matrix(test_set[, 1:8])
      y_pred <- (X_outofsample %*% coef(fit_nn))[, 1]
      
    }
    
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
    
  }
  
  return(Q)
  
}
