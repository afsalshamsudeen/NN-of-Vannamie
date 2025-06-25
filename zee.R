# Disable OneDNN optimizations
Sys.setenv(TF_ENABLE_ONEDNN_OPTS = 0)

# ------------------- Load Required Libraries -------------------
library(keras3)
library(neuralnet)
library(grnn)
library(caret)
library(ggplot2)
library(reshape2)
library(reticulate)

# Configure reticulate to use the correct Python environment
use_python("C:/Users/itsme/AppData/Local/Programs/Python/Python312/python.exe", required = TRUE)
py_config()
if (!py_module_available("tensorflow")) {
  py_install("tensorflow")
}

# ------------------- Load Data -------------------
data <- read.csv("DATA SET.csv", stringsAsFactors = FALSE)
data$Cycle_ID <- as.factor(data$Cycle_ID)
data$Variant <- as.factor(data$Variant)
data$Pond_Type <- as.factor(data$Pond_Type)

# ------------------- Feature Selection -------------------
features <- c("DOC", "PL__SIZE", "Variant", "Pond_Type", "POND_SIZE_Ha", "Stocking_Density",
              "Temp", "Temp_PM", "pH", "PH_PM", "DO", "Alkalinity", "Ammonia", "Nitrite",
              "Nitrate", "Salinity", "FTF1", "FTF2", "FTF3", "FTF4", "Daily__feed", "Cum_Feed")
target <- "Avg_Weight"

model_data <- data[, c(features, target)]
model_data <- na.omit(model_data)

# ------------------- One-hot Encoding -------------------
# Create dummy variables based on the entire dataset
dummies <- dummyVars(Avg_Weight ~ ., data = model_data)
data_encoded <- predict(dummies, newdata = model_data)
model_data <- data.frame(data_encoded, Avg_Weight = model_data$Avg_Weight)

# ------------------- Train-Test Split -------------------
set.seed(123)
trainIndex <- createDataPartition(model_data$Avg_Weight, p = 0.75, list = FALSE)
train_data <- model_data[trainIndex, ]
test_data <- model_data[-trainIndex, ]

# Ensure test_data has the same columns as train_data
# Add missing columns with zeros if necessary
missing_cols <- setdiff(names(train_data), names(test_data))
if (length(missing_cols) > 0) {
  for (col in missing_cols) {
    test_data[[col]] <- 0
  }
  # Reorder columns to match train_data
  test_data <- test_data[, names(train_data)]
}

# ------------------- Normalize Data -------------------
normalize <- function(x) {
  rng <- max(x) - min(x)
  if (rng == 0) return(rep(0, length(x)))  # avoid divide by 0
  (x - min(x)) / rng
}

# Store min and max for denormalization
feature_mins <- apply(train_data, 2, min)
feature_maxs <- apply(train_data, 2, max)
feature_ranges <- feature_maxs - feature_mins
min_target <- feature_mins["Avg_Weight"]
range_target <- feature_ranges["Avg_Weight"]

# Check for zero-variance features (you can optionally drop them)
zero_variance <- which(feature_ranges == 0)
if (length(zero_variance) > 0) {
  warning(paste("The following features have zero variance:", paste(names(zero_variance), collapse = ", ")))
}

normalize_fixed <- function(x, min_val, range_val) {
  if (range_val == 0) return(rep(0, length(x)))  # Prevent division by zero
  (x - min_val) / range_val
}

# Normalize train
train_norm <- as.data.frame(mapply(normalize_fixed, train_data, feature_mins, feature_ranges, SIMPLIFY = FALSE))

# Normalize test using training min and max
test_norm <- as.data.frame(mapply(normalize_fixed, test_data, feature_mins, feature_ranges, SIMPLIFY = FALSE))

X_train <- as.matrix(train_norm[, -ncol(train_norm)])
y_train <- as.matrix(train_norm[, ncol(train_norm)])

X_test <- as.matrix(test_norm[, -ncol(test_norm)])
y_test_actual <- test_data$Avg_Weight

# Denormalize function
denormalize <- function(x, min_val, range_val) {
  x * range_val + min_val
}

# ------------------- Metrics Function (Updated) -------------------
calc_metrics <- function(actual, predicted, min_val, range_val) {
  # Denormalize predicted values only
  predicted <- denormalize(predicted, min_val, range_val)
  
  rmse <- sqrt(mean((actual - predicted)^2))
  mae <- mean(abs(actual - predicted))
  
  # Handle division by zero in MAPE
  nonzero_actual <- actual != 0
  if (sum(nonzero_actual) == 0) {
    mape <- NA  # If all actual values are 0, MAPE is undefined
  } else {
    mape <- mean(abs((actual[nonzero_actual] - predicted[nonzero_actual]) / actual[nonzero_actual])) * 100
  }
  
  r2 <- cor(actual, predicted)^2
  return(c(RMSE = rmse, MAE = mae, MAPE = mape, R2 = r2))
}


results <- list()
predictions <- list()

# ------------------- LSTM Model -------------------
X_train_lstm <- array(X_train, dim = c(nrow(X_train), 1, ncol(X_train)))
X_test_lstm <- array(X_test, dim = c(nrow(X_test), 1, ncol(X_test)))


# Create and build the model using the pipe operator
lstm_model <- keras_model_sequential() %>%
  layer_lstm(units = 50, input_shape = c(1, ncol(X_train)), return_sequences = FALSE) %>%
  layer_dense(units = 1)

# Compile the model
lstm_model %>% compile(loss = "mean_squared_error", optimizer = optimizer_adam())

# Fit the model
lstm_model %>% fit(
  x = X_train_lstm,
  y = y_train,
  epochs = 100,
  batch_size = 4,
  verbose = 1
)

# Predict
lstm_pred <- predict(lstm_model, X_test_lstm)
results$LSTM <- calc_metrics(y_test_norm, lstm_pred, min_target, range_target)
# Denormalize for plotting
lstm_pred_denorm <- denormalize(lstm_pred, min_target, range_target)
predictions$LSTM <- data.frame(Actual = y_test_actual, Predicted = lstm_pred_denorm)
print(class(lstm_model))



# ------------------- BPNN Model -------------------
formula <- as.formula(paste("Avg_Weight ~", paste(names(train_norm)[-ncol(train_norm)], collapse = " + ")))
bpnn_model <- neuralnet(formula, data = train_norm, hidden = c(10, 5), linear.output = TRUE)
bp_pred <- compute(bpnn_model, test_norm[, -ncol(test_norm)])$net.result
results$BPNN <- calc_metrics(y_test_norm, bp_pred, min_target, range_target)
bp_pred_denorm <- denormalize(bp_pred, min_target, range_target)
predictions$BPNN <- data.frame(Actual = y_test_actual, Predicted = bp_pred_denorm)

save(bpnn_model, file = "bpnn_model_fixed.RData")



# ------------------- GRNN Model -------------------
grnn_model <- learn(train_norm, variable.column = ncol(train_norm))
grnn_model <- smooth(grnn_model, sigma = 0.1)
grnn_pred <- sapply(1:nrow(test_norm), function(i) {
  guess(grnn_model, as.matrix(test_norm[i, -ncol(test_norm)]))
})
results$GRNN <- calc_metrics(y_test_norm, grnn_pred, min_target, range_target)
grnn_pred_denorm <- denormalize(grnn_pred, min_target, range_target)
predictions$GRNN <- data.frame(Actual = y_test_actual, Predicted = grnn_pred_denorm)
save(grnn_model, file = "grnn_model.RData")

# ------------------- Model Performance Table -------------------
performance_table <- do.call(rbind, lapply(names(results), function(model) {
  data.frame(Model = model, t(results[[model]]))
}))
print(performance_table)
write.csv(performance_table, "neural_model_performance.csv", row.names = FALSE)

# ------------------- Plot: Fitted vs Predicted -------------------
library(gridExtra)
plot_list <- lapply(names(predictions), function(model) {
  ggplot(predictions[[model]], aes(x = Actual, y = Predicted)) +
    geom_point(alpha = 0.6, color = "darkblue") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    ggtitle(paste("Fitted vs Predicted -", model)) +
    xlab("Actual") + ylab("Predicted") + theme_minimal()
})
pdf("neural_fitted_vs_predicted.pdf")
do.call(grid.arrange, c(plot_list, ncol = 2))
dev.off()
# ------------------- Residual Plots -------------------
residual_plots <- lapply(names(predictions), function(model) {
  df <- predictions[[model]]
  df$Residuals <- df$Actual - df$Predicted
  ggplot(df, aes(x = Predicted, y = Residuals)) +
    geom_point(alpha = 0.6, color = "darkgreen") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    ggtitle(paste("Residual Plot -", model)) +
    xlab("Predicted") + ylab("Residuals") + theme_minimal()
})

pdf("residual_plots.pdf")
do.call(grid.arrange, c(residual_plots, ncol = 2))
dev.off()

# ------------------- Sensitivity Analysis -------------------
sensitivity_plots <- list()

for (model in names(predictions)) {
  base <- colMeans(X_test)
  baseline_pred <- if (model == "LSTM") {
    predict(lstm_model, array(base, dim = c(1, 1, ncol(X_test))))
  } else if (model == "BPNN") {
    compute(bp_model, matrix(base, nrow = 1))$net.result
  } else {
    guess(grnn_model, matrix(base, nrow = 1))
  }
  baseline_pred <- as.numeric(denormalize(baseline_pred, min_target, range_target))
  
  # Sensitivity: change in output due to a small change in each feature
  sens <- sapply(1:ncol(X_test), function(i) {
    temp <- base
    temp[i] <- temp[i] + 0.01  # Small perturbation
    sample_input <- matrix(temp, nrow = 1)
    
    if (model == "LSTM") {
      sample_input_lstm <- array(sample_input, dim = c(1, 1, ncol(sample_input)))
      pred <- predict(lstm_model, sample_input_lstm)
    } else if (model == "BPNN") {
      pred <- compute(bp_model, sample_input)$net.result
    } else {
      pred <- guess(grnn_model, sample_input)
    }
    
    pred <- as.numeric(denormalize(pred, min_target, range_target))
    return(abs(pred - baseline_pred))  # Difference from baseline
  })
  
  sens_df <- data.frame(Feature = colnames(X_test), Importance = sens)
  sens_df <- sens_df[order(-sens_df$Importance), ][1:20, ]  # Show top 20 features only
  
  sensitivity_plots[[model]] <- ggplot(sens_df, aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    ggtitle(paste("Sensitivity Analysis -", model)) +
    xlab("Feature") + ylab("Output Change") +
    theme_minimal() +
    theme(axis.text.y = element_text(size = 6))  # Reduce Y-axis text size
}
pdf("sensitivity_analysis.pdf")
do.call(grid.arrange, c(sensitivity_plots, ncol = 1))
dev.off()

# ------------------- Combined Model Performance -------------------
library(reshape2)
performance_long <- melt(performance_table, id.vars = "Model")

ggplot(performance_long, aes(x = Model, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  ggtitle("Model Performance Comparison") +
  ylab("Metric Value") + xlab("Model") +
  theme_minimal() + scale_fill_brewer(palette = "Set2")

ggsave("combined_model_performance.pdf", width = 8, height = 5)

# ------------------- Combined Feature Importance (Top 10 per model) -------------------
combined_importance <- do.call(rbind, lapply(names(sensitivity_plots), function(model) {
  sens <- sensitivity_plots[[model]]$data
  sens$Model <- model
  return(sens)
}))

# Only keep top 10 per model
top_features_combined <- do.call(rbind, lapply(split(combined_importance, combined_importance$Model), function(df) {
  df[order(-df$Importance), ][1:10, ]
}))

# Plot
ggplot(top_features_combined, aes(x = reorder(Feature, Importance), y = Importance, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +
  facet_wrap(~ Model, scales = "free_y") +
  ggtitle("Top 10 Feature Importance Across Models") +
  xlab("Feature") + ylab("Output Sensitivity") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 7))

ggsave("all_model_feature_importance.pdf", width = 10, height = 6)



