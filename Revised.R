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
dummies <- dummyVars(Avg_Weight ~ ., data = model_data)
data_encoded <- predict(dummies, newdata = model_data)
model_data <- data.frame(data_encoded, Avg_Weight = model_data$Avg_Weight)

# ------------------- Train-Test Split -------------------
set.seed(123)
trainIndex <- createDataPartition(model_data$Avg_Weight, p = 0.75, list = FALSE)
train_data <- model_data[trainIndex, ]
test_data <- model_data[-trainIndex, ]

# Ensure test_data has the same columns as train_data
missing_cols <- setdiff(names(train_data), names(test_data))
if (length(missing_cols) > 0) {
  for (col in missing_cols) {
    test_data[[col]] <- 0
  }
  test_data <- test_data[, names(train_data)]
}

# ------------------- Normalize Data -------------------
normalize <- function(x) {
  rng <- max(x) - min(x)
  if (rng == 0) return(rep(0, length(x)))
  (x - min(x)) / rng
}

feature_mins <- apply(train_data, 2, min)
feature_maxs <- apply(train_data, 2, max)
feature_ranges <- feature_maxs - feature_mins
min_target <- feature_mins["Avg_Weight"]
range_target <- feature_ranges["Avg_Weight"]

zero_variance <- which(feature_ranges == 0)
if (length(zero_variance) > 0) {
  warning(paste("The following features have zero variance:", paste(names(zero_variance), collapse = ", ")))
}

normalize_fixed <- function(x, min_val, range_val) {
  if (range_val == 0) return(rep(0, length(x)))
  (x - min_val) / range_val
}

train_norm <- as.data.frame(mapply(normalize_fixed, train_data, feature_mins, feature_ranges, SIMPLIFY = FALSE))
test_norm <- as.data.frame(mapply(normalize_fixed, test_data, feature_mins, feature_ranges, SIMPLIFY = FALSE))

X_train <- as.matrix(train_norm[, -ncol(train_norm)])
y_train <- as.matrix(train_norm[, ncol(train_norm)])
X_test <- as.matrix(test_norm[, -ncol(test_norm)])
y_test_actual <- test_data$Avg_Weight

denormalize <- function(x, min_val, range_val) {
  x * range_val + min_val
}

# ------------------- Metrics Function (Fixed) -------------------
calc_metrics <- function(actual, predicted, min_val, range_val) {
  # Denormalize values
  actual <- denormalize(actual, min_val, range_val)
  predicted <- denormalize(predicted, min_val, range_val)
  
  # Debugging prints
  print(paste("Length of actual:", length(actual)))
  print(paste("Length of predicted:", length(predicted)))
  print(head(actual))
  print(head(predicted))
  
  # Ensure compatible dimensions
  if (length(actual) != length(predicted)) {
    warning("Length mismatch between actual and predicted values")
    if (length(predicted) > length(actual)) {
      predicted <- predicted[1:length(actual)]
    } else {
      actual <- actual[1:length(predicted)]
    }
  }
  
  if (all(is.na(predicted)) || all(is.na(actual))) {
    warning("NA values detected in actual or predicted")
    return(c(RMSE = NA, MAE = NA, MAPE = NA, R2 = NA))
  }
  
  rmse <- sqrt(mean((actual - predicted)^2, na.rm = TRUE))
  mae <- mean(abs(actual - predicted), na.rm = TRUE)
  
  # Handle MAPE with nonzero_actual defined
  nonzero_actual <- actual != 0 & !is.na(actual)
  mape <- if (sum(nonzero_actual) == 0) 0 else mean(abs((actual[nonzero_actual] - predicted[nonzero_actual]) / actual[nonzero_actual]) * 100, na.rm = TRUE)
  
  r2 <- if (var(actual, na.rm = TRUE) == 0) 0 else cor(actual, predicted, use = "complete.obs")^2
  
  return(c(RMSE = rmse, MAE = mae, MAPE = mape, R2 = r2))
}

results <- list()
predictions <- list()

# ------------------- LSTM Model -------------------
X_train_lstm <- array(X_train, dim = c(nrow(X_train), 1, ncol(X_train)))
X_test_lstm <- array(X_test, dim = c(nrow(X_test), 1, ncol(X_test)))

lstm_model <- keras_model_sequential() %>%
  layer_lstm(units = 50, input_shape = c(1, ncol(X_train)), return_sequences = FALSE) %>%
  layer_dense(units = 1)

lstm_model %>% compile(loss = "mean_squared_error", optimizer = optimizer_adam())
lstm_model %>% fit(x = X_train_lstm, y = y_train, epochs = 100, batch_size = 4, verbose = 1)

lstm_pred <- predict(lstm_model, X_test_lstm)
results$LSTM <- calc_metrics(y_test_actual, lstm_pred, min_target, range_target)
lstm_pred_denorm <- denormalize(lstm_pred, min_target, range_target)
predictions$LSTM <- data.frame(Actual = y_test_actual, Predicted = lstm_pred_denorm)
print("LSTM Results:")
print(results$LSTM)

# ------------------- BPNN Model -------------------
formula <- as.formula(paste("Avg_Weight ~", paste(names(train_norm)[-ncol(train_norm)], collapse = " + ")))
bpnn_model <- neuralnet(formula, data = train_norm, hidden = c(10, 5), linear.output = TRUE)
bp_pred <- compute(bpnn_model, test_norm[, -ncol(test_norm)])$net.result
print("BPNN Prediction Check:")
print(head(bp_pred))
results$BPNN <- calc_metrics(y_test_actual, bp_pred, min_target, range_target)
bp_pred_denorm <- denormalize(bp_pred, min_target, range_target)
predictions$BPNN <- data.frame(Actual = y_test_actual, Predicted = bp_pred_denorm)
print("BPNN Results:")
print(results$BPNN)
save(bpnn_model, file = "bpnn_model_fixed.RData")

# ------------------- GRNN Model -------------------
grnn_model <- learn(train_norm, variable.column = ncol(train_norm))
grnn_model <- smooth(grnn_model, sigma = 0.1)
grnn_pred <- sapply(1:nrow(test_norm), function(i) guess(grnn_model, as.matrix(test_norm[i, -ncol(test_norm)])))
print("GRNN Prediction Check:")
print(head(grnn_pred))
results$GRNN <- calc_metrics(y_test_actual, grnn_pred, min_target, range_target)
grnn_pred_denorm <- denormalize(grnn_pred, min_target, range_target)
predictions$GRNN <- data.frame(Actual = y_test_actual, Predicted = grnn_pred_denorm)
print("GRNN Results:")
print(results$GRNN)
save(grnn_model, file = "grnn_model.RData")

# ------------------- Model Performance Table -------------------
performance_table <- do.call(rbind, lapply(names(results), function(model) {
  res <- results[[model]]
  metrics <- c("RMSE", "MAE", "MAPE", "R2")
  res_full <- setNames(rep(NA, length(metrics)), metrics)
  if (!is.null(res) && length(res) > 0) {
    names(res) <- metrics[1:length(res)]
    res_full[names(res)] <- res
  }
  data.frame(Model = model, t(res_full))
}))

print("Performance Table:")
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
    compute(bpnn_model, matrix(base, nrow = 1))$net.result
  } else {
    guess(grnn_model, matrix(base, nrow = 1))
  }
  baseline_pred <- as.numeric(denormalize(baseline_pred, min_target, range_target))
  
  sens <- sapply(1:ncol(X_test), function(i) {
    temp <- base
    temp[i] <- temp[i] + 0.01
    sample_input <- matrix(temp, nrow = 1)
    pred <- if (model == "LSTM") {
      predict(lstm_model, array(sample_input, dim = c(1, 1, ncol(sample_input))))
    } else if (model == "BPNN") {
      compute(bpnn_model, sample_input)$net.result
    } else {
      guess(grnn_model, sample_input)
    }
    pred <- as.numeric(denormalize(pred, min_target, range_target))
    abs(pred - baseline_pred)
  })
  
  sens_df <- data.frame(Feature = colnames(X_test), Importance = sens)
  sens_df <- sens_df[order(-sens_df$Importance), ][1:20, ]
  
  sensitivity_plots[[model]] <- ggplot(sens_df, aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    ggtitle(paste("Sensitivity Analysis -", model)) +
    xlab("Feature") + ylab("Output Change") +
    theme_minimal() +
    theme(axis.text.y = element_text(size = 6))
}
pdf("sensitivity_analysis.pdf")
do.call(grid.arrange, c(sensitivity_plots, ncol = 1))
dev.off()

# ------------------- Combined Model Performance -------------------
# ------------------- Combined Model Performance -------------------
library(reshape2)
performance_long <- melt(performance_table, id.vars = "Model")
all_models <- unique(performance_long$Model)
all_metrics <- c("RMSE", "MAE", "MAPE", "R2")
complete_grid <- expand.grid(Model = all_models, variable = all_metrics)
performance_long <- merge(complete_grid, performance_long, by = c("Model", "variable"), all.x = TRUE)
performance_long$value[is.na(performance_long$value)] <- 1e-10

# Debug: Print raw performance data
print("Raw Performance Data:")
print(performance_long)

# Normalize each metric with an offset to preserve visibility
performance_long <- do.call(rbind, lapply(split(performance_long, performance_long$variable), function(df) {
  max_val <- max(df$value, na.rm = TRUE)
  min_val <- min(df$value, na.rm = TRUE)
  range_val <- max(max_val - min_val, 1e-5)  # Minimum range to avoid zero
  normalized <- (df$value - min_val) / range_val  # Normalize to [0, 1]
  df$Normalized_Value <- normalized + 0.1  # Add offset to ensure minimum is 0.1
  return(df)
}))

# Debug: Print normalized data and check range
print("Normalized Performance Data:")
print(performance_long)
print("Range of Normalized Values:")
print(range(performance_long$Normalized_Value, na.rm = TRUE))

# Plot normalized values without strict y-axis limit initially
p <- ggplot(performance_long, aes(x = Model, y = Normalized_Value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  ggtitle("Model Performance Comparison (Normalized)") +
  ylab("Normalized Metric Value") + xlab("Model") +
  theme_minimal() + scale_fill_brewer(palette = "Set2") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Print plot data for debugging
print("Plot Data Summary:")
print(summary(performance_long$Normalized_Value))

# Display plot
print(p)
ggsave("combined_model_performance.pdf", width = 10, height = 8)

# ------------------- Combined Feature Importance -------------------
combined_importance <- do.call(rbind, lapply(names(sensitivity_plots), function(model) {
  sens <- sensitivity_plots[[model]]$data
  sens$Model <- model
  sens
}))
top_features_combined <- do.call(rbind, lapply(split(combined_importance, combined_importance$Model), function(df) {
  df[order(-df$Importance), ][1:10, ]
}))

ggplot(top_features_combined, aes(x = reorder(Feature, Importance), y = Importance, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +
  facet_wrap(~ Model, scales = "free_y") +
  ggtitle("Top 10 Feature Importance Across Models") +
  xlab("Feature") + ylab("Output Sensitivity") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 7))

ggsave("all_model_feature_importance.pdf", width = 8, height = 6)