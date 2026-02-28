############################################################
# Part 1: Simple Linear Regression
# Boston Dataset (MASS Library)
# Predicting medv from rm (avg rooms per dwelling)
############################################################

library(MASS)
data(Boston)

cat("========== PART 1: Simple Linear Regression ==========\n")
cat("Dataset: Boston from MASS library\n")
cat("Response: medv (median home value in $1000s)\n")
cat("Predictor: rm (average number of rooms per dwelling)\n\n")

# --- Dataset overview ---
cat("--- Dataset overview ---\n")
cat("Observations:", nrow(Boston), "\n")
cat("Variables:", ncol(Boston), "\n\n")

cat("--- Variable descriptions ---\n")
cat("crim    - per capita crime rate by town\n")
cat("zn      - proportion of residential land zoned for lots > 25,000 sq.ft.\n")
cat("indus   - proportion of non-retail business acres per town\n")
cat("chas    - Charles River dummy variable (1 = tract bounds river)\n")
cat("nox     - nitrogen oxides concentration (parts per 10 million)\n")
cat("rm      - average number of rooms per dwelling\n")
cat("age     - proportion of owner-occupied units built prior to 1940\n")
cat("dis     - weighted distances to five Boston employment centres\n")
cat("rad     - index of accessibility to radial highways\n")
cat("tax     - full-value property-tax rate per $10,000\n")
cat("ptratio - pupil-teacher ratio by town\n")
cat("black   - 1000(Bk - 0.63)^2 where Bk is proportion of Black residents\n")
cat("lstat   - percent lower status of the population\n")
cat("medv    - median value of owner-occupied homes in $1000s\n\n")

# --- Summary statistics ---
cat("--- Summary of medv and rm ---\n")
print(summary(Boston[, c("medv", "rm")]))

# --- Fit simple linear regression ---
slr.fit <- lm(medv ~ rm, data = Boston)

cat("\n--- Simple Linear Regression: medv ~ rm ---\n")
print(summary(slr.fit))

# --- Confidence intervals ---
cat("\n--- 95% Confidence Intervals ---\n")
print(confint(slr.fit))

# --- Predictions ---
cat("\n--- Predictions at rm = 5, 6, 7, 8 ---\n")
cat("Confidence intervals:\n")
print(predict(slr.fit, data.frame(rm = c(5, 6, 7, 8)), interval = "confidence"))
cat("\nPrediction intervals:\n")
print(predict(slr.fit, data.frame(rm = c(5, 6, 7, 8)), interval = "prediction"))

# --- PLOT: Regression plot ---
plot(Boston$rm, Boston$medv,
     xlab = "rm (Average Number of Rooms per Dwelling)",
     ylab = "medv (Median Home Value in $1000s)",
     main = "Simple Linear Regression: medv ~ rm (Boston)",
     col = "steelblue", pch = 19, cex = 0.7,
     xlim = c(3, 9), ylim = c(0, 55))
abline(slr.fit, col = "red", lwd = 2.5)
new_rm <- data.frame(rm = seq(3, 9, length.out = 300))
ci <- predict(slr.fit, new_rm, interval = "confidence")
lines(new_rm$rm, ci[, "lwr"], col = "red", lty = 2)
lines(new_rm$rm, ci[, "upr"], col = "red", lty = 2)
legend("topleft", legend = c("Data", "Regression Line", "95% CI"),
       col = c("steelblue", "red", "red"),
       pch = c(19, NA, NA), lty = c(NA, 1, 2),
       lwd = c(NA, 2.5, 1), bg = "white")

# --- PLOT: 4-panel residual diagnostics ---
par(mfrow = c(2, 2))
plot(slr.fit)
par(mfrow = c(1, 1))

# --- Leverage analysis ---
cat("\n--- Leverage analysis ---\n")
cat("Max leverage:", max(hatvalues(slr.fit)), "\n")
cat("Index of max leverage:", which.max(hatvalues(slr.fit)), "\n")
cat("rm value at max leverage:", Boston$rm[which.max(hatvalues(slr.fit))], "\n")
