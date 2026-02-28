############################################################
# Part 2: Multiple Linear Regression
# Boston Dataset (MASS Library)
# Predicting medv from multiple features
############################################################

library(MASS)
data(Boston)

cat("========== PART 2: Multiple Linear Regression ==========\n")
cat("Dataset: Boston from MASS library\n")
cat("Response: medv\n")
cat("Predictors: rm, lstat, ptratio, crim, nox, dis, chas (qualitative)\n\n")

# --- Fit multiple linear regression ---
mlr.fit <- lm(medv ~ rm + lstat + ptratio + crim + nox + dis + chas,
               data = Boston)

cat("--- Multiple Linear Regression Summary ---\n")
print(summary(mlr.fit))

# --- Key metrics ---
cat("\n--- Key metrics ---\n")
cat("R-squared:", round(summary(mlr.fit)$r.squared, 4), "\n")
cat("Adjusted R-squared:", round(summary(mlr.fit)$adj.r.squared, 4), "\n")
cat("RSE:", round(summary(mlr.fit)$sigma, 4), "\n")

# --- VIF computation ---
cat("\n--- VIF values ---\n")
pred_vars <- c("rm", "lstat", "ptratio", "crim", "nox", "dis")
vif_vals <- numeric(length(pred_vars))
names(vif_vals) <- pred_vars
for (p in pred_vars) {
  others <- pred_vars[pred_vars != p]
  fmla <- as.formula(paste(p, "~", paste(others, collapse = " + ")))
  r2 <- summary(lm(fmla, data = Boston))$r.squared
  vif_vals[p] <- 1 / (1 - r2)
}
print(round(vif_vals, 2))

# --- Compare with full model ---
mlr.full <- lm(medv ~ ., data = Boston)
cat("\n--- Full model (medv ~ .) for comparison ---\n")
cat("R-squared:", round(summary(mlr.full)$r.squared, 4), "\n")
cat("Adjusted R-squared:", round(summary(mlr.full)$adj.r.squared, 4), "\n")

# --- Interaction model: rm * lstat ---
mlr.interact <- lm(medv ~ rm * lstat + ptratio + crim + nox + dis + chas,
                    data = Boston)
cat("\n--- Model with rm:lstat interaction ---\n")
print(summary(mlr.interact))

cat("\n--- ANOVA: base model vs interaction model ---\n")
print(anova(mlr.fit, mlr.interact))

# --- Non-linear: quadratic lstat ---
mlr.quad <- lm(medv ~ rm + lstat + I(lstat^2) + ptratio + crim + nox + dis + chas,
                data = Boston)
cat("\n--- Model with quadratic lstat ---\n")
print(summary(mlr.quad))
cat("\n--- ANOVA: base vs quadratic ---\n")
print(anova(mlr.fit, mlr.quad))

# --- PLOT: 4-panel residual diagnostics with leverage ---
par(mfrow = c(2, 2))
plot(mlr.fit)
par(mfrow = c(1, 1))

# --- PLOT: Pairs plot ---
pairs(Boston[, c("medv", "rm", "lstat", "ptratio", "crim", "nox", "dis")],
      col = "steelblue", pch = 19, cex = 0.3,
      main = "Pairs Plot: Boston Selected Predictors")

# --- Correlation matrix ---
cat("\n--- Correlation matrix (selected predictors) ---\n")
cor_vars <- c("medv", "rm", "lstat", "ptratio", "crim", "nox", "dis")
print(round(cor(Boston[, cor_vars]), 3))

# --- High leverage observations ---
cat("\n--- High leverage observations ---\n")
lev <- hatvalues(mlr.fit)
high_lev <- which(lev > 2 * mean(lev))
cat("Observations with leverage > 2 * mean:", length(high_lev), "observations\n")
cat("Top 5 by leverage:\n")
top5_lev <- sort(lev, decreasing = TRUE)[1:5]
for (i in seq_along(top5_lev)) {
  idx <- as.integer(names(top5_lev)[i])
  cat(sprintf("  Obs %d: leverage = %.4f, medv = %.1f, rm = %.2f, lstat = %.2f\n",
              idx, top5_lev[i], Boston$medv[idx], Boston$rm[idx], Boston$lstat[idx]))
}

# --- Cook's distance ---
cat("\n--- Influential observations (Cook's distance) ---\n")
cd <- cooks.distance(mlr.fit)
high_cd <- which(cd > 4 / nrow(Boston))
cat("Observations with Cook's D > 4/n:", length(high_cd), "observations\n")
cat("Top 5 by Cook's distance:\n")
top5_cd <- sort(cd, decreasing = TRUE)[1:5]
for (i in seq_along(top5_cd)) {
  idx <- as.integer(names(top5_cd)[i])
  cat(sprintf("  Obs %d: Cook's D = %.4f, medv = %.1f, rm = %.2f\n",
              idx, top5_cd[i], Boston$medv[idx], Boston$rm[idx]))
}
