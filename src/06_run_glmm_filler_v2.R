# 06_run_glmm_filler_v2.R
# Runs the GLMM on dataset_filler_v2.csv (paper-matched filler methodology)
# Run: Rscript src/06_run_glmm_filler_v2.R
#
# This is a companion script to 06_run_glmm.R -- it does NOT touch any
# existing output files. All results are written with a "_v2" suffix
# so the original filler run (99.95%) remains available for comparison.
#
# RUN_TABLE2 is left FALSE here deliberately -- this is a fast headline
# check first. If the result looks meaningful, we can add the full
# random-effects comparison as a follow-up run.

cat("\n=== GLMM Modeling (lme4) -- FILLER v2 (paper-matched methodology) ===\n")

.libPaths(c("C:/Rlibs", .libPaths()))
suppressPackageStartupMessages({
  library(lme4,     lib.loc="C:/Rlibs")
  library(lmerTest, lib.loc="C:/Rlibs")
})
cat("lme4 and lmerTest loaded\n")

root_dir  <- getwd()
processed <- file.path(root_dir, "data", "processed")
results   <- file.path(root_dir, "results")
dir.create(results, showWarnings=FALSE)

filler_v2_path <- file.path(processed, "dataset_filler_v2.csv")

FILLER_TARGET <- 24100   # same target as original, for comparability

feature_cols <- c(
  "prev_lex_surp","foll_lex_surp",
  "prev_syn_surp","foll_syn_surp",
  "prev_IC","foll_IC",
  "prev_SC","foll_SC",
  "prev_dur","foll_dur"
)

content_pos <- c(
  "NN","NNS","NNP","NNPS",
  "VB","VBD","VBG","VBN","VBP","VBZ",
  "JJ","JJR","JJS","RB","RBR","RBS"
)

FULL_FORMULA <- "disfluency ~ prev_lex_surp + foll_lex_surp + prev_syn_surp + foll_syn_surp + prev_IC + foll_IC + prev_SC + foll_SC + prev_dur + foll_dur + (1|word_for_model)"

cat("Full formula:\n ", FULL_FORMULA, "\n")

standardise <- function(df) {
  for (col in feature_cols) {
    if (col %in% colnames(df)) {
      s <- sd(df[[col]], na.rm=TRUE)
      if (!is.na(s) && s > 0)
        df[[col]] <- (df[[col]] - mean(df[[col]], na.rm=TRUE)) / s
    }
  }
  df$word_for_model <- as.character(df$word_for_model)
  df$disfluency     <- as.integer(df$disfluency)
  df
}

subsample <- function(df, target) {
  if (nrow(df) <= target) return(df)
  set.seed(42)
  n_each  <- target %/% 2
  pos_idx <- which(df$disfluency == 1)
  neg_idx <- which(df$disfluency == 0)
  df[c(sample(pos_idx, min(n_each, length(pos_idx))),
       sample(neg_idx, min(n_each, length(neg_idx)))), ]
}

cv_accuracy <- function(formula_str, data, k=5) {
  set.seed(42)
  folds <- sample(rep(1:k, length.out=nrow(data)))
  accs  <- numeric(k)
  for (f in 1:k) {
    train <- data[folds != f, ]
    test  <- data[folds == f, ]
    m <- tryCatch(
      glmer(as.formula(formula_str), data=train, family=binomial,
            control=glmerControl(optimizer="bobyqa",
                                 optCtrl=list(maxfun=1e5))),
      error=function(e) {
        cat("  fold", f, "error:", conditionMessage(e), "\n")
        NULL
      }
    )
    if (is.null(m)) {
      accs[f] <- NA
    } else {
      preds   <- predict(m, newdata=test, type="response",
                         allow.new.levels=TRUE)
      accs[f] <- mean((preds > 0.5) == (test$disfluency == 1))
    }
  }
  mean(accs, na.rm=TRUE) * 100
}

cat("\n", strrep("=", 55), "\n")
cat("Task: filler_v2 (paper-matched methodology)\n")
cat(strrep("=", 55), "\n")

df          <- read.csv(filler_v2_path)
avail_feats <- feature_cols[feature_cols %in% colnames(df)]
df          <- df[complete.cases(df[, avail_feats]), ]
df          <- standardise(df)
cat("Rows before subsampling:", nrow(df), "\n")
cat("Unique word_for_model values:", length(unique(df$word_for_model)), "\n")

df <- subsample(df, FILLER_TARGET)
cat("Rows after subsampling: ", nrow(df), "\n")

cat("\nBaseline (random intercept only)...\n")
base_acc <- cv_accuracy("disfluency ~ 1 + (1|word_for_model)", df)
cat("  Baseline:", round(base_acc, 2), "%")
cat("  (original v1 filler baseline was 99.96% -- expect much lower here)\n")
cat("  (paper's filler baseline: 66.80%)\n")

cat("\nFitting full model...\n")
full_model <- tryCatch(
  glmer(as.formula(FULL_FORMULA), data=df, family=binomial,
        control=glmerControl(optimizer="bobyqa",
                             optCtrl=list(maxfun=2e5))),
  error=function(e) { cat("ERROR:", conditionMessage(e), "\n"); NULL }
)

if (is.null(full_model)) {
  cat("Full model failed -- stopping.\n")
  quit(status=1)
}

coef_df           <- as.data.frame(summary(full_model)$coefficients)
coef_df$feature   <- rownames(coef_df)
rownames(coef_df) <- NULL
cat("\nCoefficients:\n")
print(coef_df[, c("feature","Estimate","Std. Error","z value","Pr(>|z|)")])

cat("\nRunning 5-fold CV...\n")
full_acc <- cv_accuracy(FULL_FORMULA, df)
cat("  Full model:", round(full_acc, 2), "%")
cat("  (original v1 filler full model: 99.95%)\n")
cat("  (paper's filler full model: 81.21%)\n")

write.csv(coef_df,
          file.path(results, "table1_filler_v2.csv"),
          row.names=FALSE)
writeLines(
  c(paste("Baseline:            ", round(base_acc, 2), "%"),
    paste("Full model (v2):     ", round(full_acc,  2), "%"),
    paste("Original v1 full model:", "99.95%"),
    paste("Paper baseline:      ", "66.80%"),
    paste("Paper full model:    ", "81.21%")),
  file.path(results, "accuracy_filler_v2.txt")
)

cat("\n", strrep("=", 55), "\n")
cat("COMPLETE -- FILLER v2\n")
cat(strrep("=", 55), "\n")
cat("v2 (paper-matched) filler accuracy:", round(full_acc, 2), "%\n")
cat("v1 (original) filler accuracy:      99.95%\n")
cat("Paper's filler accuracy:            81.21%\n")
cat("\nResults saved: results/accuracy_filler_v2.txt, results/table1_filler_v2.csv\n")
cat("Original v1 results untouched.\n")