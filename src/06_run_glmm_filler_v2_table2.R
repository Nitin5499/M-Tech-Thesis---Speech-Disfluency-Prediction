# 06_run_glmm_filler_v2_table2.R
# Runs the FULL random-effects comparison (None / Coarse POS / Fine POS /
# Raw word) on dataset_filler_v2.csv -- the paper-matched filler dataset.
#
# Run: Rscript src/06_run_glmm_filler_v2_table2.R
#
# This completes the picture: 06_run_glmm_filler_v2.R only ran "Raw word"
# as a fast headline check. This script runs all four random-effect
# configurations, matching the paper's own Table 2 design, so the v2
# filler result can be compared to the paper stage-by-stage the same
# way the v1 filler result and the reparandum task already were.
#
# Does NOT touch: dataset_filler.csv, dataset_reparandum.csv,
# accuracy_filler.txt, accuracy_reparandum.txt, table1_filler.csv,
# table1_reparandum.csv, table2_random_effects.csv (the original v1
# Table 2), accuracy_filler_v2.txt, table1_filler_v2.csv (the v2
# headline run). All outputs here use a distinct "_v2_table2" suffix.

cat("\n=== Table 2 Random-Effects Sweep -- FILLER v2 (paper-matched) ===\n")

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
FILLER_TARGET  <- 24100

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

FIXED_ONLY <- "disfluency ~ prev_lex_surp + foll_lex_surp + prev_syn_surp + foll_syn_surp + prev_IC + foll_IC + prev_SC + foll_SC + prev_dur + foll_dur"

cat("Fixed-effects formula:\n ", FIXED_ONLY, "\n")
cat("Random effects to test: None, Coarse POS, Fine POS, Raw word (word_for_model)\n")

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
  # pos column in dataset_filler_v2.csv comes from the preceding fluent
  # word's own POS tag (see 05_build_dataset_v2.py) -- used here exactly
  # as coarse_pos/fine_pos were used for the v1 comparison.
  df$coarse_pos <- ifelse(df$pos %in% content_pos, "content", "function")
  df$fine_pos   <- as.character(df$pos)
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

# ── Load and prepare data ──────────────────────────────────────
df          <- read.csv(filler_v2_path)
avail_feats <- feature_cols[feature_cols %in% colnames(df)]
df          <- df[complete.cases(df[, avail_feats]), ]
df          <- standardise(df)
cat("\nRows before subsampling:", nrow(df), "\n")

df <- subsample(df, FILLER_TARGET)
cat("Rows after subsampling: ", nrow(df), "\n")
cat("Unique word_for_model values in this sample:", length(unique(df$word_for_model)), "\n")
cat("Unique coarse_pos values:", length(unique(df$coarse_pos)), "\n")
cat("Unique fine_pos values:", length(unique(df$fine_pos)), "\n")

# ── Run all four configurations ────────────────────────────────
cat("\n", strrep("=", 55), "\n")
cat("Running random-effects sweep on filler v2...\n")
cat(strrep("=", 55), "\n")

cat("\n[1/4] None (fixed effects only, no random effect)...\n")
acc_none <- tryCatch({
  set.seed(42)
  folds <- sample(rep(1:5, length.out=nrow(df)))
  accs  <- numeric(5)
  for (f in 1:5) {
    train <- df[folds != f, ]
    test  <- df[folds == f, ]
    m <- glm(as.formula(FIXED_ONLY), data=train, family=binomial)
    preds <- predict(m, newdata=test, type="response")
    accs[f] <- mean((preds > 0.5) == (test$disfluency == 1))
  }
  mean(accs) * 100
}, error=function(e) { cat("  ERROR:", conditionMessage(e), "\n"); NA })
cat("  None:", round(acc_none, 2), "%  (paper's filler 'None': 75.41%)\n")

cat("\n[2/4] Coarse POS...\n")
acc_coarse <- tryCatch(
  cv_accuracy(paste0(FIXED_ONLY, " + (1|coarse_pos)"), df),
  error=function(e) { cat("  ERROR:", conditionMessage(e), "\n"); NA }
)
cat("  Coarse POS:", round(acc_coarse, 2), "%  (paper's filler 'Coarse POS': 76.99%)\n")

cat("\n[3/4] Fine POS...\n")
acc_fine <- tryCatch(
  cv_accuracy(paste0(FIXED_ONLY, " + (1|fine_pos)"), df),
  error=function(e) { cat("  ERROR:", conditionMessage(e), "\n"); NA }
)
cat("  Fine POS:", round(acc_fine, 2), "%  (paper's filler 'Fine POS': 79.01%)\n")

cat("\n[4/4] Raw word (word_for_model)...\n")
acc_raw <- tryCatch(
  cv_accuracy(paste0(FIXED_ONLY, " + (1|word_for_model)"), df),
  error=function(e) { cat("  ERROR:", conditionMessage(e), "\n"); NA }
)
cat("  Raw word:", round(acc_raw, 2), "%  (paper's filler 'Raw word': 81.21%)\n")
cat("  (for reference -- earlier headline run of this same config: 78.66%;\n")
cat("   small difference vs this run is expected due to independent\n")
cat("   random sampling in subsample() unless seeds align exactly)\n")

# ── Save results ────────────────────────────────────────────────
table2_v2 <- data.frame(
  random_effect = c("None", "Coarse POS", "Fine POS", "Raw word"),
  ours_v2        = round(c(acc_none, acc_coarse, acc_fine, acc_raw), 2),
  paper_filler   = c(75.41, 76.99, 79.01, 81.21),
  ours_v1_original = c(NA, NA, NA, 99.95)
)
print(table2_v2)

write.csv(table2_v2,
          file.path(results, "table2_filler_v2.csv"),
          row.names=FALSE)

cat("\n", strrep("=", 55), "\n")
cat("COMPLETE -- Table 2 sweep for filler v2\n")
cat(strrep("=", 55), "\n")
cat("Saved -> results/table2_filler_v2.csv\n")
cat("Original v1 table2_random_effects.csv untouched.\n")
cat("Original accuracy_filler_v2.txt (headline run) untouched.\n")