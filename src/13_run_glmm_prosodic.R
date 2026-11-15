# =============================================================
# 13_run_glmm_prosodic.R
# =============================================================
# PROSODIC EXTENSION -- Stage 7 of 7. THE FINAL SCRIPT.
#
# Answers the research question directly: does adding prosodic
# features (pitch accent strength, break/pause strength) improve
# disfluency prediction accuracy over the existing 10-feature model?
#
# Runs FOUR separate baseline-vs-enhanced comparisons, matching the
# four dataset variants produced by 12_merge_prosodic_dataset.py:
#   1. filler task     x accent features (n ~= 35,926)
#   2. filler task     x break features  (n ~= 2,775)
#   3. reparandum task x accent features (n ~= 36,326)
#   4. reparandum task x break features  (n ~= 2,667)
#
# For each comparison, this script fits:
#   BASELINE formula  -- the existing 10 features + (1|word_for_model
#                         or word), matching the original pipeline
#   ENHANCED formula   -- baseline + the two new prosodic columns
# ... via 5-fold cross-validation (reusing the exact same
# cv_accuracy() logic used throughout the rest of this project's R
# scripts), then reports both accuracies side by side plus a
# likelihood-ratio test (anova() on the two full-data model fits)
# for a formal significance check, not just an eyeballed accuracy
# gap.
#
# HONEST CAVEAT, stated here and printed in every comparison's
# output: the *_break comparisons run on ~7% of their *_accent
# counterparts' sample size, reflecting break_index's real ~12%
# annotation coverage in the underlying corpus (see stages 5-6 logs).
# Any break_index result should be interpreted with wider uncertainty
# than the accent_strength results, and this should be stated
# plainly in the results writeup, not glossed over.
#
# Author: Nitin Kumar
# Date: August 2026 (prosodic extension, stage 7)
# =============================================================

library(lme4)

# ---- Paths (matching existing R scripts' convention) ----
PROCESSED_DIR <- "data/processed"
RESULTS_DIR   <- "results"
if (!dir.exists(RESULTS_DIR)) dir.create(RESULTS_DIR)

LOG_FILE <- file.path(PROCESSED_DIR, "pipeline_log.txt")

log_msg <- function(msg) {
  ts <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  line <- paste0("[", ts, "] ", msg)
  cat(line, "\n")
  cat(line, "\n", file = LOG_FILE, append = TRUE)
}

log_section <- function(title) {
  line <- paste0("\n", strrep("=", 60), "\n", title, "\n", strrep("=", 60))
  cat(line, "\n")
  cat(line, "\n", file = LOG_FILE, append = TRUE)
}

# ---- Reused exactly from the existing pipeline's R scripts ----
feature_cols <- c(
  "prev_lex_surp", "foll_lex_surp", "prev_syn_surp", "foll_syn_surp",
  "prev_IC", "foll_IC", "prev_SC", "foll_SC", "prev_dur", "foll_dur"
)

standardise <- function(df) {
  for (col in feature_cols) {
    if (col %in% colnames(df)) {
      s <- sd(df[[col]], na.rm = TRUE)
      if (!is.na(s) && s > 0)
        df[[col]] <- (df[[col]] - mean(df[[col]], na.rm = TRUE)) / s
    }
  }
  df
}

# Also standardise the new prosodic columns, same treatment as every
# other feature -- keeps coefficient magnitudes comparable.
standardise_prosodic <- function(df, cols) {
  for (col in cols) {
    if (col %in% colnames(df)) {
      s <- sd(df[[col]], na.rm = TRUE)
      if (!is.na(s) && s > 0)
        df[[col]] <- (df[[col]] - mean(df[[col]], na.rm = TRUE)) / s
    }
  }
  df
}

cv_accuracy <- function(formula_str, data, k = 5) {
  set.seed(42)
  folds <- sample(rep(1:k, length.out = nrow(data)))
  accs  <- numeric(k)
  for (f in 1:k) {
    train <- data[folds != f, ]
    test  <- data[folds == f, ]
    m <- tryCatch(
      glmer(as.formula(formula_str), data = train, family = binomial,
            control = glmerControl(optimizer = "bobyqa",
                                   optCtrl = list(maxfun = 1e5))),
      error = function(e) NULL
    )
    if (is.null(m)) {
      accs[f] <- NA
    } else {
      preds   <- predict(m, newdata = test, type = "response", allow.new.levels = TRUE)
      accs[f] <- mean((preds > 0.5) == (test$disfluency == 1))
    }
  }
  mean(accs, na.rm = TRUE) * 100
}

# ---- Run one full comparison: baseline vs enhanced, with LRT ----
run_comparison <- function(task_name, variant_name, data_path,
                            prosodic_cols, random_effect_col) {
  log_section(paste0("Comparison: ", task_name, " / ", variant_name))

  if (!file.exists(data_path)) {
    log_msg(paste0("  [SKIP] File not found: ", data_path))
    return(NULL)
  }

  df <- read.csv(data_path, stringsAsFactors = FALSE)
  log_msg(paste0("  Loaded ", nrow(df), " rows from ", data_path))

  if (!(random_effect_col %in% colnames(df))) {
    log_msg(paste0("  [SKIP] Random effect column '", random_effect_col,
                    "' not found in this dataset -- check column name."))
    return(NULL)
  }

  df <- standardise(df)
  df <- standardise_prosodic(df, prosodic_cols)
  df$word <- as.character(df[[random_effect_col]])
  df$disfluency <- as.integer(df$disfluency)

  # Drop any remaining incomplete rows across ALL model columns
  # (baseline features + prosodic features + outcome + random effect)
  needed_cols <- c(feature_cols, prosodic_cols, "disfluency", "word")
  needed_cols <- needed_cols[needed_cols %in% colnames(df)]
  df_complete <- df[complete.cases(df[, needed_cols]), ]
  n_dropped <- nrow(df) - nrow(df_complete)
  log_msg(paste0("  Rows after final completeness check: ", nrow(df_complete),
                  " (", n_dropped, " dropped for missing values in model columns)"))

  if (nrow(df_complete) < 50) {
    log_msg("  [SKIP] Fewer than 50 complete rows -- not enough data to fit a model.")
    return(NULL)
  }

  baseline_terms <- paste(feature_cols[feature_cols %in% colnames(df_complete)], collapse = " + ")
  baseline_formula <- paste0("disfluency ~ ", baseline_terms, " + (1|word)")
  enhanced_formula  <- paste0("disfluency ~ ", baseline_terms, " + ",
                               paste(prosodic_cols, collapse = " + "), " + (1|word)")

  log_msg(paste0("  Baseline formula: ", baseline_formula))
  log_msg(paste0("  Enhanced formula: ", enhanced_formula))

  # ---- Cross-validated accuracy comparison ----
  log_msg("  Running 5-fold CV for baseline model...")
  acc_baseline <- cv_accuracy(baseline_formula, df_complete)
  log_msg(paste0("  Baseline CV accuracy: ", round(acc_baseline, 2), "%"))

  log_msg("  Running 5-fold CV for enhanced (prosodic) model...")
  acc_enhanced <- cv_accuracy(enhanced_formula, df_complete)
  log_msg(paste0("  Enhanced CV accuracy: ", round(acc_enhanced, 2), "%"))

  acc_diff <- acc_enhanced - acc_baseline
  log_msg(paste0("  Accuracy difference (enhanced - baseline): ",
                  round(acc_diff, 2), " percentage points"))

  # ---- Likelihood-ratio test on full-data fits, for a formal
  # significance check beyond the raw accuracy gap ----
  log_msg("  Fitting full-data models for likelihood-ratio test...")
  m_baseline <- tryCatch(
    glmer(as.formula(baseline_formula), data = df_complete, family = binomial,
          control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))),
    error = function(e) { log_msg(paste0("    Baseline fit failed: ", e$message)); NULL }
  )
  m_enhanced <- tryCatch(
    glmer(as.formula(enhanced_formula), data = df_complete, family = binomial,
          control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))),
    error = function(e) { log_msg(paste0("    Enhanced fit failed: ", e$message)); NULL }
  )

  lrt_p <- NA
  if (!is.null(m_baseline) && !is.null(m_enhanced)) {
    lrt <- tryCatch(anova(m_baseline, m_enhanced), error = function(e) NULL)
    if (!is.null(lrt)) {
      lrt_p <- lrt[["Pr(>Chisq)"]][2]
      log_msg(paste0("  Likelihood-ratio test p-value: ", format.pval(lrt_p, digits = 4)))
      if (!is.na(lrt_p) && lrt_p < 0.05) {
        log_msg("  -> Prosodic features add SIGNIFICANT explanatory power (p < 0.05)")
      } else {
        log_msg("  -> Prosodic features do NOT add significant explanatory power at p < 0.05")
      }
    } else {
      log_msg("  [WARNING] Likelihood-ratio test could not be computed.")
    }
  }

  list(
    task = task_name, variant = variant_name,
    n_rows = nrow(df_complete),
    acc_baseline = acc_baseline, acc_enhanced = acc_enhanced,
    acc_diff = acc_diff, lrt_p = lrt_p
  )
}

# =============================================================
# Run all four comparisons
# =============================================================

log_section("Prosodic Extension -- Stage 7: GLMM Comparisons")

results <- list()

results[["filler_accent"]] <- run_comparison(
  "filler", "accent",
  file.path(PROCESSED_DIR, "dataset_filler_accent.csv"),
  c("prev_accent_strength", "foll_accent_strength"),
  "word_for_model"
)

results[["filler_break"]] <- run_comparison(
  "filler", "break",
  file.path(PROCESSED_DIR, "dataset_filler_break.csv"),
  c("prev_break_index", "foll_break_index"),
  "word_for_model"
)

results[["reparandum_accent"]] <- run_comparison(
  "reparandum", "accent",
  file.path(PROCESSED_DIR, "dataset_reparandum_accent.csv"),
  c("prev_accent_strength", "foll_accent_strength"),
  "word"
)

results[["reparandum_break"]] <- run_comparison(
  "reparandum", "break",
  file.path(PROCESSED_DIR, "dataset_reparandum_break.csv"),
  c("prev_break_index", "foll_break_index"),
  "word"
)

# ---- Build and save the summary table ----
log_section("Stage 7 COMPLETE -- Summary Table")

summary_rows <- list()
for (name in names(results)) {
  r <- results[[name]]
  if (!is.null(r)) {
    summary_rows[[name]] <- data.frame(
      task = r$task, variant = r$variant, n_rows = r$n_rows,
      acc_baseline = round(r$acc_baseline, 2),
      acc_enhanced = round(r$acc_enhanced, 2),
      acc_diff = round(r$acc_diff, 2),
      lrt_p = r$lrt_p
    )
  }
}

if (length(summary_rows) > 0) {
  summary_df <- do.call(rbind, summary_rows)
  out_path <- file.path(RESULTS_DIR, "table_prosodic_comparison.csv")
  write.csv(summary_df, out_path, row.names = FALSE)
  log_msg(paste0("Saved summary table -> ", out_path))
  print(summary_df)
} else {
  log_msg("[WARNING] No comparisons completed successfully -- nothing to summarize.")
}

log_msg("\nHONEST CAVEAT for interpreting this table: filler_break and")
log_msg("reparandum_break ran on a much smaller sample (~7% of their")
log_msg("_accent counterparts), reflecting break_index's real ~12% corpus")
log_msg("annotation coverage. Any break_index finding should be reported")
log_msg("with this sample-size caveat stated explicitly, not silently.")