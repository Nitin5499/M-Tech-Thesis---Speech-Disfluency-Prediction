# =============================================================
# 16_run_glmm_gpt2.R
# =============================================================
# GPT-2 SURPRISAL EXTENSION -- Stage 3 (final).
#
# Answers the research question directly: does adding GPT-2-based
# surprisal improve disfluency prediction accuracy over the existing
# 10-feature baseline model (which already includes ModelBlocks
# syntactic surprisal)?
#
# Runs TWO comparisons, matching the two dataset variants produced
# by 15_merge_gpt2_dataset.py:
#   1. filler task     (n ~= 35,926)
#   2. reparandum task (n ~= 36,326)
#
# Unlike the prosodic extension, there is no accent/break split here
# -- GPT-2 surprisal was computed on the FULL clean corpus (not a
# small annotated subset), so both comparisons run at close to full
# scale. This is a genuine advantage over the prosodic break_index
# comparisons, which ran on only ~12% of the corpus.
#
# For each comparison, this script fits:
#   BASELINE formula -- the existing 10 features (including
#                       ModelBlocks syntactic surprisal) + (1|word)
#   ENHANCED formula  -- baseline + prev_gpt2_surp + foll_gpt2_surp
# ... via 5-fold cross-validation (reusing the exact same
# cv_accuracy() logic from 13_run_glmm_prosodic.R), then reports
# both accuracies side by side plus a likelihood-ratio test for a
# formal significance check.
#
# WHY THIS COMPARISON MATTERS SPECIFICALLY: the baseline model
# already includes ModelBlocks syntactic surprisal, which only
# covers 71.7% of sentences. If GPT-2 surprisal adds real
# explanatory power ON TOP OF that existing feature, it suggests
# GPT-2 is capturing something ModelBlocks misses -- either through
# broader coverage, or through a genuinely different kind of
# predictive signal. This script does not itself test WHICH of
# those explanations is correct (that would need a direct
# correlation analysis between the two surprisal measures, planned
# as a follow-up) -- it only tests WHETHER GPT-2 adds anything.
#
# Author: Nitin Kumar
# Date: September 2026 (GPT-2 surprisal extension, stage 3)
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

# ---- Reused exactly from 13_run_glmm_prosodic.R ----
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

standardise_gpt2 <- function(df, cols) {
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

# ---- Run one full comparison: baseline vs baseline+GPT2, with LRT ----
run_comparison <- function(task_name, data_path, gpt2_cols, random_effect_col) {
  log_section(paste0("Comparison: ", task_name, " / GPT-2 surprisal"))

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
  df <- standardise_gpt2(df, gpt2_cols)
  df$word <- as.character(df[[random_effect_col]])
  df$disfluency <- as.integer(df$disfluency)

  # Drop any remaining incomplete rows across ALL model columns
  needed_cols <- c(feature_cols, gpt2_cols, "disfluency", "word")
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
                               paste(gpt2_cols, collapse = " + "), " + (1|word)")

  log_msg(paste0("  Baseline formula: ", baseline_formula))
  log_msg(paste0("  Enhanced formula: ", enhanced_formula))

  # ---- Cross-validated accuracy comparison ----
  log_msg("  Running 5-fold CV for baseline model...")
  acc_baseline <- cv_accuracy(baseline_formula, df_complete)
  log_msg(paste0("  Baseline CV accuracy: ", round(acc_baseline, 2), "%"))

  log_msg("  Running 5-fold CV for enhanced (GPT-2) model...")
  acc_enhanced <- cv_accuracy(enhanced_formula, df_complete)
  log_msg(paste0("  Enhanced CV accuracy: ", round(acc_enhanced, 2), "%"))

  acc_diff <- acc_enhanced - acc_baseline
  log_msg(paste0("  Accuracy difference (enhanced - baseline): ",
                  round(acc_diff, 2), " percentage points"))

  # ---- Likelihood-ratio test on full-data fits ----
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
        log_msg("  -> GPT-2 surprisal adds SIGNIFICANT explanatory power (p < 0.05)")
      } else {
        log_msg("  -> GPT-2 surprisal does NOT add significant explanatory power at p < 0.05")
      }
    } else {
      log_msg("  [WARNING] Likelihood-ratio test could not be computed.")
    }
  }

  list(
    task = task_name, n_rows = nrow(df_complete),
    acc_baseline = acc_baseline, acc_enhanced = acc_enhanced,
    acc_diff = acc_diff, lrt_p = lrt_p
  )
}

# =============================================================
# Run both comparisons
# =============================================================

log_section("GPT-2 Surprisal Extension -- Stage 3: GLMM Comparisons")

results <- list()

results[["filler_gpt2"]] <- run_comparison(
  "filler",
  file.path(PROCESSED_DIR, "dataset_filler_gpt2.csv"),
  c("prev_gpt2_surp", "foll_gpt2_surp"),
  "word_for_model"
)

results[["reparandum_gpt2"]] <- run_comparison(
  "reparandum",
  file.path(PROCESSED_DIR, "dataset_reparandum_gpt2.csv"),
  c("prev_gpt2_surp", "foll_gpt2_surp"),
  "word"
)

# ---- Build and save the summary table ----
log_section("Stage 3 COMPLETE -- Summary Table")

summary_rows <- list()
for (name in names(results)) {
  r <- results[[name]]
  if (!is.null(r)) {
    summary_rows[[name]] <- data.frame(
      task = r$task, n_rows = r$n_rows,
      acc_baseline = round(r$acc_baseline, 2),
      acc_enhanced = round(r$acc_enhanced, 2),
      acc_diff = round(r$acc_diff, 2),
      lrt_p = r$lrt_p
    )
  }
}

if (length(summary_rows) > 0) {
  summary_df <- do.call(rbind, summary_rows)
  out_path <- file.path(RESULTS_DIR, "table_gpt2_comparison.csv")
  write.csv(summary_df, out_path, row.names = FALSE)
  log_msg(paste0("Saved summary table -> ", out_path))
  print(summary_df)
} else {
  log_msg("[WARNING] No comparisons completed successfully -- nothing to summarize.")
}

log_msg("\nNOTE: this script tests WHETHER GPT-2 surprisal adds explanatory")
log_msg("power on top of the existing 10-feature baseline (which already")
log_msg("includes ModelBlocks syntactic surprisal). It does NOT test WHY --")
log_msg("that requires a direct correlation analysis between GPT-2 surprisal")
log_msg("and ModelBlocks syntactic surprisal, planned as a follow-up once")
log_msg("this comparison's result is known.")