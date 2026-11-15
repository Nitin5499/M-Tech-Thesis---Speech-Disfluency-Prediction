# =============================================================
# 13b_run_glmm_prosodic_auc.R
# =============================================================
# Same four baseline-vs-enhanced prosody comparisons as 13, but this
# one also reports AUC and the always-guess accuracy (action item 2),
# so the prosody table matches the multi-LLM table.
#
# Everything that affects the accuracy is copied from 13: the same
# standardising, the same complete-case filter, the same formulas,
# the same 5-fold split (seed 42), threshold 0.5, bobyqa. So the
# accuracies should match table_prosodic_comparison_final_nov3.csv
# exactly. The script prints that check at the end.
#
# New compared to 13:
#   - out-of-fold predicted probabilities are kept for every row
#   - AUC (baseline, enhanced) from the rank formula
#   - always-guess accuracy (majority class share)
#   - out-of-fold predictions saved in results/cv_preds/
#
# out: results/table_prosodic_comparison_auc.csv
#      (does NOT overwrite table_prosodic_comparison.csv or the final_nov3 copy)
#      results/cv_preds/prosody_<task>_<variant>.csv
#
# run from repo root:
#   "C:\Program Files\R\R-4.5.3\bin\Rscript.exe" src\13b_run_glmm_prosodic_auc.R
# =============================================================

library(lme4)

PROCESSED_DIR <- "data/processed"
RESULTS_DIR   <- "results"
PRED_DIR      <- file.path(RESULTS_DIR, "cv_preds")
if (!dir.exists(RESULTS_DIR)) dir.create(RESULTS_DIR)
if (!dir.exists(PRED_DIR))    dir.create(PRED_DIR)

T_START <- Sys.time()
log_msg <- function(msg) {
  mins <- as.numeric(difftime(Sys.time(), T_START, units = "mins"))
  cat(sprintf("[%s | %5.1f min] %s\n", format(Sys.time(), "%H:%M:%S"), mins, msg))
}

# same ten baseline features as 13
feature_cols <- c(
  "prev_lex_surp", "foll_lex_surp", "prev_syn_surp", "foll_syn_surp",
  "prev_IC", "foll_IC", "prev_SC", "foll_SC", "prev_dur", "foll_dur"
)

# z-score a set of columns (same as standardise / standardise_prosodic in 13)
zscore_cols <- function(df, cols) {
  for (col in cols) {
    if (col %in% colnames(df)) {
      s <- sd(df[[col]], na.rm = TRUE)
      if (!is.na(s) && s > 0)
        df[[col]] <- (df[[col]] - mean(df[[col]], na.rm = TRUE)) / s
    }
  }
  df
}

# 5-fold CV like 13, but also returns the out-of-fold probability of every row
cv_run <- function(formula_str, data, k = 5) {
  set.seed(42)
  folds <- sample(rep(1:k, length.out = nrow(data)))
  accs  <- numeric(k)
  oof   <- rep(NA_real_, nrow(data))
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
      preds <- predict(m, newdata = test, type = "response", allow.new.levels = TRUE)
      oof[folds == f] <- preds
      accs[f] <- mean((preds > 0.5) == (test$disfluency == 1))
    }
  }
  list(acc = mean(accs, na.rm = TRUE) * 100, oof = oof)
}

# AUC from ranks: (sum of ranks of positives - n1(n1+1)/2) / (n1 * n0)
auc_rank <- function(y, p) {
  ok <- !is.na(p)
  y <- y[ok]; p <- p[ok]
  r  <- rank(p)
  n1 <- sum(y == 1)
  n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

run_comparison <- function(task_name, variant_name, data_path,
                           prosodic_cols, random_effect_col) {
  cat("\n", strrep("=", 60), "\n", task_name, " / ", variant_name, "\n",
      strrep("=", 60), "\n", sep = "")

  if (!file.exists(data_path)) {
    log_msg(paste0("[SKIP] File not found: ", data_path))
    return(NULL)
  }
  df <- read.csv(data_path, stringsAsFactors = FALSE)
  log_msg(paste0("Loaded ", nrow(df), " rows from ", data_path))

  if (!(random_effect_col %in% colnames(df))) {
    log_msg(paste0("[SKIP] random effect column '", random_effect_col, "' not found"))
    return(NULL)
  }

  # standardise first, then drop incomplete rows (same order as 13)
  df <- zscore_cols(df, feature_cols)
  df <- zscore_cols(df, prosodic_cols)
  df$word <- as.character(df[[random_effect_col]])
  df$disfluency <- as.integer(df$disfluency)

  needed_cols <- c(feature_cols, prosodic_cols, "disfluency", "word")
  needed_cols <- needed_cols[needed_cols %in% colnames(df)]
  dm <- df[complete.cases(df[, needed_cols]), ]
  log_msg(paste0("Rows after completeness check: ", nrow(dm),
                 " (", nrow(df) - nrow(dm), " dropped)"))
  if (nrow(dm) < 50) {
    log_msg("[SKIP] fewer than 50 complete rows")
    return(NULL)
  }

  base_terms <- paste(feature_cols[feature_cols %in% colnames(dm)], collapse = " + ")
  base_f <- paste0("disfluency ~ ", base_terms, " + (1|word)")
  enh_f  <- paste0("disfluency ~ ", base_terms, " + ",
                   paste(prosodic_cols, collapse = " + "), " + (1|word)")

  log_msg("baseline 5-fold CV ...")
  cvb <- cv_run(base_f, dm)
  log_msg(sprintf("baseline accuracy %.2f%%", cvb$acc))

  log_msg("enhanced 5-fold CV ...")
  cve <- cv_run(enh_f, dm)
  log_msg(sprintf("enhanced accuracy %.2f%%  (gain %+.2f)", cve$acc, cve$acc - cvb$acc))

  # likelihood-ratio test on the full data, same as 13
  ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
  m_b <- tryCatch(glmer(as.formula(base_f), data = dm, family = binomial, control = ctrl),
                  error = function(e) NULL)
  m_e <- tryCatch(glmer(as.formula(enh_f),  data = dm, family = binomial, control = ctrl),
                  error = function(e) NULL)
  lrt_p <- NA
  if (!is.null(m_b) && !is.null(m_e)) {
    lrt <- tryCatch(anova(m_b, m_e), error = function(e) NULL)
    if (!is.null(lrt)) lrt_p <- lrt[["Pr(>Chisq)"]][2]
  }

  y <- dm$disfluency
  auc_b  <- auc_rank(y, cvb$oof)
  auc_e  <- auc_rank(y, cve$oof)
  always <- max(mean(y == 1), mean(y == 0)) * 100
  log_msg(sprintf("AUC baseline %.4f, enhanced %.4f; always-guess %.2f%%; LRT p = %s",
                  auc_b, auc_e, always, format.pval(lrt_p, digits = 4)))

  write.csv(data.frame(row = seq_len(nrow(dm)), word = dm$word, y = y,
                       p_base = cvb$oof, p_enh = cve$oof),
            file.path(PRED_DIR, sprintf("prosody_%s_%s.csv", task_name, variant_name)),
            row.names = FALSE)

  data.frame(
    task = task_name, variant = variant_name, n_rows = nrow(dm),
    acc_baseline = round(cvb$acc, 2), acc_enhanced = round(cve$acc, 2),
    acc_diff = round(cve$acc - cvb$acc, 2),
    auc_baseline = round(auc_b, 4), auc_enhanced = round(auc_e, 4),
    auc_diff = round(auc_e - auc_b, 4),
    always_guess = round(always, 2),
    lrt_p = lrt_p
  )
}

# =============================================================
# four comparisons, same datasets and columns as 13
# =============================================================
cat(strrep("=", 60), "\nProsody comparisons with AUC and always-guess\n",
    strrep("=", 60), "\n", sep = "")

rows <- list()
rows[["filler_accent"]] <- run_comparison(
  "filler", "accent", file.path(PROCESSED_DIR, "dataset_filler_accent.csv"),
  c("prev_accent_strength", "foll_accent_strength"), "word_for_model")
rows[["filler_break"]] <- run_comparison(
  "filler", "break", file.path(PROCESSED_DIR, "dataset_filler_break.csv"),
  c("prev_break_index", "foll_break_index"), "word_for_model")
rows[["reparandum_accent"]] <- run_comparison(
  "reparandum", "accent", file.path(PROCESSED_DIR, "dataset_reparandum_accent.csv"),
  c("prev_accent_strength", "foll_accent_strength"), "word")
rows[["reparandum_break"]] <- run_comparison(
  "reparandum", "break", file.path(PROCESSED_DIR, "dataset_reparandum_break.csv"),
  c("prev_break_index", "foll_break_index"), "word")

rows <- rows[!sapply(rows, is.null)]
if (length(rows) == 0) stop("no comparison finished")

out <- do.call(rbind, rows)
out_path <- file.path(RESULTS_DIR, "table_prosodic_comparison_auc.csv")
write.csv(out, out_path, row.names = FALSE)
cat("\n"); print(out, row.names = FALSE)
log_msg(paste0("Saved -> ", out_path))

# check against the table already used for the results
ref_path <- file.path(RESULTS_DIR, "table_prosodic_comparison_final_nov3.csv")
if (file.exists(ref_path)) {
  ref <- read.csv(ref_path)
  chk <- merge(out[, c("task", "variant", "n_rows", "acc_baseline", "acc_enhanced")],
               ref[, c("task", "variant", "n_rows", "acc_baseline", "acc_enhanced")],
               by = c("task", "variant"), suffixes = c("_new", "_final"))
  chk$same <- chk$n_rows_new == chk$n_rows_final &
              abs(chk$acc_baseline_new - chk$acc_baseline_final) < 0.01 &
              abs(chk$acc_enhanced_new - chk$acc_enhanced_final) < 0.01
  cat("\nCheck against table_prosodic_comparison_final_nov3.csv:\n")
  print(chk, row.names = FALSE)
  if (all(chk$same)) log_msg("All four rows match the final table.")
  else log_msg("WARNING: some rows differ from the final table, look before using the AUC numbers.")
}