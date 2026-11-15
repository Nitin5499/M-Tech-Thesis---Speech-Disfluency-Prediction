# =============================================================
# 13d_run_glmm_prosody_vs_gpt2.R
# =============================================================
# Side-by-side on the SAME rows: baseline, baseline + GPT-2 surprisal,
# baseline + prosody, and baseline + both. One block for each of the four
# prosody datasets (filler/reparandum x accent/break).
#
# Why: GPT-2 numbers in the multi-LLM table are on the full data (37,543 /
# 40,732 rows), the prosody numbers are on much smaller sets. To put them in
# one table the model has to be fitted on identical rows, so here the GPT-2
# columns are merged onto each prosody dataset by word_id first.
#
# Same z-scoring, same formula style, same 5-fold CV (seed 42) as script 13.
#
# run from the project folder:
#   "C:\Program Files\R\R-4.5.3\bin\Rscript.exe" src/13d_run_glmm_prosody_vs_gpt2.R
# out: results/table_prosody_vs_gpt2_same_rows.csv
# =============================================================

library(lme4)

PROCESSED_DIR <- "data/processed"
RESULTS_DIR   <- "results"
if (!dir.exists(RESULTS_DIR)) dir.create(RESULTS_DIR)

feature_cols <- c(
  "prev_lex_surp", "foll_lex_surp", "prev_syn_surp", "foll_syn_surp",
  "prev_IC", "foll_IC", "prev_SC", "foll_SC", "prev_dur", "foll_dur"
)
gpt_cols <- c("prev_gpt2_surp", "foll_gpt2_surp")

# files that hold the GPT-2 columns (the same ones used for the LLM results)
GPT_FILE <- c(
  filler     = file.path(PROCESSED_DIR, "dataset_filler_fixfoll_multillm.csv"),
  reparandum = file.path(PROCESSED_DIR, "dataset_reparandum_multillm.csv")
)

zscore <- function(df, cols) {
  for (col in cols) {
    if (col %in% colnames(df)) {
      s <- sd(df[[col]], na.rm = TRUE)
      if (!is.na(s) && s > 0)
        df[[col]] <- (df[[col]] - mean(df[[col]], na.rm = TRUE)) / s
    }
  }
  df
}

# fit one glmer and keep track of warnings. If the first fit warns, restart it
# from where it ended with more iterations (usual fix for convergence warnings).
# returns the model and whether a warning is still there after the restart.
fit_glmer <- function(formula_str, data) {
  warned <- FALSE
  m <- withCallingHandlers(
    glmer(as.formula(formula_str), data = data, family = binomial,
          control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))),
    warning = function(w) { warned <<- TRUE; invokeRestart("muffleWarning") }
  )
  if (warned) {
    warned <- FALSE
    start_vals <- list(theta = getME(m, "theta"), fixef = fixef(m))
    m <- withCallingHandlers(
      glmer(as.formula(formula_str), data = data, family = binomial, start = start_vals,
            control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 5e5))),
      warning = function(w) { warned <<- TRUE; invokeRestart("muffleWarning") }
    )
  }
  list(model = m, warned = warned)
}

cv_accuracy <- function(formula_str, data, label = "", k = 5) {
  set.seed(42)
  folds <- sample(rep(1:k, length.out = nrow(data)))
  accs  <- numeric(k)
  n_warn <- 0
  for (f in 1:k) {
    train <- data[folds != f, ]
    test  <- data[folds == f, ]
    fit <- tryCatch(fit_glmer(formula_str, train), error = function(e) NULL)
    if (is.null(fit)) {
      accs[f] <- NA
    } else {
      if (fit$warned) n_warn <- n_warn + 1
      preds <- predict(fit$model, newdata = test, type = "response", allow.new.levels = TRUE)
      accs[f] <- mean((preds > 0.5) == (test$disfluency == 1))
    }
  }
  if (n_warn > 0) cat("   [", label, "] folds still warning after restart:", n_warn, "of", k, "\n")
  mean(accs, na.rm = TRUE) * 100
}

fit_full <- function(formula_str, data, label = "") {
  fit <- fit_glmer(formula_str, data)
  if (fit$warned) cat("   [", label, "] full-data fit still warning after restart\n")
  fit$model
}

run_one <- function(task_name, variant_name, data_path, pros_cols, re_col) {
  cat("\n---", task_name, "/", variant_name, "---\n")
  df <- read.csv(data_path, stringsAsFactors = FALSE)
  cat("prosody dataset rows:", nrow(df), "\n")

  gpt <- read.csv(GPT_FILE[[task_name]], stringsAsFactors = FALSE)
  if (!("word_id" %in% colnames(df)) || !("word_id" %in% colnames(gpt)) ||
      !all(gpt_cols %in% colnames(gpt))) {
    cat("columns in prosody file:", paste(colnames(df), collapse = ", "), "\n")
    cat("columns in GPT-2 file:  ", paste(colnames(gpt), collapse = ", "), "\n")
    stop("need word_id in both files and prev_gpt2_surp / foll_gpt2_surp in the GPT-2 file")
  }
  gpt <- gpt[!duplicated(gpt$word_id), c("word_id", gpt_cols)]
  df  <- merge(df, gpt, by = "word_id", all.x = TRUE)

  df <- zscore(df, c(feature_cols, gpt_cols, pros_cols))
  df$word <- as.character(df[[re_col]])
  df$disfluency <- as.integer(df$disfluency)

  needed <- c(feature_cols, gpt_cols, pros_cols, "disfluency", "word")
  d <- df[complete.cases(df[, needed]), ]
  cat("rows used (same for every model):", nrow(d), "\n")

  base  <- paste(feature_cols, collapse = " + ")
  f_base <- paste0("disfluency ~ ", base, " + (1|word)")
  f_gpt  <- paste0("disfluency ~ ", base, " + ", paste(gpt_cols, collapse = " + "), " + (1|word)")
  f_pros <- paste0("disfluency ~ ", base, " + ", paste(pros_cols, collapse = " + "), " + (1|word)")
  f_both <- paste0("disfluency ~ ", base, " + ", paste(gpt_cols, collapse = " + "), " + ",
                   paste(pros_cols, collapse = " + "), " + (1|word)")

  a_base <- cv_accuracy(f_base, d, "baseline"); cat("baseline      ", round(a_base, 2), "\n")
  a_gpt  <- cv_accuracy(f_gpt,  d, "+GPT-2"); cat("+ GPT-2       ", round(a_gpt, 2), "\n")
  a_pros <- cv_accuracy(f_pros, d, "+prosody"); cat("+ prosody     ", round(a_pros, 2), "\n")
  a_both <- cv_accuracy(f_both, d, "+both"); cat("+ both        ", round(a_both, 2), "\n")

  # LRT on the full-data fits: prosody vs baseline (same test as slide 18)
  m_base <- fit_full(f_base, d, "baseline")
  m_pros <- fit_full(f_pros, d, "+prosody")
  lrt_p  <- anova(m_base, m_pros)[["Pr(>Chisq)"]][2]

  data.frame(
    task = task_name, variant = variant_name, n_rows = nrow(d),
    baseline = round(a_base, 2), with_gpt2 = round(a_gpt, 2),
    with_prosody = round(a_pros, 2), with_both = round(a_both, 2),
    gain_gpt2 = round(a_gpt - a_base, 2),
    gain_prosody = round(a_pros - a_base, 2),
    gain_both = round(a_both - a_base, 2),
    lrt_prosody_p = lrt_p
  )
}

res <- list()
res[[1]] <- run_one("filler", "accent", file.path(PROCESSED_DIR, "dataset_filler_accent.csv"),
                    c("prev_accent_strength", "foll_accent_strength"), "word_for_model")
res[[2]] <- run_one("filler", "break", file.path(PROCESSED_DIR, "dataset_filler_break.csv"),
                    c("prev_break_index", "foll_break_index"), "word_for_model")
res[[3]] <- run_one("reparandum", "accent", file.path(PROCESSED_DIR, "dataset_reparandum_accent.csv"),
                    c("prev_accent_strength", "foll_accent_strength"), "word")
res[[4]] <- run_one("reparandum", "break", file.path(PROCESSED_DIR, "dataset_reparandum_break.csv"),
                    c("prev_break_index", "foll_break_index"), "word")

out <- do.call(rbind, res)
write.csv(out, file.path(RESULTS_DIR, "table_prosody_vs_gpt2_same_rows.csv"), row.names = FALSE)
cat("\nsaved -> results/table_prosody_vs_gpt2_same_rows.csv\n")
print(out, row.names = FALSE)