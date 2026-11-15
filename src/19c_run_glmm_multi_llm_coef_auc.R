## ---------------------------------------------------------------------
## 19c_run_glmm_multi_llm_coef_auc.R
##
## One script for both tasks, built from 19_run_glmm_multi_llm_FIXED_v2.R and
## 19b_run_glmm_filler_fixfoll.R. Same formulas, same 5-fold CV (seed 42),
## same LRT, so accuracies should match the existing table.
##
##   filler      -> data/processed/dataset_filler_fixfoll_multillm.csv  (corrected foll features)
##   reparandum  -> data/processed/dataset_reparandum_multillm.csv
##
## What is new compared to 19 / 19b:
##   1. fixed-effect coefficients for the baseline and enhanced model, with a
##      standardised version (estimate * sd of that column), so n-gram surprisal
##      and LM surprisal can be compared on the same scale
##   2. out-of-fold predictions saved per model/task (results/cv_preds/)
##   3. AUC (baseline, enhanced) and always-guess accuracy in the summary table
##   4. the baseline CV/fit is done once per task when the rows are the same,
##      like 19b does (it is the same model each time)
##
## out: results/table_multi_llm_comparison_v3.csv   (does NOT overwrite the final table)
##      results/table_multi_llm_coefficients.csv
##      results/table_ngram_vs_lm_weights.csv
##      results/cv_preds/<task>_<tag>.csv
##
## run from repo root:
##   "C:\Program Files\R\R-4.5.3\bin\Rscript.exe" src/19c_run_glmm_multi_llm_coef_auc.R
## ---------------------------------------------------------------------

.libPaths(c("C:/Rlibs", .libPaths()))
suppressMessages(library(lme4,     lib.loc = "C:/Rlibs"))
suppressMessages(library(lmerTest, lib.loc = "C:/Rlibs"))

BASE_DIR    <- getwd()
PROCESSED   <- file.path(BASE_DIR, "data", "processed")
RESULTS_DIR <- file.path(BASE_DIR, "results")
PRED_DIR    <- file.path(RESULTS_DIR, "cv_preds")
dir.create(RESULTS_DIR, showWarnings = FALSE)
dir.create(PRED_DIR, showWarnings = FALSE)

FILLER_CSV <- file.path(PROCESSED, "dataset_filler_fixfoll_multillm.csv")
REP_CSV    <- file.path(PROCESSED, "dataset_reparandum_multillm.csv")

OUT_TABLE  <- file.path(RESULTS_DIR, "table_multi_llm_comparison_v3.csv")
OUT_COEF   <- file.path(RESULTS_DIR, "table_multi_llm_coefficients.csv")
OUT_WEIGHT <- file.path(RESULTS_DIR, "table_ngram_vs_lm_weights.csv")

MODEL_TAGS <- c("gpt2", "dgpt2", "gpt2m", "pyt160", "opt125")
MODEL_LABELS <- c(
  gpt2   = "GPT-2 (124M)",
  dgpt2  = "DistilGPT-2 (82M)",
  gpt2m  = "GPT-2-medium (355M)",
  pyt160 = "Pythia-160m",
  opt125 = "OPT-125m"
)

## random-effect column differs by task (same as FIXED_v2 / 16_run_glmm_gpt2.R)
RANDOM_EFFECT_COL <- c(filler = "word_for_model", reparandum = "word")

BASE_TERMS <- paste(
  "prev_lex_surp + foll_lex_surp +",
  "prev_syn_surp + foll_syn_surp +",
  "prev_IC + foll_IC + prev_SC + foll_SC +",
  "prev_dur + foll_dur"
)
BASE_COLS <- c("prev_lex_surp", "foll_lex_surp", "prev_syn_surp", "foll_syn_surp",
               "prev_IC", "foll_IC", "prev_SC", "foll_SC", "prev_dur", "foll_dur")
CTRL <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

T_START <- Sys.time()
log_msg <- function(msg) {
  mins <- as.numeric(difftime(Sys.time(), T_START, units = "mins"))
  cat(sprintf("[%s | %5.1f min] %s\n", format(Sys.time(), "%H:%M:%S"), mins, msg))
}

## -----------------------------------------------------------------
## 5-fold CV. Same folds and threshold as before; now also returns the
## out-of-fold predicted probability for every row.
## -----------------------------------------------------------------
cv_run <- function(formula_str, data, k = 5) {
  set.seed(42)
  n     <- nrow(data)
  folds <- sample(rep(1:k, length.out = n))
  accs  <- numeric(k)
  oof   <- rep(NA_real_, n)
  for (f in 1:k) {
    train <- data[folds != f, ]
    test  <- data[folds == f, ]
    m <- tryCatch(
      glmer(as.formula(formula_str), data = train, family = binomial,
            control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))),
      error = function(e) NULL
    )
    if (is.null(m)) {
      accs[f] <- NA
    } else {
      preds <- predict(m, newdata = test, type = "response", allow.new.levels = TRUE)
      oof[folds == f] <- preds
      accs[f] <- mean((preds > 0.5) == (test$disfluency == 1))
    }
    log_msg(sprintf("    fold %d/%d done", f, k))
  }
  list(acc = mean(accs, na.rm = TRUE), oof = oof)
}

## AUC from ranks (probability that a random disfluent row scores above a random fluent one)
auc_rank <- function(y, p) {
  ok <- !is.na(p)
  y <- y[ok]; p <- p[ok]
  r  <- rank(p)
  n1 <- sum(y == 1)
  n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

## fixed-effect table; std_estimate = estimate * sd(column), so the weights are
## comparable even if the columns are not z-scored in the csv
coef_rows <- function(m, df_model, model_label, tag, task_name, which_model) {
  cf   <- summary(m)$coefficients
  term <- rownames(cf)
  sd_x <- sapply(term, function(t) if (t %in% names(df_model)) sd(df_model[[t]]) else NA_real_)
  sd_x <- unname(sd_x)
  data.frame(
    model = model_label, tag = tag, task = task_name, which = which_model,
    term = term,
    estimate = unname(cf[, "Estimate"]),
    se       = unname(cf[, "Std. Error"]),
    z        = unname(cf[, "z value"]),
    p        = unname(cf[, "Pr(>|z|)"]),
    sd_x     = sd_x,
    std_estimate = unname(cf[, "Estimate"]) * sd_x,
    std_se       = unname(cf[, "Std. Error"]) * sd_x,
    stringsAsFactors = FALSE
  )
}

store <- new.env()
store$results <- list()
store$coefs   <- list()
cache <- new.env()   # one baseline fit per task, reused when the rows are identical

write_all <- function() {
  if (length(store$results) == 0) return(invisible(NULL))
  write.csv(do.call(rbind, store$results), OUT_TABLE, row.names = FALSE)
  coefs <- do.call(rbind, store$coefs)
  write.csv(coefs, OUT_COEF, row.names = FALSE)

  ## n-gram vs LM weights: standardised coefficients in the enhanced model
  enh <- coefs[coefs$which == "enhanced", ]
  parts <- split(enh, list(enh$tag, enh$task), drop = TRUE)
  wt <- do.call(rbind, lapply(parts, function(d) {
    g <- function(t) { v <- d$std_estimate[d$term == t]; if (length(v) == 0) NA_real_ else v }
    tg <- d$tag[1]
    data.frame(model = d$model[1], tag = tg, task = d$task[1],
               ngram_prev = g("prev_lex_surp"), ngram_foll = g("foll_lex_surp"),
               lm_prev = g(paste0("prev_", tg, "_surp")),
               lm_foll = g(paste0("foll_", tg, "_surp")),
               stringsAsFactors = FALSE)
  }))
  write.csv(wt, OUT_WEIGHT, row.names = FALSE)
  invisible(NULL)
}

## -----------------------------------------------------------------
## One model, one task
## -----------------------------------------------------------------
compare_one_model <- function(df, tag, task_name, random_col) {
  prev_col <- paste0("prev_", tag, "_surp")
  foll_col <- paste0("foll_", tag, "_surp")
  lab <- MODEL_LABELS[[tag]]

  if (!(prev_col %in% names(df))) {
    log_msg(sprintf("[%s / %s] SKIPPED, column %s not found", tag, task_name, prev_col))
    return(NULL)
  }
  if (!(random_col %in% names(df))) {
    log_msg(sprintf("[%s / %s] SKIPPED, random effect column '%s' not found", tag, task_name, random_col))
    return(NULL)
  }

  df$.random_group <- as.character(df[[random_col]])
  need <- c(BASE_COLS, prev_col, foll_col, ".random_group", "disfluency")
  keep <- which(complete.cases(df[, need]))
  dm   <- df[keep, ]
  log_msg(sprintf("[%s / %s] %d complete rows (%d dropped)", tag, task_name, nrow(dm), nrow(df) - nrow(dm)))
  if (nrow(dm) < 100) {
    log_msg(sprintf("[%s / %s] SKIPPED, too few rows", tag, task_name))
    return(NULL)
  }

  base_f <- paste0("disfluency ~ ", BASE_TERMS, " + (1|.random_group)")
  enh_f  <- paste0("disfluency ~ ", BASE_TERMS, " + ", prev_col, " + ", foll_col, " + (1|.random_group)")

  ## baseline: reuse if same rows as the previous model for this task
  cb <- cache[[task_name]]
  if (!is.null(cb) && identical(cb$idx, keep)) {
    log_msg(sprintf("[%s / %s] same rows as before, reusing the baseline fit", tag, task_name))
  } else {
    log_msg(sprintf("[%s / %s] baseline CV ...", tag, task_name))
    cvb <- cv_run(base_f, dm)
    mb  <- glmer(as.formula(base_f), data = dm, family = binomial, control = CTRL)
    cb  <- list(idx = keep, acc = cvb$acc * 100, oof = cvb$oof, model = mb)
    cache[[task_name]] <- cb
  }
  log_msg(sprintf("[%s / %s] baseline CV accuracy: %.2f%%", tag, task_name, cb$acc))

  log_msg(sprintf("[%s / %s] enhanced CV ...", tag, task_name))
  cve     <- cv_run(enh_f, dm)
  acc_enh <- cve$acc * 100
  log_msg(sprintf("[%s / %s] enhanced CV accuracy: %.2f%%  (gain %+.2f)", tag, task_name, acc_enh, acc_enh - cb$acc))

  log_msg(sprintf("[%s / %s] full-data enhanced fit + LRT ...", tag, task_name))
  m_enh <- glmer(as.formula(enh_f), data = dm, family = binomial, control = CTRL)
  lrt_p <- anova(cb$model, m_enh)[["Pr(>Chisq)"]][2]
  log_msg(sprintf("[%s / %s] LRT p = %.3e", tag, task_name, lrt_p))

  ## coefficients
  store$coefs[[length(store$coefs) + 1]] <- coef_rows(cb$model, dm, lab, tag, task_name, "baseline")
  store$coefs[[length(store$coefs) + 1]] <- coef_rows(m_enh,    dm, lab, tag, task_name, "enhanced")

  ## predictions, AUC, always-guess
  y <- dm$disfluency
  write.csv(data.frame(row = keep, word = dm$.random_group, y = y,
                       p_base = cb$oof, p_enh = cve$oof),
            file.path(PRED_DIR, sprintf("%s_%s.csv", task_name, tag)), row.names = FALSE)
  auc_b <- auc_rank(y, cb$oof)
  auc_e <- auc_rank(y, cve$oof)
  always <- max(mean(y == 1), mean(y == 0)) * 100
  log_msg(sprintf("[%s / %s] AUC baseline %.4f, enhanced %.4f; always-guess %.2f%%",
                  tag, task_name, auc_b, auc_e, always))

  data.frame(
    model = lab, tag = tag, task = task_name, n_rows = nrow(dm),
    acc_baseline = round(cb$acc, 2), acc_enhanced = round(acc_enh, 2),
    acc_diff = round(acc_enh - cb$acc, 2),
    auc_baseline = round(auc_b, 4), auc_enhanced = round(auc_e, 4),
    auc_diff = round(auc_e - auc_b, 4),
    always_guess = round(always, 2),
    lrt_p = lrt_p, significant = lrt_p < 0.05
  )
}

## -----------------------------------------------------------------
## MAIN
## -----------------------------------------------------------------
cat(strrep("=", 60), "\n")
cat("Multi-LLM comparison v3: coefficients, AUC, always-guess\n")
cat(strrep("=", 60), "\n\n")

if (!file.exists(FILLER_CSV)) stop("dataset_filler_fixfoll_multillm.csv not found")
if (!file.exists(REP_CSV))    stop("dataset_reparandum_multillm.csv not found")

filler_df <- read.csv(FILLER_CSV)
rep_df    <- read.csv(REP_CSV)
log_msg(sprintf("Loaded %d filler rows, %d reparandum rows", nrow(filler_df), nrow(rep_df)))

for (tag in MODEL_TAGS) {
  cat("\n", strrep("-", 60), "\n")
  cat(sprintf("Model: %s (%s)\n", MODEL_LABELS[[tag]], tag))
  cat(strrep("-", 60), "\n")

  r <- compare_one_model(filler_df, tag, "filler", RANDOM_EFFECT_COL[["filler"]])
  if (!is.null(r)) { store$results[[length(store$results) + 1]] <- r; write_all() }

  r <- compare_one_model(rep_df, tag, "reparandum", RANDOM_EFFECT_COL[["reparandum"]])
  if (!is.null(r)) { store$results[[length(store$results) + 1]] <- r; write_all() }
}

if (length(store$results) == 0) stop("No models produced results")

cat("\n", strrep("=", 60), "\nDone\n", strrep("=", 60), "\n")
print(do.call(rbind, store$results), row.names = FALSE)
log_msg(sprintf("Saved -> %s", OUT_TABLE))
log_msg(sprintf("Saved -> %s", OUT_COEF))
log_msg(sprintf("Saved -> %s", OUT_WEIGHT))
log_msg("Total run time is in the left column of the log.")