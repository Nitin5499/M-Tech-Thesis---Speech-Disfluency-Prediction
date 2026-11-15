## 19b_run_glmm_filler_fixfoll.R
##
## Filler comparison for all five language models on the dataset whose baseline
## following-word features were corrected (data/processed/dataset_filler_fixfoll_multillm.csv,
## built by test_filler_foll_offset.py). Same CV code, formulas and LRT as
## 19_run_glmm_multi_llm_FIXED_v2.R. The only differences: it reads the corrected file, and the
## baseline model is fitted once when every model has the same rows (it is the same model each time).
##
## The GPT-2-only test gave baseline 78.56, enhanced 79.89, gain +1.33.
## out: results/table_filler_fixfoll_all_models.csv
##
## run from repo root:
##   "C:\Program Files\R\R-4.5.3\bin\Rscript.exe" src/19b_run_glmm_filler_fixfoll.R

.libPaths(c("C:/Rlibs", .libPaths()))
suppressMessages(library(lme4,     lib.loc = "C:/Rlibs"))
suppressMessages(library(lmerTest, lib.loc = "C:/Rlibs"))

BASE_DIR <- getwd()
IN_CSV   <- file.path(BASE_DIR, "data", "processed", "dataset_filler_fixfoll_multillm.csv")
OUT_CSV  <- file.path(BASE_DIR, "results", "table_filler_fixfoll_all_models.csv")
dir.create(file.path(BASE_DIR, "results"), showWarnings = FALSE)

MODEL_TAGS   <- c("gpt2", "dgpt2", "gpt2m", "pyt160", "opt125")
MODEL_LABELS <- c(gpt2 = "GPT-2 (124M)", dgpt2 = "DistilGPT-2 (82M)", gpt2m = "GPT-2-medium (355M)",
                  pyt160 = "Pythia-160m", opt125 = "OPT-125m")

log_msg <- function(msg) cat(sprintf("[%s] %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), msg))

cv_run <- function(formula_str, data, k = 5) {
  set.seed(42)
  n     <- nrow(data)
  folds <- sample(rep(1:k, length.out = n))
  accs  <- numeric(k)
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
      preds   <- predict(m, newdata = test, type = "response", allow.new.levels = TRUE)
      accs[f] <- mean((preds > 0.5) == (test$disfluency == 1))
    }
    log_msg(sprintf("    fold %d/%d done", f, k))
  }
  mean(accs, na.rm = TRUE)
}

BASE_TERMS <- paste(
  "prev_lex_surp + foll_lex_surp +",
  "prev_syn_surp + foll_syn_surp +",
  "prev_IC + foll_IC + prev_SC + foll_SC +",
  "prev_dur + foll_dur"
)
BASE_COLS <- c("prev_lex_surp", "foll_lex_surp", "prev_syn_surp", "foll_syn_surp",
               "prev_IC", "foll_IC", "prev_SC", "foll_SC", "prev_dur", "foll_dur")
CTRL <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

if (!file.exists(IN_CSV)) stop("run python src/test_filler_foll_offset.py first")
df <- read.csv(IN_CSV)
df$.random_group <- as.character(df$word_for_model)
log_msg(sprintf("Loaded %d filler rows", nrow(df)))

cache <- list(idx = NULL, acc = NA, model = NULL)
results <- list()

for (tag in MODEL_TAGS) {
  prev_col <- paste0("prev_", tag, "_surp")
  foll_col <- paste0("foll_", tag, "_surp")
  if (!(prev_col %in% names(df))) { log_msg(sprintf("[%s] skipped, column missing", tag)); next }

  need <- c(BASE_COLS, prev_col, foll_col, ".random_group", "disfluency")
  keep <- which(complete.cases(df[, need]))
  dm   <- df[keep, ]
  log_msg(sprintf("[%s] %d complete rows (%d dropped)", tag, nrow(dm), nrow(df) - nrow(dm)))

  base_f <- paste0("disfluency ~ ", BASE_TERMS, " + (1|.random_group)")
  enh_f  <- paste0("disfluency ~ ", BASE_TERMS, " + ", prev_col, " + ", foll_col, " + (1|.random_group)")

  if (!is.null(cache$idx) && identical(cache$idx, keep)) {
    log_msg(sprintf("[%s] same rows as before, reusing the baseline fit", tag))
  } else {
    log_msg(sprintf("[%s] baseline CV ...", tag))
    cache$acc   <- cv_run(base_f, dm) * 100
    cache$model <- glmer(as.formula(base_f), data = dm, family = binomial, control = CTRL)
    cache$idx   <- keep
  }
  acc_b <- cache$acc
  log_msg(sprintf("[%s] baseline CV accuracy: %.2f%%", tag, acc_b))

  log_msg(sprintf("[%s] enhanced CV ...", tag))
  acc_e <- cv_run(enh_f, dm) * 100
  log_msg(sprintf("[%s] enhanced CV accuracy: %.2f%%  (gain %+.2f)", tag, acc_e, acc_e - acc_b))

  log_msg(sprintf("[%s] likelihood-ratio test ...", tag))
  m_enh <- glmer(as.formula(enh_f), data = dm, family = binomial, control = CTRL)
  lrt_p <- anova(cache$model, m_enh)[["Pr(>Chisq)"]][2]
  log_msg(sprintf("[%s] LRT p = %.3e", tag, lrt_p))

  results[[length(results) + 1]] <- data.frame(
    model = MODEL_LABELS[[tag]], tag = tag, task = "filler", n_rows = nrow(dm),
    acc_baseline = round(acc_b, 2), acc_enhanced = round(acc_e, 2),
    acc_diff = round(acc_e - acc_b, 2), lrt_p = lrt_p, significant = lrt_p < 0.05)

  write.csv(do.call(rbind, results), OUT_CSV, row.names = FALSE)   # saved after every model
}

cat("\n", strrep("=", 60), "\nDone\n", strrep("=", 60), "\n")
print(do.call(rbind, results), row.names = FALSE)
log_msg(sprintf("Saved -> %s", OUT_CSV))