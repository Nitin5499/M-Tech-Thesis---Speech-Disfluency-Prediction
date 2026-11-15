## ---------------------------------------------------------------------
## 19_run_glmm_multi_llm.R
##
## Multi-LLM surprisal extension -- Stage 3.
##
## Same comparison 16_run_glmm_gpt2.R did for GPT-2 alone (baseline
## 10-feature GLMM vs. baseline + that model's prev/foll surprisal),
## looped over every model tag found in dataset_filler_multillm.csv /
## dataset_reparandum_multillm.csv, so results are directly comparable
## across GPT-2, DistilGPT-2, GPT-2-medium, Pythia-160m and OPT-125m.
##
## FIX (Oct 2026): dataset_filler_v2.csv (the corrected, larger filler
## dataset -- see 18_merge_multi_llm.py's own Oct 2026 fix) has no "word"
## column, only "word_for_model". 16_run_glmm_gpt2.R already knew this --
## it grouped the filler task's random intercept by "word_for_model" and
## the reparandum task's by "word" (two different columns, deliberately).
## This script originally hardcoded "word" for both tasks via a single
## BASELINE_FORMULA constant, which (a) crashed outright once the merge
## was fixed to use dataset_filler_v2.csv (no "word" column to select),
## and (b) even before that, on the OLD dataset_filler.csv, "word" and
## "word_for_model" happened to hold identical values row-for-row in that
## older file, so the hardcoding silently worked there without masking
## any real difference -- the bug only became visible once the merge
## switched to the correct v2 dataset. Fixed by building the formula (and
## required_cols) per task, same as 16_run_glmm_gpt2.R's run_comparison().
##
## This tests WHETHER each model's surprisal adds explanatory power on
## top of the existing 10-feature baseline (which already includes
## ModelBlocks syntactic surprisal) -- same scope note as 16_run_glmm_gpt2.R.
##
## in:  data/processed/dataset_filler_multillm.csv
##      data/processed/dataset_reparandum_multillm.csv
## out: results/table_multi_llm_comparison.csv
## ---------------------------------------------------------------------

.libPaths(c("C:/Rlibs", .libPaths()))
suppressMessages(library(lme4,     lib.loc = "C:/Rlibs"))
suppressMessages(library(lmerTest, lib.loc = "C:/Rlibs"))

BASE_DIR    <- getwd()
PROCESSED   <- file.path(BASE_DIR, "data", "processed")
RESULTS_DIR <- file.path(BASE_DIR, "results")
dir.create(RESULTS_DIR, showWarnings = FALSE)

FILLER_CSV <- file.path(PROCESSED, "dataset_filler_multillm.csv")
REP_CSV    <- file.path(PROCESSED, "dataset_reparandum_multillm.csv")

## same tag list as 17_multi_llm_surprisal.py / 18_merge_multi_llm.py --
## kept in sync manually since this is a separate R script
MODEL_TAGS <- c("gpt2", "dgpt2", "gpt2m", "pyt160", "opt125")
MODEL_LABELS <- c(
  gpt2   = "GPT-2 (124M)",
  dgpt2  = "DistilGPT-2 (82M)",
  gpt2m  = "GPT-2-medium (355M)",
  pyt160 = "Pythia-160m",
  opt125 = "OPT-125m"
)

## FIX: random-effect grouping column differs by task, matching
## 16_run_glmm_gpt2.R exactly (filler -> word_for_model, reparandum -> word)
RANDOM_EFFECT_COL <- c(filler = "word_for_model", reparandum = "word")

log_msg <- function(msg) {
  cat(sprintf("[%s] %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), msg))
}

## -----------------------------------------------------------------
## Same 5-fold CV helper as 16_run_glmm_gpt2.R
## -----------------------------------------------------------------
cv_run <- function(formula_str, data, k = 5) {
  set.seed(42)
  n     <- nrow(data)
  folds <- sample(rep(1:k, length.out = n))
  accs  <- numeric(k)
  for (f in 1:k) {
    train <- data[folds != f, ]
    test  <- data[folds == f, ]
    m <- tryCatch(
      glmer(as.formula(formula_str), data = train,
            family = binomial,
            control = glmerControl(optimizer = "bobyqa",
                                    optCtrl = list(maxfun = 1e5))),
      error = function(e) NULL
    )
    if (is.null(m)) {
      accs[f] <- NA
    } else {
      preds   <- predict(m, newdata = test, type = "response",
                          allow.new.levels = TRUE)
      accs[f] <- mean((preds > 0.5) == (test$disfluency == 1))
    }
  }
  mean(accs, na.rm = TRUE)
}

BASE_TERMS <- paste(
  "prev_lex_surp + foll_lex_surp +",
  "prev_syn_surp + foll_syn_surp +",
  "prev_IC + foll_IC + prev_SC + foll_SC +",
  "prev_dur + foll_dur"
)

## -----------------------------------------------------------------
## Compare one model's surprisal against the baseline, for one task.
## random_col is the task-specific random-effect grouping column.
## -----------------------------------------------------------------
compare_one_model <- function(df, tag, task_name, random_col) {
  prev_col <- paste0("prev_", tag, "_surp")
  foll_col <- paste0("foll_", tag, "_surp")

  if (!(prev_col %in% names(df))) {
    log_msg(sprintf("  [%s / %s] SKIPPED -- column %s not found (run stages 1-2 for this model first)",
                     tag, task_name, prev_col))
    return(NULL)
  }
  if (!(random_col %in% names(df))) {
    log_msg(sprintf("  [%s / %s] SKIPPED -- random effect column '%s' not found in this dataset",
                     tag, task_name, random_col))
    return(NULL)
  }

  df$.random_group <- as.character(df[[random_col]])

  required_cols <- c("prev_lex_surp", "foll_lex_surp", "prev_syn_surp", "foll_syn_surp",
                      "prev_IC", "foll_IC", "prev_SC", "foll_SC", "prev_dur", "foll_dur",
                      prev_col, foll_col, ".random_group", "disfluency")
  df_model <- df[complete.cases(df[, required_cols]), ]

  log_msg(sprintf("  [%s / %s] Rows after completeness check: %d (%d dropped)",
                   tag, task_name, nrow(df_model), nrow(df) - nrow(df_model)))

  if (nrow(df_model) < 100) {
    log_msg(sprintf("  [%s / %s] SKIPPED -- too few complete rows (%d)", tag, task_name, nrow(df_model)))
    return(NULL)
  }

  baseline_formula <- paste0("disfluency ~ ", BASE_TERMS, " + (1|.random_group)")
  enhanced_formula  <- paste0(
    "disfluency ~ ", BASE_TERMS, " + ", prev_col, " + ", foll_col, " + (1|.random_group)"
  )

  log_msg(sprintf("  [%s / %s] Running 5-fold CV, baseline...", tag, task_name))
  acc_base <- cv_run(baseline_formula, df_model) * 100
  log_msg(sprintf("  [%s / %s] Baseline CV accuracy: %.2f%%", tag, task_name, acc_base))

  log_msg(sprintf("  [%s / %s] Running 5-fold CV, enhanced...", tag, task_name))
  acc_enh <- cv_run(enhanced_formula, df_model) * 100
  log_msg(sprintf("  [%s / %s] Enhanced CV accuracy: %.2f%%", tag, task_name, acc_enh))

  log_msg(sprintf("  [%s / %s] Fitting full-data models for LRT...", tag, task_name))
  m_base <- glmer(as.formula(baseline_formula), data = df_model, family = binomial,
                   control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
  m_enh  <- glmer(as.formula(enhanced_formula), data = df_model, family = binomial,
                   control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
  lrt <- anova(m_base, m_enh)
  lrt_p <- lrt[["Pr(>Chisq)"]][2]

  log_msg(sprintf("  [%s / %s] LRT p-value: %.3e  (%s)", tag, task_name, lrt_p,
                   ifelse(lrt_p < 0.05, "SIGNIFICANT", "not significant")))

  data.frame(
    model       = MODEL_LABELS[[tag]],
    tag         = tag,
    task        = task_name,
    n_rows      = nrow(df_model),
    acc_baseline = round(acc_base, 2),
    acc_enhanced = round(acc_enh, 2),
    acc_diff    = round(acc_enh - acc_base, 2),
    lrt_p       = lrt_p,
    significant = lrt_p < 0.05
  )
}

## -----------------------------------------------------------------
## MAIN
## -----------------------------------------------------------------
cat(strrep("=", 60), "\n")
cat("Multi-LLM Surprisal Extension -- Stage 3: GLMM Comparisons\n")
cat(strrep("=", 60), "\n\n")

if (!file.exists(FILLER_CSV) || !file.exists(REP_CSV)) {
  stop("dataset_filler_multillm.csv / dataset_reparandum_multillm.csv not found -- run 18_merge_multi_llm.py first")
}

filler_df <- read.csv(FILLER_CSV)
rep_df    <- read.csv(REP_CSV)
log_msg(sprintf("Loaded %d filler rows, %d reparandum rows", nrow(filler_df), nrow(rep_df)))

## optional: run only one task, e.g.  set ONLY_TASK=filler   (cmd)  before Rscript.
## Empty = both tasks. A single-task run writes to its own file so it cannot
## overwrite the full table.
ONLY_TASK <- Sys.getenv("ONLY_TASK", "")
all_results <- list()
i <- 1
for (tag in MODEL_TAGS) {
  cat("\n", strrep("-", 60), "\n")
  cat(sprintf("Model: %s (%s)\n", MODEL_LABELS[[tag]], tag))
  cat(strrep("-", 60), "\n")

  if (ONLY_TASK != "reparandum") {
    r_filler <- compare_one_model(filler_df, tag, "filler", RANDOM_EFFECT_COL[["filler"]])
    if (!is.null(r_filler)) { all_results[[i]] <- r_filler; i <- i + 1 }
  }

  if (ONLY_TASK != "filler") {
    r_rep <- compare_one_model(rep_df, tag, "reparandum", RANDOM_EFFECT_COL[["reparandum"]])
    if (!is.null(r_rep)) { all_results[[i]] <- r_rep; i <- i + 1 }
  }
}

if (length(all_results) == 0) {
  stop("No models produced results -- check that stages 1-2 ran for at least one model")
}

summary_table <- do.call(rbind, all_results)

cat("\n", strrep("=", 60), "\n")
cat("Stage 3 COMPLETE -- Summary Table\n")
cat(strrep("=", 60), "\n\n")
print(summary_table, row.names = FALSE)

out_name <- if (ONLY_TASK == "") "table_multi_llm_comparison.csv" else
  paste0("table_multi_llm_comparison_", ONLY_TASK, "_rerun.csv")
out_path <- file.path(RESULTS_DIR, out_name)
write.csv(summary_table, out_path, row.names = FALSE)
log_msg(sprintf("Saved summary table -> %s", out_path))

cat("\nNOTE: filler task's random intercept is grouped by word_for_model,\n")
cat("reparandum's by word -- matching 16_run_glmm_gpt2.R's original design.\n")
cat("acc_diff values are directly comparable across all models in this table.\n")