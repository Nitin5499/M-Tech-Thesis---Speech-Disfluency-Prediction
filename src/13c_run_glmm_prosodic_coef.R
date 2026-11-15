# =============================================================
# 13c_run_glmm_prosodic_coef.R
# =============================================================
# Coefficients of the enhanced (baseline + prosody) model for the four
# prosody datasets. Same data prep and same formula as 13_run_glmm_prosodic.R,
# but here I only fit the full-data model once and save the coefficient table.
# All predictors are z-scored, so each estimate is per 1 SD of the predictor.
#
# run from the project folder:  Rscript src/13c_run_glmm_prosodic_coef.R
# =============================================================

library(lme4)

PROCESSED_DIR <- "data/processed"
RESULTS_DIR   <- "results"
if (!dir.exists(RESULTS_DIR)) dir.create(RESULTS_DIR)

feature_cols <- c(
  "prev_lex_surp", "foll_lex_surp", "prev_syn_surp", "foll_syn_surp",
  "prev_IC", "foll_IC", "prev_SC", "foll_SC", "prev_dur", "foll_dur"
)

# z-score the given columns (same as in script 13)
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

get_coefs <- function(task_name, variant_name, data_path, prosodic_cols, random_effect_col) {
  cat("\n---", task_name, "/", variant_name, "---\n")
  df <- read.csv(data_path, stringsAsFactors = FALSE)
  df <- zscore(df, feature_cols)
  df <- zscore(df, prosodic_cols)
  df$word <- as.character(df[[random_effect_col]])
  df$disfluency <- as.integer(df$disfluency)

  needed <- c(feature_cols, prosodic_cols, "disfluency", "word")
  needed <- needed[needed %in% colnames(df)]
  d <- df[complete.cases(df[, needed]), ]
  cat("rows used:", nrow(d), "\n")   # should match the rows on slide 18

  terms <- paste(feature_cols[feature_cols %in% colnames(d)], collapse = " + ")
  f <- paste0("disfluency ~ ", terms, " + ",
              paste(prosodic_cols, collapse = " + "), " + (1|word)")

  m <- glmer(as.formula(f), data = d, family = binomial,
             control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))

  cs <- coef(summary(m))
  out <- data.frame(
    task = task_name, variant = variant_name, n_rows = nrow(d),
    term = rownames(cs),
    estimate = round(cs[, "Estimate"], 3),
    se = round(cs[, "Std. Error"], 3),
    z = round(cs[, "z value"], 2),
    p = cs[, "Pr(>|z|)"],
    row.names = NULL
  )
  print(out[out$term %in% prosodic_cols, ], row.names = FALSE)  # just the new terms
  out
}

res <- list()
res[[1]] <- get_coefs("filler", "accent", file.path(PROCESSED_DIR, "dataset_filler_accent.csv"),
                      c("prev_accent_strength", "foll_accent_strength"), "word_for_model")
res[[2]] <- get_coefs("filler", "break", file.path(PROCESSED_DIR, "dataset_filler_break.csv"),
                      c("prev_break_index", "foll_break_index"), "word_for_model")
res[[3]] <- get_coefs("reparandum", "accent", file.path(PROCESSED_DIR, "dataset_reparandum_accent.csv"),
                      c("prev_accent_strength", "foll_accent_strength"), "word")
res[[4]] <- get_coefs("reparandum", "break", file.path(PROCESSED_DIR, "dataset_reparandum_break.csv"),
                      c("prev_break_index", "foll_break_index"), "word")

all_coefs <- do.call(rbind, res)
out_path <- file.path(RESULTS_DIR, "table_prosodic_coefficients.csv")
write.csv(all_coefs, out_path, row.names = FALSE)
cat("\nsaved ->", out_path, "\n")