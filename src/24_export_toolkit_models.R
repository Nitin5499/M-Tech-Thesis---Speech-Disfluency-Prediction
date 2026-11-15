# =============================================================
# 24_export_toolkit_models.R
# =============================================================
# Fits the text-only models used by the toolkit (toolkit/disfluency_toolkit.py)
# and saves their coefficients, so the toolkit can score a typed sentence
# without running R.
#
# Only features that can be computed from typed text are used:
#   trigram surprisal, GPT-2 surprisal, DLT integration cost, DLT storage cost
#   (previous and following word each) = 8 features.
# Duration, prosody and ModelBlocks syntactic surprisal need audio / annotation
# / a long parser run, so they are not part of the toolkit model.
#
# Same data as the thesis results: dataset_filler_fixfoll_multillm.csv and
# dataset_reparandum_multillm.csv. Predictors are z-scored (mean/sd saved).
# The accuracy is 5-fold CV (seed 42, threshold 0.5), reported two ways:
#   fixed only  : re.form = NA            (a word the model has never seen)
#   with words  : re.form = NULL, new words get 0   (same way as the thesis GLMM,
#                 each known word keeps its own random intercept)
#
# run from the project folder:
#   "C:\Program Files\R\R-4.5.3\bin\Rscript.exe" src/24_export_toolkit_models.R
# out: toolkit/models/model_filler.csv, model_reparandum.csv, model_meta.csv
# =============================================================

library(lme4)

PROCESSED_DIR <- "data/processed"
OUT_DIR <- file.path("toolkit", "models")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

feats <- c("prev_lex_surp", "foll_lex_surp", "prev_gpt2_surp", "foll_gpt2_surp",
           "prev_IC", "foll_IC", "prev_SC", "foll_SC")

tasks <- list(
  filler     = list(file = "dataset_filler_fixfoll_multillm.csv", re = "word_for_model"),
  reparandum = list(file = "dataset_reparandum_multillm.csv",     re = "word")
)

ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
meta_rows <- list()

for (task in names(tasks)) {
  cat("\n====", task, "====\n")
  df <- read.csv(file.path(PROCESSED_DIR, tasks[[task]]$file), stringsAsFactors = FALSE)
  re_col <- tasks[[task]]$re
  df <- df[complete.cases(df[, c(feats, "disfluency", re_col)]), ]
  df$disfluency <- as.integer(df$disfluency)
  df$g <- as.character(df[[re_col]])
  cat("rows used:", nrow(df), "\n")
  cat("columns in this file:", paste(names(df), collapse = ", "), "\n")
  cat("random-effect column:", re_col, "| first rows:\n")
  print(head(df[, intersect(c("word", "word_for_model", "disfluency"), names(df))], 8))

  # z-score with the training mean / sd (saved, so the toolkit can do the same)
  mu <- sapply(feats, function(f) mean(df[[f]]))
  sdv <- sapply(feats, function(f) sd(df[[f]]))
  for (f in feats) df[[f]] <- (df[[f]] - mu[[f]]) / sdv[[f]]

  fml <- as.formula(paste("disfluency ~", paste(feats, collapse = " + "), "+ (1|g)"))

  # 5-fold CV, fixed effects only
  set.seed(42)
  folds <- sample(rep(1:5, length.out = nrow(df)))
  accs <- numeric(5)
  accs_w <- numeric(5)
  for (k in 1:5) {
    m <- glmer(fml, data = df[folds != k, ], family = binomial, control = ctrl)
    test <- df[folds == k, ]
    y <- (test$disfluency == 1)
    p  <- predict(m, newdata = test, type = "response", re.form = NA)
    pw <- predict(m, newdata = test, type = "response", re.form = NULL, allow.new.levels = TRUE)
    accs[k]   <- mean((p > 0.5) == y)
    accs_w[k] <- mean((pw > 0.5) == y)
    cat("  fold", k, "fixed only", round(accs[k] * 100, 2), "| with word effects", round(accs_w[k] * 100, 2), "\n")
  }
  cv_acc <- mean(accs) * 100
  cv_acc_w <- mean(accs_w) * 100
  always <- max(mean(df$disfluency), 1 - mean(df$disfluency)) * 100
  cat("CV accuracy fixed only:", round(cv_acc, 2), "| with word effects:", round(cv_acc_w, 2),
      "| always-guess:", round(always, 2), "\n")

  # full-data fit -> coefficients
  m_full <- glmer(fml, data = df, family = binomial, control = ctrl)
  fe <- fixef(m_full)
  out <- data.frame(term = names(fe), estimate = unname(fe), mean = NA_real_, sd = NA_real_)
  out$mean[match(feats, out$term)] <- unname(mu[feats])
  out$sd[match(feats, out$term)]   <- unname(sdv[feats])
  write.csv(out, file.path(OUT_DIR, paste0("model_", task, ".csv")), row.names = FALSE)
  print(out)

  # random intercept of every word seen in training (used for known words)
  re_tab <- ranef(m_full)$g
  we <- data.frame(word = rownames(re_tab), effect = re_tab[, 1])
  write.csv(we, file.path(OUT_DIR, paste0("wordeffects_", task, ".csv")), row.names = FALSE)
  cat("word effects saved:", nrow(we), "words\n")

  meta_rows[[task]] <- data.frame(task = task, n_rows = nrow(df),
                                  cv_accuracy = round(cv_acc, 2),
                                  always_guess = round(always, 2),
                                  cv_accuracy_word = round(cv_acc_w, 2))
}

write.csv(do.call(rbind, meta_rows), file.path(OUT_DIR, "model_meta.csv"), row.names = FALSE)
cat("\nsaved models ->", OUT_DIR, "\n")