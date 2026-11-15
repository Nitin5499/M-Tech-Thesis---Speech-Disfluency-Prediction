import os

content = '''# 06_run_glmm.R
cat("\\n=== GLMM Modeling (lme4) ===\\n")
.libPaths(c("C:/Rlibs", .libPaths()))
suppressPackageStartupMessages({
  library(lme4,     lib.loc="C:/Rlibs")
  library(lmerTest, lib.loc="C:/Rlibs")
})
cat("lme4 and lmerTest loaded\\n")

root_dir  <- getwd()
processed <- file.path(root_dir, "data", "processed")
results   <- file.path(root_dir, "results")
dir.create(results, showWarnings=FALSE)

filler_path <- file.path(processed, "dataset_filler.csv")
rep_path    <- file.path(processed, "dataset_reparandum.csv")

FILLER_TARGET <- 24100
REP_TARGET    <- 29040

feature_cols <- c("prev_lex_surp","foll_lex_surp","prev_syn_surp","foll_syn_surp",
                  "prev_IC","foll_IC","prev_SC","foll_SC","prev_dur","foll_dur")

content_pos <- c("NN","NNS","NNP","NNPS","VB","VBD","VBG","VBN","VBP","VBZ",
                 "JJ","JJR","JJS","RB","RBR","RBS")

FULL_FORMULA <- "disfluency ~ prev_lex_surp + foll_lex_surp + prev_syn_surp + foll_syn_surp + prev_IC + foll_IC + prev_SC + foll_SC + prev_dur + foll_dur + (1|word)"
FIXED_ONLY   <- "disfluency ~ prev_lex_surp + foll_lex_surp + prev_syn_surp + foll_syn_surp + prev_IC + foll_IC + prev_SC + foll_SC + prev_dur + foll_dur"

cat("Full formula:\\n ", FULL_FORMULA, "\\n")

standardise <- function(df) {
  for (col in feature_cols) {
    if (col %in% colnames(df)) {
      s <- sd(df[[col]], na.rm=TRUE)
      if (!is.na(s) && s > 0)
        df[[col]] <- (df[[col]] - mean(df[[col]], na.rm=TRUE)) / s
    }
  }
  df$word       <- as.character(df$word_for_model)
  df$disfluency <- as.integer(df$disfluency)
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
            control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=1e5))),
      error=function(e) { cat("  fold", f, "error:", conditionMessage(e), "\\n"); NULL }
    )
    if (is.null(m)) { accs[f] <- NA } else {
      preds   <- predict(m, newdata=test, type="response", allow.new.levels=TRUE)
      accs[f] <- mean((preds > 0.5) == (test$disfluency == 1))
    }
  }
  mean(accs, na.rm=TRUE) * 100
}

run_task <- function(csv_path, task_name) {
  cat("\\n", strrep("=", 55), "\\n")
  cat("Task:", task_name, "\\n")
  cat(strrep("=", 55), "\\n")
  df          <- read.csv(csv_path)
  avail_feats <- feature_cols[feature_cols %in% colnames(df)]
  df          <- df[complete.cases(df[, avail_feats]), ]
  df          <- standardise(df)
  cat("Rows before subsampling:", nrow(df), "\\n")
  target <- if (task_name == "filler") FILLER_TARGET else REP_TARGET
  df     <- subsample(df, target)
  cat("Rows after subsampling: ", nrow(df), "\\n")
  cat("\\nBaseline (random intercept only)...\\n")
  base_acc <- cv_accuracy("disfluency ~ 1 + (1|word)", df)
  cat("  Baseline:", round(base_acc, 2), "%")
  cat("  (paper:", if (task_name=="filler") "66.80%)\\n" else "66.29%)\\n")
  cat("\\nFitting full model...\\n")
  full_model <- tryCatch(
    glmer(as.formula(FULL_FORMULA), data=df, family=binomial,
          control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5))),
    error=function(e) { cat("ERROR:", conditionMessage(e), "\\n"); NULL }
  )
  if (is.null(full_model)) { cat("Full model failed\\n"); return(NULL) }
  coef_df         <- as.data.frame(summary(full_model)$coefficients)
  coef_df$feature <- rownames(coef_df)
  rownames(coef_df) <- NULL
  cat("\\nCoefficients:\\n")
  print(coef_df[, c("feature","Estimate","Std. Error","z value","Pr(>|z|)")])
  cat("\\nRunning 5-fold CV...\\n")
  full_acc <- cv_accuracy(FULL_FORMULA, df)
  cat("  Full model:", round(full_acc, 2), "%")
  cat("  (paper:", if (task_name=="filler") "81.21%)\\n" else "71.11%)\\n")
  write.csv(coef_df, file.path(results, paste0("table1_", task_name, ".csv")), row.names=FALSE)
  writeLines(
    c(paste("Baseline:      ", round(base_acc, 2), "%"),
      paste("Full model:    ", round(full_acc,  2), "%"),
      if (task_name == "filler") c("Paper baseline: 66.80%", "Paper full: 81.21%")
      else c("Paper baseline: 66.29%", "Paper full: 71.11%")),
    file.path(results, paste0("accuracy_", task_name, ".txt"))
  )
  list(df=df, base_acc=base_acc, full_acc=full_acc, model=full_model)
}

re_comparison <- function(res_obj, task_name) {
  cat("\\nRandom effects comparison:", task_name, "\\n")
  df <- res_obj$df
  configs <- list(
    list(name="None",       f=FIXED_ONLY),
    list(name="Coarse POS", f=paste0(FIXED_ONLY, " + (1|coarse_pos)")),
    list(name="Fine POS",   f=paste0(FIXED_ONLY, " + (1|fine_pos)")),
    list(name="Raw word",   f=paste0(FIXED_ONLY, " + (1|word)"))
  )
  out <- data.frame(random_effect=character(), accuracy=numeric(), stringsAsFactors=FALSE)
  for (cfg in configs) {
    cat(" ", cfg$name, "...", sep="")
    acc <- tryCatch(cv_accuracy(cfg$f, df), error=function(e) { cat(" ERROR\\n"); NA })
    cat(" ", round(acc, 2), "%\\n")
    out <- rbind(out, data.frame(random_effect=cfg$name, accuracy=round(acc,2), stringsAsFactors=FALSE))
  }
  colnames(out)[2] <- task_name
  out
}

filler_res <- run_task(filler_path, "filler")
rep_res    <- run_task(rep_path,    "reparandum")

cat("\\n", strrep("=", 55), "\\n")
cat("Table 2: Random effects\\n")
cat(strrep("=", 55), "\\n")
cat("Paper: None=75.41/65.09  CoarsePOS=76.99/65.23  FinePOS=79.01/67.70  RawWord=81.21/71.11\\n\\n")
re_f   <- re_comparison(filler_res,    "Fillers")
re_r   <- re_comparison(rep_res, "Reparandums")
table2 <- merge(re_f, re_r, by="random_effect")
cat("\\nOur Table 2:\\n")
print(table2)
write.csv(table2, file.path(results, "table2_random_effects.csv"), row.names=FALSE)

cat("\\n", strrep("=", 55), "\\n")
cat("COMPLETE\\n")
cat(strrep("=", 55), "\\n")
cat("Filler accuracy:    ", round(filler_res$full_acc, 2), "% (paper: 81.21%)\\n")
cat("Reparandum accuracy:", round(rep_res$full_acc,    2), "% (paper: 71.11%)\\n")
cat("\\nResults saved in results/\\n")
'''

with open(os.path.join("src", "06_run_glmm.R"), "w", encoding="utf-8") as f:
    f.write(content)

print("Successfully wrote src/06_run_glmm.R")
print(f"File size: {os.path.getsize(os.path.join('src', '06_run_glmm.R'))} bytes")
print("\nNow run:")
print('  "C:\\\\Program Files\\\\R\\\\R-4.5.3\\\\bin\\\\Rscript.exe" src/06_run_glmm.R')
