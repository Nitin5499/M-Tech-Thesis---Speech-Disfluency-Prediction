# =============================================================
# 22_check_lex_sign.R
# =============================================================
# Quick check on the sign of n-gram (lexical) surprisal for fillers.
# In Dammalapati et al. (2021, Table 1) following lexical surprisal is
# positive for fillers (+0.74) and preceding is negative (-0.55).
# In my baseline fit (19c) they came out the other way round
# (prev +0.142, foll -0.552). This script looks at where that happens:
#   1. mean surprisal for disfluent vs fluent rows (raw)
#   2. each surprisal column alone (z-scored), random intercept per word
#   3. foll_lex with the duration columns, to see if duration flips it
#   4. prev_lex + foll_lex together
# Same for GPT-2 surprisal, so the two can be compared.
#
# Read only: nothing in data/ or results/ is changed.
# Prints to the console and writes results/check_lex_sign.csv
#
# run from repo root:
#   "C:\Program Files\R\R-4.5.3\bin\Rscript.exe" src\22_check_lex_sign.R
# =============================================================

.libPaths(c("C:/Rlibs", .libPaths()))
suppressMessages(library(lme4, lib.loc = "C:/Rlibs"))

PROCESSED <- file.path("data", "processed")
FILLER_CSV <- file.path(PROCESSED, "dataset_filler_fixfoll_multillm.csv")
REP_CSV    <- file.path(PROCESSED, "dataset_reparandum_multillm.csv")

ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
rows <- list()

check_task <- function(path, task, re_col) {
  df <- read.csv(path)
  cols <- c("prev_lex_surp", "foll_lex_surp", "prev_gpt2_surp", "foll_gpt2_surp",
            "prev_dur", "foll_dur")
  df <- df[complete.cases(df[, c(cols, "disfluency", re_col)]), ]
  df$g <- as.character(df[[re_col]])
  for (cc in cols) df[[paste0("z_", cc)]] <- as.numeric(scale(df[[cc]]))
  cat("\n", strrep("=", 60), "\n", task, " (", nrow(df), " rows)\n", strrep("=", 60), "\n", sep = "")

  # 1. raw means by class
  cat("\nMean of each column by class (0 = fluent, 1 = disfluent):\n")
  print(round(aggregate(df[, cols], by = list(disfluency = df$disfluency), FUN = mean), 3))

  # helper: fit and return the fixed effects
  fit <- function(rhs) {
    m <- glmer(as.formula(paste("disfluency ~", rhs, "+ (1|g)")), data = df,
               family = binomial, control = ctrl)
    fixef(m)[-1]
  }
  add <- function(label, est) {
    for (nm in names(est)) {
      rows[[length(rows) + 1]] <<- data.frame(task = task, model = label, term = nm,
                                              estimate = unname(est[nm]))
      cat(sprintf("   %-34s %-18s %+.3f\n", label, nm, est[nm]))
    }
  }

  cat("\nEach column alone (z-scored), random intercept per word:\n")
  for (cc in cols) add(paste("alone:", cc), fit(paste0("z_", cc)))

  cat("\nFollowing lexical surprisal with duration:\n")
  add("foll_lex + both durations", fit("z_foll_lex_surp + z_prev_dur + z_foll_dur"))

  cat("\nPrevious and following lexical surprisal together:\n")
  add("prev_lex + foll_lex", fit("z_prev_lex_surp + z_foll_lex_surp"))

  cat("\nSame two, GPT-2 surprisal:\n")
  add("prev_gpt2 + foll_gpt2", fit("z_prev_gpt2_surp + z_foll_gpt2_surp"))

  cat("\nCorrelations:\n")
  print(round(cor(df[, cols]), 2))
}

check_task(FILLER_CSV, "filler", "word_for_model")
check_task(REP_CSV, "reparandum", "word")

out <- do.call(rbind, rows)
write.csv(out, file.path("results", "check_lex_sign.csv"), row.names = FALSE)
cat("\nSaved -> results/check_lex_sign.csv\n")