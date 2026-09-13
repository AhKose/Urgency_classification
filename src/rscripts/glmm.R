#!/usr/bin/env Rscript
# Mixed-effects logistic regression for cross-validated model comparison.
#   correct ~ model + (1 | patient_id) + (1 | fold)
# `model` fixed effect => proposed (reference) vs each baseline, with SEs that
# account for (i) the same patient scored by every model and (ii) fold clustering.
suppressMessages(library(lme4))

args <- commandArgs(trailingOnly = TRUE)
infile <- args[1]; outfile <- args[2]; ref <- args[3]

d <- read.csv(infile, stringsAsFactors = FALSE)
d$model <- relevel(factor(d$model), ref = ref)
d$patient_id <- factor(d$patient_id)
d$fold <- factor(d$fold)

fit <- glmer(correct ~ model + (1 | patient_id) + (1 | fold),
             data = d, family = binomial,
             control = glmerControl(optimizer = "bobyqa",
                                    optCtrl = list(maxfun = 2e5)))

co <- summary(fit)$coefficients
ci <- tryCatch(confint(fit, parm = "beta_", method = "Wald"),
               error = function(e) matrix(NA, nrow(co), 2))

res <- data.frame(
  term      = rownames(co),
  estimate  = co[, "Estimate"],
  std_error = co[, "Std. Error"],
  z         = co[, "z value"],
  p_value   = co[, "Pr(>|z|)"],
  ci_low    = ci[match(rownames(co), rownames(ci)), 1],
  ci_high   = ci[match(rownames(co), rownames(ci)), 2],
  row.names = NULL
)
vc <- as.data.frame(VarCorr(fit))
attr(res, "note") <- ""
write.csv(res, outfile, row.names = FALSE)
write.csv(vc[, c("grp", "vcov")], sub("\\.csv$", "_varcomp.csv", outfile), row.names = FALSE)
cat("glmm ok:", nrow(d), "rows,", nlevels(d$model), "models\n")
