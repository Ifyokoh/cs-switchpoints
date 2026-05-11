# Mixed-effects logistic regression for code-switch point analysis.
#
# Model: switch ~ relation + (1 | sent_id)
#   - switch     : binary outcome — did a code-switch occur between
#                  this adjacent token pair?
#   - relation   : within | boundary — syntactic position of the pair
#   - (1|sent_id): random intercept per sentence — pairs are nested
#                  inside sentences and share tokens, so they are not
#                  independent. This lets each sentence have its own
#                  baseline switching rate.
#
# Outputs (written to outputs/):
#   glmm_odds_ratio_plot.svg     — forest plot, one row per dataset
#   glmm_predicted_prob_plot.svg — predicted P(switch) for hi_en
#   stats_summary.txt            — numerical results

suppressPackageStartupMessages({
  library(lme4)
  library(ggplot2)
  library(broom.mixed)
})


# Paths 

args     <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("--file=", args, value = TRUE)
if (length(file_arg) > 0) {
  script_dir <- dirname(normalizePath(sub("--file=", "", file_arg)))
  repo_root  <- normalizePath(file.path(script_dir, ".."))
} else {
  repo_root <- normalizePath(".")
}

switchpairs_dir <- file.path(repo_root, "data", "switchpairs")
outputs_dir     <- file.path(repo_root, "outputs")
dir.create(outputs_dir, showWarnings = FALSE, recursive = TRUE)

DATASETS <- list(
  hi_en = list(file = "hi_en_pairs.tsv", label = "Hindi–English"),
  te_en = list(file = "te_en_pairs.tsv", label = "Telugu–English"),
  tr_en = list(file = "tr_en_pairs.tsv", label = "Turkish–English")
)


# Data loading

load_pairs <- function(filepath) {
  df <- read.table(
    filepath,
    header           = TRUE,
    sep              = "\t",
    stringsAsFactors = FALSE,
    quote            = "",
    comment.char     = ""
  )

  # Drop pairs where switch or relation could not be determined.
  df <- df[df$switch != "unknown" & df$relation != "unknown", ]

  # Binary outcome: 1 = code-switch, 0 = same language.
  df$switch_bin <- as.integer(df$switch == "true")

  # Relation factor with "within" as reference so the coefficient
  # directly reads as the boundary-vs-within contrast.
  df$relation <- factor(df$relation, levels = c("within", "boundary"))

  df
}


# Model fitting

fit_model <- function(df, label) {
  cat(sprintf(
    "\nFitting model for %s (%d pairs, %d sentences)...\n",
    label, nrow(df), length(unique(df$sent_id))
  ))

  ctrl  <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
  model <- glmer(
    switch_bin ~ relation + (1 | sent_id),
    data    = df,
    family  = binomial,
    control = ctrl
  )

  if (isSingular(model))
    cat(sprintf("  WARNING: singular fit (%s) — treat as exploratory\n", label))

  list(label = label, df = df, model = model)
}


# Helper: extract OR for the relation term

extract_relation_or <- function(model, dataset_label) {
  tidied <- broom.mixed::tidy(
    model,
    effects      = "fixed",
    conf.int     = TRUE,
    exponentiate = TRUE
  )
  row <- tidied[tidied$term == "relationboundary", ]
  if (nrow(row) == 0) return(NULL)
  data.frame(
    dataset  = dataset_label,
    or       = row$estimate,
    ci_low   = row$conf.low,
    ci_high  = row$conf.high,
    z        = row$statistic,
    p_value  = row$p.value,
    stringsAsFactors = FALSE
  )
}


# Helper: predicted probability with 95% CI

predict_with_ci <- function(model, newdata, alpha = 0.05) {
  mm      <- model.matrix(delete.response(terms(model)), data = newdata)
  fe      <- fixef(model)
  vc      <- as.matrix(vcov(model))
  lp      <- as.vector(mm %*% fe)
  se_lp   <- sqrt(diag(mm %*% vc %*% t(mm)))
  z_crit  <- qnorm(1 - alpha / 2)
  logistic <- function(x) 1 / (1 + exp(-x))
  data.frame(
    newdata,
    prob    = logistic(lp),
    ci_low  = logistic(lp - z_crit * se_lp),
    ci_high = logistic(lp + z_crit * se_lp),
    stringsAsFactors = FALSE
  )
}


# Load and fit 

cat("Loading datasets...\n")
datasets <- lapply(names(DATASETS), function(prefix)
  load_pairs(file.path(switchpairs_dir, DATASETS[[prefix]]$file)))
names(datasets) <- names(DATASETS)

results <- lapply(names(DATASETS), function(prefix)
  fit_model(datasets[[prefix]], DATASETS[[prefix]]$label))
names(results) <- names(DATASETS)


# Plot 1: forest plot — OR per dataset
# Each row is one language pair. The dot is the odds ratio for
# boundary vs within; the bar is the 95% confidence interval.
# A dashed line at OR = 1 marks "no effect." If the bar does not
# cross that line, the effect is statistically significant.

cat("\nBuilding glmm_odds_ratio_plot.svg...\n")

forest_data <- do.call(rbind, lapply(names(results), function(prefix) {
  res <- results[[prefix]]
  extract_relation_or(res$model, res$label)
}))

forest_data$dataset <- factor(
  forest_data$dataset,
  levels = forest_data$dataset[order(forest_data$or)]
)

forest_plot <- ggplot(
  forest_data,
  aes(x = or, y = dataset)
) +
  geom_vline(
    xintercept = 1, linetype = "dashed",
    colour = "grey50", linewidth = 0.6
  ) +
  geom_point(size = 3, colour = "steelblue") +
  geom_errorbarh(
    aes(xmin = ci_low, xmax = ci_high),
    height = 0.18, linewidth = 0.7, colour = "steelblue"
  ) +
  scale_x_log10(breaks = c(0.1, 0.3, 0.5, 1, 2, 3, 5)) +
  labs(
    x        = "Odds ratio: boundary vs within (log scale)",
    y        = NULL,
    title    = "Effect of syntactic position on code-switching",
    subtitle = "Dot = odds ratio, bar = 95% CI. Dashed line at OR = 1 means no effect."
  ) +
  theme_bw(base_size = 12)

ggsave(
  file.path(outputs_dir, "glmm_odds_ratio_plot.svg"),
  forest_plot, width = 7, height = 3.5
)
cat("  Saved glmm_odds_ratio_plot.svg\n")


# Plot 2: predicted P(switch) by relation, hi_en
# Shows the model's estimated probability of a switch for within-
# constituent pairs vs boundary pairs, for Hindi-English.

cat("Building glmm_predicted_prob_plot.svg...\n")

newdata <- data.frame(
  relation = factor(c("within", "boundary"), levels = c("within", "boundary"))
)
predicted <- predict_with_ci(results$hi_en$model, newdata)
predicted$label <- factor(c("Within", "Boundary"), levels = c("Within", "Boundary"))

predicted_prob_plot <- ggplot(predicted, aes(x = label, y = prob, fill = label)) +
  geom_col(width = 0.5, colour = "white") +
  geom_errorbar(
    aes(ymin = ci_low, ymax = ci_high),
    width = 0.12, linewidth = 0.7
  ) +
  scale_y_continuous(
    limits = c(0, 0.65),
    labels = function(x) paste0(round(x * 100), "%")
  ) +
  scale_fill_manual(values = c("steelblue", "coral")) +
  labs(
    x        = "Syntactic position",
    y        = "Predicted P(switch)",
    title    = "Predicted probability of code-switching by syntactic position",
    subtitle = "Hindi–English. Population-level prediction, 95% CI (delta method)."
  ) +
  theme_bw(base_size = 12) +
  theme(legend.position = "none")

ggsave(
  file.path(outputs_dir, "glmm_predicted_prob_plot.svg"),
  predicted_prob_plot, width = 5, height = 5
)
cat("  Saved glmm_predicted_prob_plot.svg\n")


# Write stats_summary.txt

cat("Writing stats_summary.txt...\n")

lines <- c(
  "Mixed-Effects Logistic Regression Results",
  strrep("=", 60),
  "",
  "Model: switch ~ relation + (1 | sent_id)",
  "",
  "How to read the table:",
  "  Odds Ratio : how much more likely a switch is at a boundary",
  "               than within a constituent.",
  "               > 1  boundary pairs switch more often",
  "               = 1  no difference",
  "               < 1  boundary pairs switch less often",
  "  95% CI     : confidence interval. If it does not include 1,",
  "               the result is significant at p < 0.05.",
  "  z-value    : test statistic. Larger absolute value = stronger",
  "               evidence against the null (no effect).",
  "  p-value    : probability of this result by chance if there is",
  "               truly no effect. < 0.05 = conventionally significant.",
  "  Singular   : YES means the sentence random effect collapsed to",
  "               zero. The model ran but results are unreliable;",
  "               treat as exploratory only.",
  strrep("-", 60)
)

for (prefix in names(results)) {
  res      <- results[[prefix]]
  singular <- isSingular(res$model)
  tidied   <- broom.mixed::tidy(
    res$model,
    effects      = "fixed",
    conf.int     = TRUE,
    exponentiate = TRUE
  )

  lines <- c(lines, sprintf("\n=== %s ===", res$label))
  lines <- c(lines, sprintf(
    "  Pairs: %d   Sentences: %d   Singular: %s",
    nrow(res$df),
    length(unique(res$df$sent_id)),
    ifelse(singular, "YES — treat with caution", "No")
  ))
  lines <- c(lines, sprintf(
    "  %-30s  %10s  %14s  %8s  %s",
    "Predictor", "Odds Ratio", "95% CI", "z-value", "p-value"
  ))
  lines <- c(lines, sprintf("  %s", strrep("-", 72)))

  for (i in seq_len(nrow(tidied))) {
    row          <- tidied[i, ]
    term_display <- row$term
    term_display <- gsub("^\\(Intercept\\)$", "Intercept (baseline log-odds)", term_display)
    term_display <- gsub("^relationboundary$", "relation = boundary (vs within)", term_display)
    ci_str       <- sprintf("[%.3f, %.3f]", row$conf.low, row$conf.high)
    lines <- c(lines, sprintf(
      "  %-30s  %10.3f  %14s  %8.2f  %.4g",
      term_display, row$estimate, ci_str, row$statistic, row$p.value
    ))
  }
}

writeLines(lines, file.path(outputs_dir, "stats_summary.txt"))
cat("  Saved stats_summary.txt\n")
cat("\nDone.\n")
