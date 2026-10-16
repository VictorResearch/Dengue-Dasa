# auto.arima (pacote forecast, Hyndman) por UF — chamado pelo Python: Rscript uf_autoarima.R <pasta>
#
# Tudo automático: para cada série, testa K = 1..Kmax pares de Fourier (sazonalidade anual, frequência 365,25/7)
# e, para cada K, o auto.arima escolhe d (teste KPSS), p, q e o drift pelo AICc (regressão com erros ARIMA).
# Fica o K de menor AICc (Hyndman & Athanasopoulos, FPP3, "dynamic harmonic regression": a forma recomendada
# para sazonalidade longa como a semanal, em vez de SARIMA com período 52).
# Exógenas opcionais (ARIMAX): colunas extras em dados.csv e futuro.csv.
#
# Entrada (na pasta): dados.csv (uid, y, [exógenas]), futuro.csv (uid, h, [exógenas]), params.txt (H, Kmax, threads)
# Saída: previsao.csv (uid, h, media, lo50, hi50, lo80, hi80, lo90, hi90, lo95, hi95, K, ordem, aicc)
args <- commandArgs(trailingOnly = TRUE)
pasta <- args[1]
suppressMessages(library(forecast))
par <- read.table(file.path(pasta, "params.txt"), sep = "=", stringsAsFactors = FALSE)
P <- setNames(as.character(par$V2), par$V1)
H <- as.integer(P[["H"]]); Kmax <- as.integer(P[["Kmax"]]); thr <- as.integer(P[["threads"]])
est <- as.integer(P[["estacionario"]]) == 1          # 1: d = 0 e sem drift (o resto continua automático)
d <- read.csv(file.path(pasta, "dados.csv"))
f <- read.csv(file.path(pasta, "futuro.csv"))
xcols <- setdiff(names(d), c("uid", "y"))
niv <- c(50, 80, 90, 95)

uma <- function(u) {
  y <- ts(d$y[d$uid == u], frequency = 365.25 / 7)
  Xp <- if (length(xcols)) as.matrix(d[d$uid == u, xcols, drop = FALSE]) else NULL
  Xf <- if (length(xcols)) as.matrix(f[f$uid == u, xcols, drop = FALSE]) else NULL
  melhor <- NULL
  for (K in 1:Kmax) {
    Fo <- fourier(y, K = K); Ff <- fourier(y, K = K, h = H)
    xr <- if (is.null(Xp)) Fo else cbind(Fo, Xp)
    xf <- if (is.null(Xf)) Ff else cbind(Ff, Xf)
    fit <- tryCatch(if (est) auto.arima(y, xreg = xr, seasonal = FALSE, d = 0, allowdrift = FALSE)
                    else auto.arima(y, xreg = xr, seasonal = FALSE), error = function(e) NULL)
    if (!is.null(fit) && (is.null(melhor) || fit$aicc < melhor$fit$aicc)) melhor <- list(fit = fit, xf = xf, K = K)
  }
  if (is.null(melhor)) stop(paste("auto.arima falhou para", u))
  fc <- forecast(melhor$fit, xreg = melhor$xf, h = H, level = niv)
  out <- data.frame(uid = u, h = 1:H, media = as.numeric(fc$mean))
  for (k in seq_along(niv)) {
    out[[paste0("lo", niv[k])]] <- as.numeric(fc$lower[, k])
    out[[paste0("hi", niv[k])]] <- as.numeric(fc$upper[, k])
  }
  out$K <- melhor$K
  out$ordem <- paste(arimaorder(melhor$fit), collapse = ",")
  out$aicc <- melhor$fit$aicc
  out
}
uids <- unique(d$uid)
res <- if (.Platform$OS.type == "windows" || thr <= 1) lapply(uids, uma) else parallel::mclapply(uids, uma, mc.cores = thr)
erros <- sapply(res, function(r) inherits(r, "try-error") || is.character(r))
if (any(erros)) stop(paste("falhou:", paste(uids[erros], collapse = ",")))
write.csv(do.call(rbind, res), file.path(pasta, "previsao.csv"), row.names = FALSE)
cat("auto.arima ok:", length(uids), "séries\n")
