# Competição · modelos bayesianos em R-INLA (chamado pelo Python: Rscript comp_inla.R <pasta>)
#
#  bb  = modelo-base oficial do 3º IMDC (PROCC, Freitas et al. 2025), macro = 5 grandes regiões do país:
#        NB; efeito aleatório de semana (RW2 cíclico) e de ano, ambos por macrorregião; intercepto por região.
#  ghr = modelo do BSC/GHR (Lowe): NB; BYM2 no grafo de regiões; RW2 cíclico de semana por UF;
#        efeito de ano por UF; anomalias de temperatura e chuva defasadas (SEAS5 depois do corte).
#
# Otimizações:
#  - ajuste só com as linhas observadas (as semanas a prever não entram no ajuste);
#  - inla.mode = "compact" (o mais rápido), estratégia "gaussian", int.strategy "eb";
#    se o compact falhar, repete em "classic";
#  - amostras: sorteia só os efeitos (hiperparâmetros + componentes latentes) e monta o preditor das semanas
#    futuras aqui, em vez de amostrar o preditor inteiro.
#
# Entrada (na pasta): dados.csv (y = NA nas semanas a prever), grafo.adj, params.txt
# Saída: amostras.bin (int32 n, int32 M, depois float32 M × n), hiper.csv, tempo.txt
args <- commandArgs(trailingOnly = TRUE)
pasta <- args[1]
suppressMessages(library(INLA))
par <- read.table(file.path(pasta, "params.txt"), sep = "=", stringsAsFactors = FALSE)
P <- setNames(as.character(par$V2), par$V1)
M <- as.integer(P[["n_amostras"]])
thr <- as.integer(P[["threads"]])
modelo <- P[["modelo"]]
inla.setOption(num.threads = paste0(thr, ":1"))

d <- read.csv(file.path(pasta, "dados.csv"))
treino <- d[!is.na(d$y), ]
fut <- d[is.na(d$y), ]
pc <- list(prec = list(prior = "pc.prec", param = c(1, 0.01)))
n_woy <- 52

if (modelo == "bb") {
  f <- y ~ 1 + offset(log_E) +
    f(reg_id, model = "iid", hyper = pc, values = sort(unique(d$reg_id))) +
    f(woy, model = "rw2", cyclic = TRUE, scale.model = TRUE, replicate = macro_id, values = 1:n_woy, hyper = pc) +
    f(ano_macro, model = "iid", hyper = pc, values = sort(unique(d$ano_macro)))
} else if (modelo == "ghr_nl") {
  g <- inla.read.graph(file.path(pasta, "grafo.adj"))
  f <- y ~ 1 + offset(log_E) + enso +
    f(tbin, model = "rw1", values = 1:10, scale.model = TRUE, hyper = pc) +
    f(pbin, model = "rw1", values = 1:10, scale.model = TRUE, hyper = pc) +
    f(reg_id, model = "bym2", graph = g, scale.model = TRUE, constr = TRUE,
      hyper = list(phi = list(prior = "pc", param = c(0.5, 2 / 3)),
                   prec = list(prior = "pc.prec", param = c(1, 0.01)))) +
    f(woy, model = "rw2", cyclic = TRUE, scale.model = TRUE, replicate = uf_id, values = 1:n_woy, hyper = pc) +
    f(ano_uf, model = "iid", hyper = pc, values = sort(unique(d$ano_uf)))
} else {
  g <- inla.read.graph(file.path(pasta, "grafo.adj"))
  f <- y ~ 1 + offset(log_E) + tanom + panom +
    f(reg_id, model = "bym2", graph = g, scale.model = TRUE, constr = TRUE,
      hyper = list(phi = list(prior = "pc", param = c(0.5, 2 / 3)),
                   prec = list(prior = "pc.prec", param = c(1, 0.01)))) +
    f(woy, model = "rw2", cyclic = TRUE, scale.model = TRUE, replicate = uf_id, values = 1:n_woy, hyper = pc) +
    f(ano_uf, model = "iid", hyper = pc, values = sort(unique(d$ano_uf)))
}

ajusta <- function(modo) {
  inla(f, family = "nbinomial", data = treino,
       control.predictor = list(compute = FALSE),
       control.compute = list(config = TRUE, dic = FALSE, waic = FALSE, cpo = FALSE),
       control.inla = list(strategy = "gaussian", int.strategy = "eb"),
       inla.mode = modo, verbose = FALSE)
}
t0 <- Sys.time()
modo <- "compact"
r <- tryCatch(ajusta("compact"), error = function(e) NULL)
if (is.null(r)) { modo <- "classic"; cat("compact falhou; usando classic\n"); r <- ajusta("classic") }
t_ajuste <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat("ajuste INLA (", modo, "):", round(t_ajuste), "s\n")

# ---------------------------------------------------------------- amostras dos efeitos → preditor das semanas futuras
am <- tryCatch(inla.posterior.sample(M, r, num.threads = thr, add.names = FALSE),
               error = function(e) {   # sem o pacote 'sn': amostra sem a correção de assimetria
                 message("posterior.sample sem skew.corr: ", conditionMessage(e))
                 inla.posterior.sample(M, r, num.threads = thr, add.names = FALSE, skew.corr = FALSE)
               })
nomes <- rownames(am[[1]]$latent)
pega <- function(a, comp) a$latent[startsWith(nomes, paste0(comp, ":")), 1]
nome_size <- grep("size", names(am[[1]]$hyperpar), value = TRUE)[1]
ind_ano <- if (modelo == "bb") "ano_macro" else "ano_uf"
lv_ano <- sort(unique(d[[ind_ano]]))
lv_reg <- sort(unique(d$reg_id))
rep_col <- if (modelo == "bb") "macro_id" else "uf_id"
i_woy <- fut$woy + (fut[[rep_col]] - 1) * n_woy
prec_ano_nome <- grep(paste0("Precision for ", ind_ano), names(am[[1]]$hyperpar), value = TRUE)[1]

y <- matrix(0, nrow = nrow(fut), ncol = M)
for (k in seq_len(M)) {
  a <- am[[k]]
  eta <- fut$log_E + pega(a, "(Intercept)") + pega(a, "woy")[i_woy]
  if (modelo == "bb") {
    eta <- eta + pega(a, "reg_id")[match(fut$reg_id, lv_reg)]
  } else if (modelo == "ghr_nl") {
    eta <- eta + pega(a, "reg_id")[fut$reg_id] + pega(a, "enso") * fut$enso +
      pega(a, "tbin")[fut$tbin] + pega(a, "pbin")[fut$pbin]
  } else {
    eta <- eta + pega(a, "reg_id")[fut$reg_id] + pega(a, "tanom") * fut$tanom + pega(a, "panom") * fut$panom
  }
  # efeito de ano: níveis da temporada prevista não têm dados → a posterior é a priori; sorteia com a precisão amostrada
  ea <- pega(a, ind_ano)[match(fut[[ind_ano]], lv_ano)]
  novos <- unique(fut[[ind_ano]])
  sd_ano <- 1 / sqrt(a$hyperpar[[prec_ano_nome]])
  z <- setNames(rnorm(length(novos), 0, sd_ano), novos)
  ea <- ifelse(is.na(ea) | fut[[ind_ano]] %in% novos, z[as.character(fut[[ind_ano]])], ea)
  mu <- exp(eta + ea)
  y[, k] <- rnbinom(nrow(fut), size = a$hyperpar[[nome_size]], mu = mu)
}
con <- file(file.path(pasta, "amostras.bin"), "wb")
writeBin(as.integer(c(nrow(fut), M)), con, size = 4)
writeBin(as.numeric(y), con, size = 4)                   # coluna a coluna = amostra a amostra
close(con)
write.csv(data.frame(parametro = rownames(r$summary.hyperpar), r$summary.hyperpar[, c("mean", "sd")]),
          file.path(pasta, "hiper.csv"), row.names = FALSE)
writeLines(c(paste0("modo=", modo), paste0("segundos_ajuste=", round(t_ajuste)),
             paste0("segundos_total=", round(as.numeric(difftime(Sys.time(), t0, units = "secs"))))),
           file.path(pasta, "tempo.txt"))
cat("amostras:", M, "x", nrow(fut), "\n")
