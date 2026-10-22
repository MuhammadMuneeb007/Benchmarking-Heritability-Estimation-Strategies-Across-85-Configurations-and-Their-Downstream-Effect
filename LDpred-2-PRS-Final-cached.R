args <- commandArgs(trailingOnly=TRUE)

library(bigsnpr)
library(data.table)
library(magrittr)

phenodir     <- args[1]
outdir       <- args[2]
prunedpfx    <- args[3]
size_p       <- as.numeric(args[4])
alpha_p      <- as.numeric(args[5])
thr_r2_p     <- as.numeric(args[6])
h2_external  <- as.numeric(args[7])
cache_dir    <- args[8]
global_cache <- args[9]

if (!is.finite(h2_external) || h2_external <= 0 || h2_external >= 1)
    stop(paste("Invalid external h2:",h2_external))

dir.create(cache_dir,recursive=TRUE,showWarnings=FALSE)
dir.create(global_cache,recursive=TRUE,showWarnings=FALSE)
dir.create(outdir,recursive=TRUE,showWarnings=FALSE)

NCORES <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK","1"))

corr_rds  <- file.path(cache_dir,"LD_MATRIX.rds")
corr_sbk  <- file.path(cache_dir,"LD_MATRIX.sbk")
beta_rds  <- file.path(cache_dir,"DF_BETA.rds")
gwas_cache <- file.path(cache_dir,"MATCHED_GWAS.csv")

if (file.exists(corr_rds) &&
    file.exists(corr_sbk) &&
    file.exists(beta_rds) &&
    file.exists(gwas_cache)) {

    cat("LD CACHE HIT -- reusing existing LD matrix\n")

    corr     <- readRDS(corr_rds)
    df_beta  <- readRDS(beta_rds)
    gwas_out <- fread(gwas_cache)

} else {

    cat("LD CACHE MISS -- building LD matrix once\n")

    sumstats_path <- file.path(phenodir,paste0(phenodir,".txt"))
    sumstats <- bigreadr::fread2(sumstats_path)

    names(sumstats) <- c(
        "chr","pos","rsid","a1","a0",
        "n_eff","beta_se","p","BETA","INFO","MAF"
    )

    sumstats$beta <- sumstats$BETA

    bed_f <- paste0(prunedpfx,".bed")
    geno_rds <- paste0(prunedpfx,".rds")

    if (!file.exists(geno_rds))
        snp_readBed(bed_f)

    obj.bigSNP <- snp_attach(geno_rds)

    map <- obj.bigSNP$map[-3]
    names(map) <- c("chr","rsid","pos","a1","a0")

    genotype <- obj.bigSNP$genotypes
    CHR <- map$chr
    POS <- map$pos

    # Genetic maps are downloaded only once globally and then reused.
    POS2 <- snp_asGeneticPos(
        CHR,
        POS,
        dir=global_cache
    )

    info_snp <- snp_match(sumstats,map)

    cat("Matched SNPs:",nrow(info_snp),"\n")

    if (file.exists(corr_sbk))
        file.remove(corr_sbk)

    corr  <- NULL
    check <- TRUE

    tmp <- file.path(cache_dir,"LD_MATRIX")

    for (chr in 1:22) {

        ind.chr <- which(info_snp$chr == chr)

        if (length(ind.chr) == 0)
            next

        ind.chr2 <- info_snp$`_NUM_ID_`[ind.chr]

        corr0 <- snp_cor(
            genotype,
            ind.col=ind.chr2,
            ncores=NCORES,
            infos.pos=POS2[ind.chr2],
            size=size_p,
            alpha=alpha_p,
            thr_r2=thr_r2_p
        )

        if (check) {
            corr <- as_SFBM(corr0,tmp)
            check <- FALSE
        } else {
            corr$add_columns(corr0,nrow(corr))
        }
    }

    df_beta <- info_snp[,c(
        "beta","beta_se","n_eff"
    )]

    gwas_out <- info_snp[,c(
        "rsid","a0","beta"
    )]

    colnames(gwas_out) <- c(
        "SNP","a0","beta"
    )

    saveRDS(corr,corr_rds)
    saveRDS(df_beta,beta_rds)
    fwrite(gwas_out,gwas_cache)

    cat("LD CACHE CREATED\n")
}

# Same p grid for every external h2.
p_seq <- 10^seq(-4,0,length.out=10)

grid_params <- data.frame(
    p=p_seq,
    h2=rep(h2_external,10),
    sparse=rep(FALSE,10)
)

cat("External h2:",h2_external,"\n")

set.seed(1)

beta_ldpred2 <- snp_ldpred2_grid(
    corr,
    df_beta,
    grid_params,
    ncores=NCORES
)

ok <- colSums(is.na(beta_ldpred2)) == 0

beta_ldpred2 <- beta_ldpred2[,ok,drop=FALSE]
grid_params  <- grid_params[ok,,drop=FALSE]

if (nrow(grid_params) == 0)
    stop("All LDpred2 configurations failed")

write.csv(
    grid_params,
    file.path(outdir,"ldpred2_grid.csv"),
    row.names=FALSE,
    quote=FALSE
)

write.csv(
    as.data.frame(beta_ldpred2),
    file.path(outdir,"ldpred2_betas.csv"),
    row.names=FALSE,
    quote=FALSE
)

write.csv(
    as.data.frame(gwas_out),
    file.path(outdir,"ldpred2_gwas.csv"),
    row.names=FALSE,
    quote=FALSE
)

cat("LDpred2 complete. Grid points:",nrow(grid_params),"\n")
