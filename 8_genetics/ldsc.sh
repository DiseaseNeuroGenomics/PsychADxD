# Install ldsc
# see https://github.com/bulik/ldsc

# Download GWAS summary statistics
# see Methods for a list of publications used

# Download supplementary data
wget https://data.broadinstitute.org/alkesgroup/LDSCORE/eur_w_ld_chr.tar.bz2
wget https://data.broadinstitute.org/alkesgroup/LDSCORE/w_hm3.snplist.bz2

# Munge data
tar -jxvf eur_w_ld_chr.tar.bz2
bunzip2 w_hm3.snplist.bz2

# Activate conda environment with ldsc
conda activate ldsc

# Munge sumstats
./munge_sumstats.py --sumstats AD_2022_Bellenguez.tsv --N 788989 --out ad --merge-alleles w_hm3.snplist --ignore beta
./munge_sumstats.py --sumstats LBD_2021_Chia.tsv --N 7372 --out dlbd --merge-alleles w_hm3.snplist --ignore beta
./munge_sumstats.py --sumstats PD_2019_Nalls.tsv --N 1437700 --out pd --merge-alleles w_hm3.snplist --ignore beta
./munge_sumstats.py --sumstats FTD_2014_Ferrari.tsv --N 12928 --out ftd --merge-alleles w_hm3.snplist --ignore beta
./munge_sumstats.py --sumstats SCZ_2022_Trubetskoy.tsv --N 320404 --out sc --merge-alleles w_hm3.snplist --ignore beta
./munge_sumstats.py --sumstats BD_2021_Mullins.tsv --N 413466 --out bd --merge-alleles w_hm3.snplist --ignore beta

# Run ldsc.py script - genetic correlation between a pair of traits
./ldsc.py --rg sc.sumstats.gz,bd.sumstats.gz --ref-ld-chr eur_w_ld_chr/ --w-ld-chr eur_w_ld_chr/ --out sc_bd
./ldsc.py --rg sc.sumstats.gz,ftd.sumstats.gz --ref-ld-chr eur_w_ld_chr/ --w-ld-chr eur_w_ld_chr/ --out sc_ftd
./ldsc.py --rg sc.sumstats.gz,ad.sumstats.gz --ref-ld-chr eur_w_ld_chr/ --w-ld-chr eur_w_ld_chr/ --out sc_ad
./ldsc.py --rg sc.sumstats.gz,dlbd.sumstats.gz --ref-ld-chr eur_w_ld_chr/ --w-ld-chr eur_w_ld_chr/ --out sc_dlbd
./ldsc.py --rg sc.sumstats.gz,pd.sumstats.gz --ref-ld-chr eur_w_ld_chr/ --w-ld-chr eur_w_ld_chr/ --out sc_pd
./ldsc.py --rg bd.sumstats.gz,ftd.sumstats.gz --ref-ld-chr eur_w_ld_chr/ --w-ld-chr eur_w_ld_chr/ --out bd_ftd
./ldsc.py --rg bd.sumstats.gz,ad.sumstats.gz --ref-ld-chr eur_w_ld_chr/ --w-ld-chr eur_w_ld_chr/ --out bd_ad
./ldsc.py --rg bd.sumstats.gz,dlbd.sumstats.gz --ref-ld-chr eur_w_ld_chr/ --w-ld-chr eur_w_ld_chr/ --out bd_dlbd
./ldsc.py --rg bd.sumstats.gz,pd.sumstats.gz --ref-ld-chr eur_w_ld_chr/ --w-ld-chr eur_w_ld_chr/ --out bd_pd
./ldsc.py --rg ftd.sumstats.gz,ad.sumstats.gz --ref-ld-chr eur_w_ld_chr/ --w-ld-chr eur_w_ld_chr/ --out ftd_ad
./ldsc.py --rg ftd.sumstats.gz,dlbd.sumstats.gz --ref-ld-chr eur_w_ld_chr/ --w-ld-chr eur_w_ld_chr/ --out ftd_dlbd
./ldsc.py --rg ftd.sumstats.gz,pd.sumstats.gz --ref-ld-chr eur_w_ld_chr/ --w-ld-chr eur_w_ld_chr/ --out ftd_pd
./ldsc.py --rg ad.sumstats.gz,dlbd.sumstats.gz --ref-ld-chr eur_w_ld_chr/ --w-ld-chr eur_w_ld_chr/ --out ad_dlbd
./ldsc.py --rg ad.sumstats.gz,pd.sumstats.gz --ref-ld-chr eur_w_ld_chr/ --w-ld-chr eur_w_ld_chr/ --out ad_pd
./ldsc.py --rg dlbd.sumstats.gz,pd.sumstats.gz --ref-ld-chr eur_w_ld_chr/ --w-ld-chr eur_w_ld_chr/ --out dlbd_pd
