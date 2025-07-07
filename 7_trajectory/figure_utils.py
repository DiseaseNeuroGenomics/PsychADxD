import copy
import pandas as pd

"""Miscellaneous helper functions and constants"""

go_bp_paths_for_paper = [
    "neuron cell-cell adhesion",
    "postsynaptic membrane assembly",
    "presynaptic membrane organization",
    "ionotropic glutamate receptor signaling pathway",
    "cytoplasmic translation",
    "aerobic electron transport chain",
    "Golgi lumen acidification",
    "cyclic nucleotide catabolic process",
    "cholesterol biosynthetic process",
    "aspartate transmembrane transport",
    "chaperone-mediated autophagy",
    "chaperone-mediated protein complex assembly",
    "regulation of cellular response to heat",
    "NK T cell differentiation",
    "antigen processing and presentation of exogenous peptide antigen via MHC class II",
    "positive regulation of monocyte differentiation",
    "positive regulation of T-helper 17 type immune response",
    "negative regulation of B cell activation",
    "detection of bacterium",
    "alpha-beta T cell proliferation",
    "connective tissue replacement",
    "cellular response to interleukin-17",
    "detoxification of copper ion",
    "positive regulation of nitric oxide biosynthetic process",
    "regulation of endothelial cell differentiation",
    "adherens junction assembly",
    "response to laminar fluid shear stress",
    "regulation of glycolytic process",
    "platelet aggregation",
    "regulation of amyloid precursor protein catabolic process",
    "negative regulation of focal adhesion assembly",
    "transport across blood-brain barrier",
]

go_bp_paths_for_paper_sliding_window = [
    'cytoplasmic translation',
    'detection of bacterium',
    'macrophage activation',
    'toll-like receptor 2 signaling pathway',
    'regulation of phagocytosis',
    'antigen processing and presentation of peptide antigen',
    'regulation of monocyte differentiation',
    'negative regulation of B cell proliferation',
    'positive regulation of CD4-positive, alpha-beta T cell proliferation',
    'regulation of macrophage derived foam cell differentiation',
    'positive regulation of amyloid-beta clearance',
    'positive regulation of macrophage derived foam cell differentiation',
    'cholesterol catabolic process',
    'reverse cholesterol transport',
    'high-density lipoprotein particle assembly',
    'phospholipid efflux',
    'regulation of adipose tissue development',
    'regulation of macrophage cytokine production',
    'negative regulation of amyloid precursor protein catabolic process',
    'cellular response to hydroperoxide',
    'chaperone-mediated autophagy',
    'chaperone-mediated protein complex assembly',
    'negative regulation of B cell activation',
    'signal complex assembly',
    'ionotropic glutamate receptor signaling pathway',
]


bad_path_words = [
    "limb",
    "blastocyst",
    "gastric",
    "podocyte",
    "cognition",
    "decidualization",
    "acrosome",
    "cardiac",
    "vocalization",
    "auditory stimulus",
    "sensory",
    "learning",
    "memory",
    "walking",
    "behavior",
    "social",
    "locomotor",
    "nervous system",
    "corpus callosum",
    "forebrain",
    "cerebral cortex",
    "hippocamp",
    "cerebellar",
    "cranial",
    "forelimb",
    "hindlimb",
    "startle",
    "prepulse inhibition",
    "dosage",
    "substantia nigra",
    "retina",
    "optic"
    "bone",
    "kidney",
    "glomerulus",
    "heart",
    "ventricular",
    "metanep",
    "nephron",
    "glomerular",
    "of muscle",
    "bone",
    "respiratory",
    "pigmentation",
    "outflow tract septum",
    "placenta",
    "olfactory",
    "aortic",
    "germ layer",
    "mesodermal",
    "epithelial",
    "pulmonary",
    "lung",
    "embronic",
    "embryo",
    "mammary",
    "egg",
    "sperm",
    "cadmium",
    "sarcoplasmic",
    "neuron fate",
    "pregnancy",
    "osteoblast",
    "prostate",
    "hepatocyte",
    "estrous",
    "muscle atrophy",
    "neuromuscular",
    "egg",
    "ovulation",
    "with host",
    "pancreatic",
    " organ ",
    "hematopoietic",
    "epiderm",
    " head ",
    "mesoderm",
    "endoderm",
    "cerebellum",
    "embryonic",
    "ossification",
    "cochlea",
    "digestive",
    "melanocyte",
    "lead ion",
    "coronary",
    "skeletal",
    "developmental growth",
    "metanephric",
    " otic ",
]

gwas_dict = {
    "alzBellenguezNoApoe": "AD_2022_Bellenguez",
    "ms": "MS_2019_IMSGC",
    "pd_without_23andMe": "PD_2019_Nalls",
    "migraines_2021": "Migraines_2021_Donertas",
    "als2021": "ALS_2021_vanRheenen",
    "stroke": "Stroke_2018_Malik",
    "epilepsyFocal": "Epilepsy_2018_ILAECCE",

    "sz3": "SCZ_2022_Trubetskoy",
    "bip2": "BD_2021_Mullins",
    "asd": "ASD_2019_Grove",
    "adhd_ipsych": "ADHD_2023_Demontis",
    "mdd_ipsych": "MDD_2023_AlsBroad",
    "ocd": "OCD_2018_IOCDF_GC",
    "insomn2": "Insomnia_2019_Jansen",
    "alcohilism_2019": "Alcoholism_2019_SanchezRoige",
    "tourette": "Tourettes_2019_Yu",
    "intel": "IQ_2018_Savage",
    "eduAttainment": "Education_2018_Lee",
}

def condense_pathways(pathway):

    pathway = pathway.split()

    for n, p in enumerate(pathway):
        p = copy.deepcopy(p.lower())
        if p == "positive":
            pathway[n] = "pos."
        elif p == "negative":
            pathway[n] = "neg."
        elif p == "regulation":
            pathway[n] = "reg."
        elif p == "response":
            pathway[n] = "resp."
        elif p == "neurotransmitter":
            pathway[n] = "neurotrans."
        elif p == "neurotransmitter":
            pathway[n] = "neurotrans."
        elif p == "modulation":
            pathway[n] = "mod."
        elif p == "differentiation":
            pathway[n] = "diff."
        elif p == "biosynthetic":
            pathway[n] = "biosynth."
        # elif p == "mitochondrial":
        #    pathway[n] = "mito."
        elif p == "nitric-oxide":
            pathway[n] = "NO"
        elif p == "glutamate":
            pathway[n] = "GLU"
        elif p == "glutamatergic":
            pathway[n] = "GLU"
        elif p == "homeostasis":
            pathway[n] = "homeo."
        elif p == "signaling":
            pathway[n] = "sign."
        elif p == "exocytosis":
            pathway[n] = "exocyt."
        elif p == "colony-stimulating":
            pathway[n] = "colony-stim."
        elif p == "derived":
            pathway[n] = "der."
        elif p == "multicellular":
            pathway[n] = "multicell."
        elif p == "presentation":
            pathway[n] = "present."
        elif p == "organismal-level":
            pathway[n] = "org.-level"
        elif p == "proliferation":
            pathway[n] = "prolif."
        elif p == "homeostasis":
            pathway[n] = "homeo."
        elif p == "t-helper":
            pathway[n] = "T-help."
            # elif p == "immune":
            #    pathway[n] = "imm.."
        elif p == "helper":
            pathway[n] = "help."
        elif p == "ligand-gated":
            pathway[n] = "lig.-gated"
        elif p == "macrophage":
            pathway[n] = "macrophg."
        elif p == "associated":
            pathway[n] = "ass."
        elif p == "transport":
            pathway[n] = "trans."
        elif p == "synthesis":
            pathway[n] = "synth."
        elif p == "contraction":
            pathway[n] = "contract."
        elif p == "migration":
            pathway[n] = "migrat."
        elif p == "processing":
            pathway[n] = "proc."
        elif p == "exogenous":
            pathway[n] = "exon."
        elif p == "gamma-aminobutyric acid":
            pathway[n] = "GABA"

        for n in range(len(pathway) - 1):
            if pathway[n] == "calcium" and pathway[n + 1] == "ion":
                pathway[n] = "Ca2+"
                pathway[n + 1] = ""
            elif pathway[n] == "iron" and pathway[n + 1] == "ion":
                pathway[n] = "Fe2+"
                pathway[n + 1] = ""
            elif pathway[n] == "manganese" and pathway[n + 1] == "ion":
                pathway[n] = "Mn2+"
                pathway[n + 1] = ""
            elif pathway[n] == "calcium" and "ion-" in pathway[n + 1]:
                pathway[n] = "Ca2+ " + pathway[n + 1][4:]
                pathway[n + 1] = ""
        for n in range(len(pathway) - 2):
            if pathway[n] == "mhc" and pathway[n + 1] == "class" and pathway[n + 2] == "ii":
                pathway[n] = "MHC-II"
                pathway[n + 1] = ""
                pathway[n + 2] = ""

        pathway = " ".join(pathway)
        pathway = pathway.split()
        return " ".join(pathway)


def remove_bad_terms(df):
    idx = []
    for i in range(len(df)):
        include = True
        for w in bad_path_words:
            if w in df.loc[i]["pathway"]:
                include = False
        idx.append(include)

    return df[idx]