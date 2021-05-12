# -*- coding: utf-8 -*-
"""
Biological Boolean Networks
=================================

A series of biological Boolean networks that can be directly loaded for experimentation.


"""
#   Copyright (C) 2021 by
#   Alex Gates <ajgates42@gmail.com>
#   Rion Brattig Correia <rionbr@gmail.com>
#   Xuan Wang <xw47@indiana.edu>
#   Thomas Parmer <tjparmer@indiana.edu>
#   All rights reserved.
#   MIT license.
import os
from .. boolean_network import BooleanNetwork


_path = os.path.dirname(os.path.realpath(__file__))
""" Make sure we know what the current directory is """


def THALIANA():
    """Boolean network model of the control of flower morphogenesis in Arabidopsis thaliana

    The network is defined in :cite:`Chaos:2006`.

    Returns:
        (BooleanNetwork)
    """
    return BooleanNetwork.from_file(_path + '/thaliana.txt', name="Arabidopsis Thaliana", keep_constants=True)


def DROSOPHILA(cells=1):
    """Drosophila Melanogaster boolean model.
    This is a simplification of the original network defined in :cite:`Albert:2008`.
    In the original model, some nodes receive inputs from neighboring cells.
    In this single cell network, they are condensed (nhhnHH) and treated as constants.

    There is currently only one model available, where the original neighboring cell signals are treated as constants.

    Args:
        cells (int) : Which model to return.

    Returns:
        (BooleanNetwork)
    """
    if cells == 1:
        return BooleanNetwork.from_file(_path + '/drosophila_single_cell.txt', name="Drosophila Melanogaster", keep_constants=True)
    else:
        raise AttributeError('Only single cell drosophila boolean model currently available.')


def BUDDING_YEAST():
    """

    The network is defined in :cite:`Fangting:2004`.

    Returns:
        (BooleanNetwork)
    """
    return BooleanNetwork.from_file(_path + '/yeast_cell_cycle.txt', name="Budding Yeast Cell Cycle", keep_constants=True)


def MARQUESPITA():
    """Boolean network used for the Two-Symbol schemata example.

    The network is defined in :cite:`Marques-Pita:2013`.

    Returns:
        (BooleanNetwork)
    """
    return BooleanNetwork.from_file(_path + '/marques-pita_rocha.txt', name="Marques-Pita & Rocha", keep_constants=True)


def LEUKEMIA():
    """Boolean network model of survival signaling in T-LGL leukemia

    The network is defined in :cite:`Zhang:2008`.

    Returns:
        (BooleanNetwork)
    """
    return BooleanNetwork.from_file(_path + '/leukemia.txt', type='logical', name="T-LGL Leukemia", keep_constants=True)


def BREAST_CANCER():
    """Boolean network model of signal transduction in ER+ breast cancer

    The network is defined in :cite:`Zanudo:2017`.

    Returns:
        (BooleanNetwork)
    """
    return BooleanNetwork.from_file(_path + '/breast_cancer.txt', type='logical', name="ER+ Breast Cancer", keep_constants=True)


_cell_collective_models = [
    'Apoptosis Network',
    'Arabidopsis thaliana Cell Cycle',
    'Aurora Kinase A in Neuroblastoma',
    'B bronchiseptica and T retortaeformis coinfection',
    'B cell differentiation',
    'Bordetella bronchiseptica',
    'Bortezomib Responses in U266 Human Myeloma Cells',
    'BT474 Breast Cell Line Long-term ErbB Network',
    'BT474 Breast Cell Line Short-term ErbB Network',
    'Budding Yeast Cell Cycle 2009',
    'Budding Yeast Cell Cycle',
    'Cardiac development',
    'CD4 T cell signaling',
    'CD4+ T Cell Differentiation and Plasticity',
    'CD4+ T cell Differentiation',
    'Cell Cycle Transcription by Coupled CDK and Network Oscillators',
    'Cholesterol Regulatory Pathway',
    'Colitis-associated colon cancer',
    'Cortical Area Development',
    'Death Receptor Signaling',
    'Differentiation of T lymphocytes',
    'EGFR & ErbB Signaling',
    'FA BRCA pathway',
    'Fanconi anemia and checkpoint recovery',
    'FGF pathway of Drosophila Signalling Pathways',
    'Glucose Repression Signaling 2009',
    'Guard Cell Abscisic Acid Signaling',
    'HCC1954 Breast Cell Line Long-term ErbB Network',
    'HCC1954 Breast Cell Line Short-term ErbB Network',
    'HGF Signaling in Keratinocytes',
    'HH Pathway of Drosophila Signaling Pathways',
    'HIV-1 interactions with T Cell Signalling Pathway',
    'Human Gonadal Sex Determination',
    'IGVH mutations in chronic lymphocytic leukemia',
    'IL-1 Signaling',
    'IL-6 Signalling',
    'Influenza A Virus Replication Cycle',
    'Iron acquisition and oxidative stress response in aspergillus fumigatus',
    'Lac Operon',
    'Lymphoid and myeloid cell specification and transdifferentiation',
    'Lymphopoiesis Regulatory Network',
    'Mammalian Cell Cycle 2006',
    'Mammalian Cell Cycle',
    'MAPK Cancer Cell Fate Network',
    'Metabolic Interactions in the Gut Microbiome',
    'Neurotransmitter Signaling Pathway',
    'Oxidative Stress Pathway',
    'PC12 Cell Differentiation',
    'Predicting Variabilities in Cardiac Gene',
    'Pro-inflammatory Tumor Microenvironment in Acute Lymphoblastic Leukemia',
    'Processing of Spz Network from the Drosophila Signaling Pathway',
    'Regulation of the L-arabinose operon of Escherichia coli',
    'Senescence Associated Secretory Phenotype',
    'Septation Initiation Network',
    'Signal Transduction in Fibroblasts',
    'Signaling in Macrophage Activation',
    'Signaling Pathway for Butanol Production in Clostridium beijerinckii NRRL B-598',
    'SKBR3 Breast Cell Line Long-term ErbB Network',
    'SKBR3 Breast Cell Line Short-term ErbB Network',
    'Stomatal Opening Model',
    'T cell differentiation',
    'T Cell Receptor Signaling',
    'T-Cell Signaling 2006',
    'T-LGL Survival Network 2008',
    'T-LGL Survival Network 2011 Reduced Network',
    'T-LGL Survival Network 2011',
    'TOL Regulatory Network',
    'Toll Pathway of Drosophila Signaling Pathway',
    'Treatment of Castration-Resistant Prostate Cancer',
    'Trichostrongylus retortaeformis',
    'Tumour Cell Invasion and Migration',
    'VEGF Pathway of Drosophila Signaling Pathway',
    'Wg Pathway of Drosophila Signalling Pathways',
    'Yeast Apoptosis']


def load_cell_collective_model(name=None):
    """Loads one of the Cell Collective :cite:`Helikar:2012` models.
    Models collected on Aug 2020.

    Args:
        name (str): the name of the model to be loaded.
            Accepts: ["Apoptosis Network", "Arabidopsis thaliana Cell Cycle", "Aurora Kinase A in Neuroblastoma", ...,
            "Wg Pathway of Drosophila Signalling Pathways", "Yeast Apoptosis"]

    Returns:
        (BooleanNetwork)

    Note:
        See source code for full list of models. Credits to Xuan Wang for compiling these models.
        We are working on making a Cell Collective direct loader. 
    """

    #
    if name not in _cell_collective_models:
        models_str = "'" + "','".join(_cell_collective_models) + "'"
        raise TypeError('Model name could not be found. Please specify one of the following models: {models:s}.'.format(models=models_str))
    else:
        return BooleanNetwork.from_file(_path + '/cell_collective/' + name + '.txt', name=name, keep_constants=True)


def load_all_cell_collective_models():
    """Load all the Cell Collective models, instanciating +70 models.

    Returns:
        (list)

    Note:
        See source code for full list of models.
    """
    return [load_cell_collective_model(name=name) for name in _cell_collective_models]
