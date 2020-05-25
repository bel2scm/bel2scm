import pyro

PYRO_DISTRIBUTIONS = {
    "Bernoulli": pyro.distributions.Bernoulli,
    "Categorical": pyro.distributions.Categorical,
    "Normal": pyro.distributions.Normal,
    "LogNormal": pyro.distributions.LogNormal,
    "Gamma": pyro.distributions.Gamma,
    "Delta": pyro.distributions.Delta,
    "MultivariateNormal": pyro.distributions.MultivariateNormal,
    "BetaBinomial": pyro.distributions.BetaBinomial
}

VARIABLE_TYPE = {
    "Continuous": ["abundance", "transformation"],
    "Categorical": ["process", "activity", "reaction", "pathology"]
}

LABEL_DICT = {
    'transformation': ['sec', 'surf', 'deg', 'rxn', 'tloc', 'fromLoc',
                       'products', 'reactants', 'toLoc'],
    'abundance': ['a', 'abundance', 'complex', 'complexAbundance', 'geneAbundance', 'g',
                  'microRNAAbundance', 'm', 'populationAbundance', 'pop', 'proteinAbundance', 'p',
                  'rnaAbundance', 'r', 'frag', 'fus', 'loc', 'pmod', 'var'
                                                                     'compositeAbundance', 'composite'],
    'activity': ['activity', 'act', 'molecularActivity', 'ma'],
    'reaction': ['reaction', 'rxn'],
    'process': ['biologicalProcess', 'bp'],
    'pathology': ['pathology', 'path']
}
VALID_RELATIONS = ["increases", "decreases", "directlyIncreases", "directlyDecreases"]
