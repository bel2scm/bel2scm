# Folder Description

## Files in this folder:

### 1. scm.py
The master/driver file to run bel2scm algorithm. 
It contains the logic for bel2scm algorithm along with functions for
intervention and counterfactual inference.

### 2. bel_graph.py
This is a helper file for scm.py which takes graph structure as input
and returns a tree data structure to run the script.
Supported input formats include:
- json
- nanopub
- string
- jgf

### 3. config.py
This is a helper file for scm.py which assigns
configurable inputs from config.json file. 
Parameters taken here:
- pyro distribution based on node type
- prior threshold
- parent-child interaction type
- max abundance for continuous variable

### 4. constants.py
This is a helper file to convert string inputs to meaningful outputs.
Currently it supports following variables:
- pyro distribution names as string (e.g. "Bernoulli") to actual pyro distribution (e.g. pyro.distributions.Bernoulli)
- bel variables to categorical or continuous variables
- distribution for noise based on their variable type
- categories for sub-categories of bel variable types

### 5. node.py
creates a node object containing relevant information like parent name, parent type, 
child name, child type, relationship type of a particular node in bel graph.

### 6. parameter_estimation.py
This file is used to take in node structure and relevant observational data for the given node and it's parents
and learns parameters, i.e parameters for their pyro distribution.

### 7. parent_interaction_types.py
This file is used to translate whether the relationship between parent-child is AND or OR.

### 8. utils.py
This file contains every other utility function to run scm.py ranging from updating queue for graph traversal to
getting relevant distribution for a given node.