# Tutorial on metabolic modelling

## Using conda

The following has to be done once

```
# load the module
module load conda

# conda init
```

To activate an environment:

```
conda activate my_env
````

To deactivate an environment:

```
conda deactivate
```

## Paths to the conda environments for this tutorial

```
/shared/ifbstor1/projects/tp_2534_ai_microbiomes_181502/conda/envs/ebame_metabo_gapseq
/shared/projects/tp_2534_ai_microbiomes_181502/conda/envs/ebame_metabo_reasoning
/shared/projects/tp_2534_ai_microbiomes_181502/conda/envs/numerical_modelling
```

## Tutorial

See [https://gitlab.inria.fr/cfrioux/ebame](https://gitlab.inria.fr/cfrioux/ebame)
The tutorial has a section dedicated to metabolic network reconstruction which we will not consider during this training.

We will do the following:

- Manipulation of metabolic models with [Fluxer](https://fluxer.umbc.edu/)
- A closer look at a metabolic network: the SBML file
- Metabolic network modelling
    - [Using toy data](https://gitlab.inria.fr/cfrioux/ebame#using-toy-data-)
    - [Using real networks](https://gitlab.inria.fr/cfrioux/ebame#using-real-networks-)
- Screening the metabolism of microbial communities
    - [Using toy community data](https://gitlab.inria.fr/cfrioux/ebame#using-toy-community-data-)
    - [Using realistic communities](https://gitlab.inria.fr/cfrioux/ebame#using-realistic-communities-)
- BONUS - [Inferring seed metabolites (nutrients) from the metabolic network](https://gitlab.inria.fr/cfrioux/ebame#inferring-seed-metabolites-nutrients-from-the-metabolic-network)
