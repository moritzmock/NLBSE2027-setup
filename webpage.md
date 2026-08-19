# Introduction

TBD


# Multitask Code Classification

## Competition overview

Automatically identifying low-quality and insecure code is an important
software-maintenance challenge. Relevant evidence is often distributed across
two complementary artefacts: source code and the natural-language comments
written by developers.

The NLBSE'27 Code Classification Competition invites participants to build a
model that uses both artefacts to address two related tasks jointly:

- vulnerability classification for source code; and
- Self-Admitted Technical Debt (SATD) classification for developer comments.

Models are assessed not only by their predictive
performance, but also by their execution time and computational cost in a
shared evaluation environment.

Participants must develop **one multitask model** that receives a composite
code-and-comment artefact and jointly predicts two binary labels. Each input
contains a normalised source-code function together with its associated
developer comments.

The expected output is a two-element binary vector in the following order:

| Output | Meaning |
| --- | --- |
| `[0, 0]` | No vulnerability and no SATD |
| `[1, 0]` | Vulnerability and no SATD |
| `[0, 1]` | No vulnerability and SATD |
| `[1, 1]` | Vulnerability and SATD |

The primary predictive objective is to maximise the arithmetic mean of the
vulnerability and SATD F1-scores. Participants are encouraged to investigate
representations and integration strategies that exploit the complementary
information in code and comments.

## Dataset

The competition uses a multi-annotated dataset based on
[MADE-WIC](https://doi.org/10.1145/3691620.3695348), a collection of functions
and comments mined from open-source projects. Each instance contains source
code and its associated comments, together with the following annotations:

1. a vulnerability label associated with the source-code component; and
2. an SATD label associated with the comment component.

The competition dataset is divided into three partitions:

| Partition | Availability | Intended use |
| --- | --- | --- |
| Training set | Public | Train the submitted model |
| Validation set | Public | Measure the performance of the trained model |
| Test set | Hidden | Final evaluation by the competition organisers only |

The hidden test set will be used to determine the final winner. It is not
disclosed to participants and is withheld to measure how well submissions
generalise beyond the public training and validation data.


# Requirements

You must train, tune, and evaluate your model on the provided data. We look
forward to solutions that outperform our baseline model.

Detailed instructions about the competition (data, rules, baseline, results,
etc.) will be available in the GitHub repository and Google Colab notebook.

## Competition Organizers

The competition is organised by Moritz Mock (momock@unibz.it), Thomas Borsani
(tborsani@unibz.it), and Barbara Russo (brusso@unibz.it).


## Participation Requirements

To participate in the competition, you must train, tune, and evaluate your
model using the provided training and validation sets.

Additionally, you must write a paper (2–4 pages) describing:

- The architecture and details of the classification model
- The procedure used to pre-process the data
- The procedure used to tune the classifier on the training set
- The results of your classifier on the validation set
- A link to the code/tool with proper documentation on how to run it and replicate the results

Submit the paper by the deadline using our submission form. All submissions
must conform to the [ICSE'27 formatting and submission instructions](https://conf.researchr.org/track/icse-2027/icse-2027-research-track#submission-process)
and need not be double-blind.


## Submission Acceptance

Submissions will be evaluated and accepted based on correctness and reproducibility, defined by the following criteria:

- Clarity and detail of the paper content
- Availability of the code/tool, including the training/tuning/evaluation pipeline, released as open-source
- Correct training/tuning/evaluation of your code/tool on the provided data
- Correct report of the metrics and results
- Clarity of the code documentation

We will use a formula to rank the competition submissions and determine a
winner. Details will be provided in the Google Colab notebook.

The accepted submissions will be published in the workshop proceedings.


# Citing Relevant Work

General Competition Citation:

```bibtex
@inproceedings{nlbse2027,
  author = {Moritz Mock and Thomas Borsani and Barbara Russo},
  title={The NLBSE'27 Tool Competition},
  booktitle={Proceedings of the 6th International Workshop on Natural Language-based Software Engineering (NLBSE'27)},
  year={2027}
}
```

## Multitask Code Classification Citations

Please cite these works when participating in the multitask code classification
competition:

```bibtex
@inproceedings{MockEtAl2024MadeWIC,
  author = {Mock, Moritz and Melegati, Jorge and Kretschmann, Max and Diaz Ferreyra, Nicolas E. and Russo, Barbara},
  title = {MADE-WIC: Multiple Annotated Datasets for Exploring Weaknesses In Code},
  year = {2024},
  isbn = {9798400712487},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3691620.3695348},
  doi = {10.1145/3691620.3695348},
  booktitle = {Proceedings of the 39th IEEE/ACM International Conference on Automated Software Engineering},
  pages = {2346–2349},
  numpages = {4},
  location = {Sacramento, CA, USA},
  series = {ASE '24}
}
```

```bibtex
@INPROCEEDINGS{RussoEtAl2025VulSATD,
  author={Russo, Barbara and Melegati, Jorge and Mock, Moritz},
  booktitle={2025 IEEE/ACM 33rd International Conference on Program Comprehension (ICPC)}, 
  title={Leveraging Multi-Task Learning to Improve the Detection of SATD and Vulnerability}, 
  year={2025},
  volume={},
  number={},
  pages={01-12},
  doi={10.1109/ICPC66645.2025.00017}
}
```
