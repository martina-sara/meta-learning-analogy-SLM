# meta-learning-analogy-SLM
Code and databased used for "Inducing Structural Mapping: a few-shot meta-learning approagch to Analogical reasoning in SLMs"

## Data
The dataset was constructed selecting items from [AnalogyKB Wikidata](https://github.com/siyuyuan/analogykb) and the [BATs](https://vecto.space/projects/BATS/). Items were first analyzed employing the code evaluate_knowledge.py from [Lee et al. (2026)](https://github.com/dmis-lab/analogical-reasoning/tree/main). The result was then used to construct both training and testing episodes.
Content of the folder:
* evaluate_knowledge_combined.jsonl
* episode_builder.py
* test_episodes.py
* train_episodes.py

## Source code
To run the training and testing code use train_test.bh
Model and other parameters can be costumized.
