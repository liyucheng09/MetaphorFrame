# FrameBERT: Conceptual Metaphor Detection with Frame Embedding Learning

This repository contains the implementation of our EACL 2023 paper "FrameBERT: Conceptual Metaphor Detection with Frame Embedding Learning" (https://arxiv.org/abs/2302.04834). FrameBERT is a BERT-based model that leverages FrameNet embeddings for improved metaphor detection and model explainability. Our extensive experiments demonstrate the effectiveness of FrameBERT on four public benchmark datasets (VUA, MOH-X, TroFi) compared to the base model and state-of-the-art models.

**Important updates**: I have just added a `inference.py` to enable quick metaphor and frame detection on your customized data. I plan to add more features in the future. So please star our project to get posted.

## 0. To Start:

1. Clone the repository:

```
git clone https://github.com/liyucheng09/MetaphorFrame.git
cd MetaphorFrame
```

2. Install the required packages:

```
pip install -r requirements.txt
```

## 1. Run FrameBERT on Your data:

3. If you just want to **run FrameBERT directly on your own data**, just run:

```
python inference.py example_articles.json
```

Put your own data in `example_articles.json`. Check out `example_articles.json` and `inference.py`, you can easily edit them to run the program on large amount of articles.

This will produce the results to a `predictions.tsv`, which look like this:
| Tokens         | Borderline_metaphor | Real_metaphors | Frame_label     |
|----------------|---------------------|----------------|-----------------|
| The            | 0                   | 0              | _               |
| Frozen         | 1                   | 1              | _               |
| Political      | 0                   | 0              | _               |
| Battlefield    | 1                   | 1              | _               |
| In             | 1                   | 0              | _               |
| fact           | 0                   | 0              | _               |
| ,              | 0                   | 0              | _               |
| in             | 1                   | 0              | _               |
| normal         | 0                   | 0              | Typicality      |
| circumstances  | 0                   | 0              | _               |
| ,              | 0                   | 0              | _               |
| the            | 0                   | 0              | _               |
| incumbent      | 0                   | 0              | _               |
| would          | 0                   | 0              | _               |
| look           | 1                   | 0              | Give_impression |

The column `Borderline_metaphor` indicates a wide range of metaphor which can be very conventional, but `Real_metaphor` represents more interesting and novel metaphors. The `Frame_label` represents the identified Frame labels.

## Citation

If you find this repository helpful for your research, please cite our paper:

```
@misc{li2023framebert,
      title={FrameBERT: Conceptual Metaphor Detection with Frame Embedding Learning}, 
      author={Yucheng Li and Shun Wang and Chenghua Lin and Frank Guerin and Loïc Barrault},
      year={2023},
      eprint={2302.04834},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

For any questions or issues, please feel free to open an issue on GitHub or contact the authors directly.

## Reproduce the paper (optional)

You don't have to reproduce the results in the paper, if you just want to use a metaphor detection tool.

But if you want to **reproduce FrameBERT from scratch**:

3. Unzip the data:

```
unzip data_all.zip
```

After unzipping, the frame data can be found at `data_all/open_sesame_v1_data`, and other data such as VUA, MOH, and TroFi datasets can be found in their respective directories.

4. Prepare the `frame_finder` model first before we run the entire framewrok. Traning the frame model will take around 2 hours.

```
./scripts/ff.sh
```

5. config data path and `frame_finder` path in `main_config.cfg`

6. Run the main script, training on `VUA18` will take about 5 hours:

```
./scripts/run.sh
```

To see the meaning of all variables, check the explaination in the config file `main_config.cfg`.

## Repository Structure

The repository is organized as follows:

- `scripts/`: Contains all bash scripts with relevant code execution and arguments for each script.
    - `scripts/run.sh`: The main script for running FrameBERT.
- `main_config.cfg`: Configuration file for `main.py`.
- `data_all.zip`: Compressed file containing all the data needed for the project.
- `frame_finder/`: Directory containing the frame embedding model.
- `requirements.txt`: Lists the required packages for the project.

## Configuration

You can modify the configuration of the FrameBERT model by editing the `main_config.cfg` file. This file contains various settings and hyperparameters for the model.
