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
python inference.py CreativeLang/metaphor_detection_roberta_seq liyucheng/frame_finder
```

Open and edit `inference.py` to see how to use your own data.

## 2. Reproduce FrameBERT from scratch (optional)

But if you want to **reproduce FrameBERT from scratch**, you need:

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
