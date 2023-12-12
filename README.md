# LearningMotorFeedback

Code for the paper "Learning to segment self-generated from externally caused optic flow through sensorimotor mismatch circuits" by Brucklacher, Pezzulo, Mannella, Galati and Pennartz (2023): https://www.biorxiv.org/content/10.1101/2023.11.15.567170v1

## Dependencies
- [Conda](https://www.anaconda.com/)
- Setup the conda environment `motorpred_env` by running:

    ```bash
    conda env create -f environment.yml
    ```

- With the activated environment, manually run:
    ```bash
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

    ```

- With the activated environment, install the local package 'mspc' to allow absolute imports of modules. To do so run the following from directory 'LearningMotorFeedback':
    ```bash
    pip install -e .
    ```

## Recreate figures from the paper
- Recreate figures for the microcircuit by running the respective `fig<figure_number>.py` files located in `/model1_global/`

- Recreate figures for the retinotopic model by running `/model2_retinotopic/create_figures/recreate_all.py`

