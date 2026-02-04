# Plant Leaf Disease Detection

Interactive web app to identify plant leaf diseases from images using a TensorFlow CNN model and Streamlit.

## Features
- Upload an image and get the predicted disease class
- Streamlit UI with Home, About, and Disease Recognition pages
- Pretrained model loaded from `trained_model.h5`

## Project Structure
- [main.py](file:///c:/leaf_disease/main.py): Streamlit app and prediction pipeline
- [trained_model.h5](file:///c:/leaf_disease/trained_model.h5): Saved Keras model used for inference
- [trained_model.keras](file:///c:/leaf_disease/trained_model.keras): Not used by the app
- [Train_plant_disease.ipynb](file:///c:/leaf_disease/Train_plant_disease.ipynb): Model training notebook
- [Test_Plant_Disease.ipynb](file:///c:/leaf_disease/Test_Plant_Disease.ipynb): Evaluation/testing notebook
- [requirement.txt](file:///c:/leaf_disease/requirement.txt): Python dependencies
- [run.txt](file:///c:/leaf_disease/run.txt) and [conda activate leafenv.txt](file:///c:/leaf_disease/conda%20activate%20leafenv.txt): Example run commands
- [home_page.jpeg](file:///c:/leaf_disease/home_page.jpeg): Home page image
- `Plant_Disease_Dataset/`: Dataset directory (ignored by Git)

## Prerequisites
- Python 3.9+ (Anaconda recommended on Windows)
- TensorFlow 2.10 (as pinned)
- Streamlit

## Setup
Create and activate a virtual environment (example with conda), then install dependencies:

```bash
conda create -n leafenv python=3.9 -y
conda activate leafenv
pip install -r requirement.txt
```

On Windows with TensorFlow/Streamlit, set a protobuf variable to avoid runtime issues:

```bash
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

## Run the App
From the project root:

```bash
streamlit run main.py
```

If using a specific Python from Anaconda, you can run:

```bash
"C:\Users\<YOUR_USER>\anaconda3\envs\leafenv\python.exe" -m streamlit run main.py
```

## How It Works
- The app loads the model once via a cached function and resizes inputs to 128×128
- See prediction pipeline and class list in [main.py](file:///c:/leaf_disease/main.py)
- The app expects `trained_model.h5` to exist in the project root

## Training
Use the notebooks to train or fine-tune the model:
- Open [Train_plant_disease.ipynb](file:///c:/leaf_disease/Train_plant_disease.ipynb) for end-to-end training
- Evaluate with [Test_Plant_Disease.ipynb](file:///c:/leaf_disease/Test_Plant_Disease.ipynb)

Note: The dataset folder is large and excluded from Git via `.gitignore`. Place data under `Plant_Disease_Dataset/` when running notebooks locally.

## GitHub Notes
- Large files: `trained_model.h5` (~90MB) is below GitHub’s 100MB hard limit but above the 50MB recommendation. Consider Git LFS if the model grows:
  - https://git-lfs.com
  - `git lfs install` then `git lfs track "*.h5"`

## Troubleshooting
- If Streamlit can’t start or TensorFlow errors on Windows, ensure:
  - Correct Python/conda environment is active
  - `pip install -r requirement.txt` succeeded
  - `set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` set before running
  - `trained_model.h5` exists in the root

## License
Add your preferred license (MIT, Apache-2.0, etc.).

