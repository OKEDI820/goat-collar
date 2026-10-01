# Goat Collar: prototype

Working prototype for the CSC 3118 final year project **Bio-Inspired Medicinal Plant Species Crowd-sourcing Using Goats for Medical Heritage Preservation** (Department of Computer Science, COCIS, Makerere University).

A goat collar with two small AI models:

| Model | Question | Chosen model | Size |
|---|---|---|---|
| **M1** | Is the goat eating? (collar motion) | Integer Random Forest | 0.123 MB |
| **M2** | Which plant? (camera photo) | Distilled INT8 CNN (student of MobileNetV2) | 0.034 MB |

The camera wakes only when M1 says the goat is eating; M2 then names the plant or says "not sure", and uncertain detections go to a review queue for a person to check.

**Real vs simulated:** both models are real and run on held-out test recordings and leaf photos. The herd, dates and GPS positions are simulated, because the collars are not built yet.

## Run it

1. Install the libraries (once):
   python -m pip install -r requirements.txt
2. From the Kaggle notebooks' Output tabs, download outputs_kamminga_for_app.zip (notebook M1_7) and plant_results_for_app.zip (notebook M2_6), and unzip both into this folder.
3. Start the app:
   python -m streamlit run goat_collar_prototype.py

## Data sources
- M1: Kamminga et al. (2018), goat collar motion data, https://doi.org/10.17026/dans-zp6-fmna
- M2: Jannat, Preety & Mojumdar (2024), medicinal plant leaves, https://doi.org/10.17632/d297rvr3r6.1

## Team
Okedi Ismail Musa, Talemwa Edson, Magara Gideon, Musinguzi Dennis
Supervisor: Mr. Ggaliwango Marvin
