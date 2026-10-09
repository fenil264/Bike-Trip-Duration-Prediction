# Seoul Bike Trip Duration Prediction

Regression models that predict how long a Seoul public bike-share trip will last, following the CRISP-DM process.

## Goal

Predict trip `Duration` from trip, location, time and weather features.

## Data

- Seoul bike-share trips joined with weather variables (temperature, precipitation, wind, humidity, solar radiation, snow, ground temperature, dust) and trip fields (distance, pickup and drop-off coordinates, month/day/hour/minute, day of week, Haversine distance).
- The notebook (`CGC/Bike trip duration spss/Project_code.ipynb`) loads a 9,601,139-row (about 9.6M), 26-column modeling file (`For_modeling.csv`). That full file is **not** in this repository.
- `CGC/Bike trip duration spss/project.csv` is a 49,999-row extract.

## Approach

- Data description, exploration and quality reports (Word files) and SPSS Modeler streams (`*.str`, including `project_cleaning.str`), in `CGC/Bike trip duration spss/`.
- Python notebook: Min-Max scaling, feature selection with `SelectKBest` (`f_regression`) and mutual information, 75/25 train/test split, then three regressors.

## Results (test set, R²)

| Model | R² |
|---|---|
| Linear Regression | 0.459 |
| Random Forest (200 trees, max depth 7) | 0.630 |
| XGBoost (100 trees, learning rate 0.61) | 0.728 |

XGBoost performed best in the notebook.

## Notes

- The SPSS Modeler results are not recorded in the repository files I could read, so none are reported here.
