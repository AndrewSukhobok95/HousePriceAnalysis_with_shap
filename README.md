## SHAP experiments

Experiments with SHAP package on [Sberbank](https://www.kaggle.com/c/sberbank-russian-housing-market) Kaggle Competetion.

Competition link: [https://www.kaggle.com/c/sberbank-russian-housing-market](https://www.kaggle.com/c/sberbank-russian-housing-market)

### Visualizations

**Data Preprocessing:**

- Price (Pic. 1) and Log Price (Pic. 2)

| Picture 1                                         | Picture 2
| :-----------------------------------------------: | :--------------------------------------------------: |
| ![Price](imgs/features_description/raw_price.png) | ![LogPrice](imgs/features_description/log_price.png) |

- Filling missing values
    - `build_year` is filled with mode by `sub_area` (Pic. 3)
    - `max_floor` is filled with mode by `build_year` (Pic. 4)
    - `kitchen_sq` and `life_sq` (Pic. 5 and Pic. 6)
        - NaNs and anomalies are replaced with regression on `full_sq`

| Picture 3                                         |
| :-----------------------------------------------: |
| ![Price](imgs/feature_prep_additional/build_year_from_sub_area_part.png) |

| Picture 4                                         |
| :-----------------------------------------------: |
| ![Price](imgs/features_prep/max_floor_from_build_year.png) |


| Picture 5                                         | Picture 6
| :-----------------------------------------------: | :--------------------------------------------------: |
| ![Price](imgs/features_prep/reg_full_sq_vs_kitch_sq.png) | ![LogPrice](imgs/features_prep/reg_full_sq_vs_life_sq.png) |

