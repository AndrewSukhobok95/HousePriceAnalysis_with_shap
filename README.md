## SHAP experiments

Experiments with SHAP package on [Sberbank](https://www.kaggle.com/c/sberbank-russian-housing-market) Kaggle Competetion.

Competition link: [https://www.kaggle.com/c/sberbank-russian-housing-market](https://www.kaggle.com/c/sberbank-russian-housing-market)

### Visualizations

**Data Preprocessing:**

- Price (Pic. 1) is converted to Log Price (Pic. 2)
- Filling missing values
    - `build_year` is filled with mode by `sub_area` (Pic. 3)
    - `max_floor` is filled with mode by `build_year` (Pic. 4)
    - `kitchen_sq` and `life_sq` (Pic. 5 and Pic. 6)
        - NaNs and anomalies are replaced with regression on `full_sq`

| Picture 1                                         | Picture 2
| :-----------------------------------------------: | :--------------------------------------------------: |
| ![](imgs/features_description/raw_price.png) | ![](imgs/features_description/log_price.png) |

| Picture 3                                         |
| :-----------------------------------------------: |
| ![](imgs/feature_prep_additional/build_year_from_sub_area_part.png) |

| Picture 4                                         |
| :-----------------------------------------------: |
| ![](imgs/features_prep/max_floor_from_build_year.png) |


| Picture 5                                         | Picture 6
| :-----------------------------------------------: | :--------------------------------------------------: |
| ![](imgs/features_prep/reg_full_sq_vs_kitch_sq.png) | ![](imgs/features_prep/reg_full_sq_vs_life_sq.png) |

**SHAP results:**

- Summary Plot

![](imgs/shap/summary_plot.png)

- Example of Force Plot (observation 3)

![](imgs/shap/force_plot/0.png)

- Example of Force Plot (observation 3)
    - `full_sq`
    - `build_year`
    - `macro_cpi`

![](imgs/shap/dependence_plot/full_sq_dependence_plot.png)

![](imgs/shap/dependence_plot/build_year_dependence_plot.png)

![](imgs/shap/dependence_plot/macro_cpi_dependence_plot.png)


