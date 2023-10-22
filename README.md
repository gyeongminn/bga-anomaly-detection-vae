# BGA Anomaly detection by exploiting Variational AutoEncoders

BGA (Ball Grid Array) is a type of semiconductor package that features hundreds of connection points arranged on its underside. Common flaws in BGA include cracks, scratches, missing connections, and bridging, which can adversely affect the semiconductor's electrical performance. Accurate and efficient inspection of BGA defects is therefore essential. However, most existing methods for BGA defect detection rely on rule-based approaches, making them dependent on pre-set parameters by the user. This poses challenges as material changes would require frequent reconfiguration.

In this study, we propose a method based on Variational AutoEncoders (VAE), eliminating the need for pre-configuring inspection parameters. Based on the small set of actual package samples, we augmented BGA package images. To detect semiconductor defects, the cosine similarity between the original image and the image generated through VAE is used as the anomaly score. By running our experiment, the proposed model outperforms rule-based methods. In the future, utilizing this model will enable defect detection without the need for pre-set parameters, thus providing a more user-friendly machine vision user interface.

# Poster
[ISMP 2023 Poster PDF](https://github.com/gyeongminn/bga-anomaly-detection/blob/main/docs/ismp_2023_gyeongmin.pdf)

# Method
![Method](https://github.com/gyeongminn/bga-anomaly-detection/assets/97784561/3f16c781-97be-469e-ab74-95fceeedba3b)

# Model evaluation
|Accuracy|Precision|Recall|Specificity|F1-Score|AUC|Speed|
|---|---|---|---|---|---|---|
|0.849|0.958|0.946|0.859|0.898|0.919|3.424ms|
### Density Plot
![Density Plots](/assets/predict_density_plot.png)
### ROC Curve
![ROC Curve](/assets/roc_curve.png)
