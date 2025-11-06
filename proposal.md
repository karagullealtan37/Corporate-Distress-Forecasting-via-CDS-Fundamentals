# Project Proposal  

**Title:** *Predicting Corporate Financial Distress Using PCA, K-Means Clustering, and Machine Learning*  

After receiving feedback on my initial idea, I decided to shift toward a more structured and data-driven project. The previous focus on AI-related stock returns was limited by high data noise and weak predictive signals. This new approach leverages firm fundamentals to build a cleaner and more interpretable model aligned with data science methods.  

The project aims to predict corporate financial distress using a hybrid learning framework that combines unsupervised and supervised techniques. Using **Capital IQ** and **Datastream**, I will gather firm-level financial ratios capturing profitability, leverage, liquidity, and growth. **Principal Component Analysis (PCA)** will first reduce dimensionality and identify the main latent financial factors. **K-Means clustering** will then group firms into archetypes with similar structures, such as stable, leveraged, or high-growth profiles.  

These PCA components and cluster assignments will serve as features for a **supervised classifier** (e.g., logistic regression or random forest) designed to predict the likelihood of financial distress, defined by negative earnings or significant market drawdowns. Model accuracy, recall, and ROC-AUC will assess predictive performance, while PCA loadings and feature importance will be used to interpret key financial risk drivers.


