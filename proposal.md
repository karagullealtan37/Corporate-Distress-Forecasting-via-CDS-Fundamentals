# Project Proposal  

**Title:** *Predicting Corporate Financial Distress Using PCA, K-Means Clustering, and Machine Learning*  

Following feedback on my initial idea — which relied on modeling the returns of AI-related equities — I decided to reformulate my project. The original topic was deemed too econometric in nature, and I also realized that financial returns and market-based features tend to be extremely noisy, limiting the predictive value of such models. Therefore, I shifted toward a more structured and data-science–aligned approach using firm fundamentals.  

This project applies a hybrid learning framework combining unsupervised and supervised methods to predict corporate financial distress using firm-level accounting and market data. The goal is to identify patterns of financial vulnerability and develop an interpretable, data-driven risk classification system.  

Using **Capital IQ** and **Datastream**, I will collect financial ratios capturing profitability, leverage, liquidity, and growth. **Principal Component Analysis (PCA)** will extract the key latent factors summarizing firms’ financial structures, and **K-Means clustering** will group firms into archetypes with similar profiles (e.g., stable, leveraged, or high-growth). A **supervised classifier** (e.g., logistic regression or random forest) will then predict the likelihood of future distress based on these PCA components and cluster labels.  

Model accuracy, recall, and ROC-AUC will be used for evaluation, while PCA loadings and feature importance will ensure interpretability.  

