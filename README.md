<h1>Predicting fatalities in US Police shootings data 2010-2016</h1>

This project analyses data collected by <a href="https://news.vice.com/en_us/article/xwvv3a/shot-by-cops" target="_blank">VICE</a> that looks at both fatal and non-fatal shootings from the 50 largest local police departments found in the United States from 2010 to 2016. I was interested in how this data was collected and how some of their findings to show different results to similar reports. I chose this data after reading the VICE report on the subject. I had already found another dataset that collected all fatal shootings by the police but did not record non- fatal shootings. Looking at the VICE data, I was interested to see if the ‘Fatal’ field could be predicted by using the identity factors of the officer, subject or the area in which the incident occurred. 

I have used machine learning classification techniques to predict if an incident was fatal using the remaining fields as predictors. The models I have chosen to use are Random Forest, Logistic Regression, Neural Networks and KNN. In carrying out this project it was important for me to understand the limitations of the dataset as well as the potential for bias with multiple sources feeding into the dataset. In their report, VICE describe the data collection methods as “directly from law enforcement agencies and district attorneys, though we also sometimes relied on local media reports”. They also discuss how some departments would give limited results, leading to blank fields. I wanted test if these ‘unknown’ and ‘NA’ fields would be predictive, and so, as far as possible I tried to leave these in.

Looking at the performance across all models the Random Forest model using the Upsampled training data achieved the best results across the training, test and validation sets. The overall best result was achieved on the validation set with 98% AUC and a 95% prediction accuracy. It was interesting to see the Number of Officers as the strong predictor in both the Random Forest and Logistic regression model, giving strong evidence to show the increase in officers increases the likelihood of the incident being fatal. This could be as the number of officers called to a scene increased with the severity of the incident or may be as officers need a need to protect their colleagues, it would be interesting to look into this further.

To develop this project, I would be interested to look further into the relationships between the cities and the subject being Fatal, as well as the time of the incident. I chose to remove the time feature for this project, only looking at the year the incident took place. However, this feature could potentially be utilised for a time series project that looks at what factors effect the fatal to non-fatal field over time. Looking briefly into this I found a <a href="https://static1.squarespace.com/static/56996151cbced68b170389f4/t/57e1b5cc2994ca4ac1d97700/1474409936835/Police+Use+of+Force+Report.pdf" target="_blank">report</a> that depicts the police gun fire policies that have been implemented each city. Looking at this data I found that the average percentage of policies implemented in the cities documented was 39.8%. This number increases to 87.5% in Philadelphia, which logistic regression showed the subject to be less likely to be fatal in that area. Development in this area could be used to evidence the need for police gun fire polices. 

I have utilised and experimented with many of the Classification Machine Learning techniques learnt across the module. Whilst the pre-processing stage of the project was a challenge, I felt the project benefited overall from decisions made at this stage. With more time I would look at further classification techniques such as SVM and extend my analysis using Neutral Networks, as this achieved promising results. While this project was undertaken to practice Classification Machine Learning approaches, the results highlight the importance of data analysis for public policy.

Read the <a href="https://github.com/izzyStewart/predicting-fatalities-in-US-Police-shootings/blob/master/Police_shootings_project_report.pdf" target="_blank">full report</a> and findings from this project.

<h2>Instructions</h2>

<b>Project code:</b> R

<b>R code files</b>
- preprocess_data-cleaning.R (to be run first)
- predictive_analysis.R

