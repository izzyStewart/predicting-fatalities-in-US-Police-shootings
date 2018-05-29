<h1>Predicting fatalities in US Police shootings data 2010-2016</h1>

This project analyses data collected by <a href="https://news.vice.com/en_us/article/xwvv3a/shot-by-cops">VICE</a> that looks at both fatal and non-fatal shootings from the 50 largest local police departments found in the United States from 2010 to 2016. I was interested in how this data was collected and how some of their findings to show different results to similar reports. I chose this data after reading the VICE report on the subject. I had already found another dataset that collected all fatal shootings by the police but did not record non- fatal shootings. Looking at the VICE data, I was interested to see if the ‘Fatal’ field could be predicted by using the identity factors of the officer, subject or the area in which the incident occurred. 

I have used machine learning classification techniques to predict if an incident was fatal using the remaining fields as predictors. The models I have chosen to use are Random Forest, Logistic Regression, Neural Networks and KNN. In carrying out this project it was important for me to understand the limitations of the dataset as well as the potential for bias with multiple sources feeding into the dataset. In their report, VICE describe the data collection methods as “directly from law enforcement agencies and district attorneys, though we also sometimes relied on local media reports”. They also discuss how some departments would give limited results, leading to blank fields. I wanted test if these ‘unknown’ and ‘NA’ fields would be predictive, and so, as far as possible I tried to leave these in.

I have utilised and experimented with many of the Classification Machine Learning techniques learnt across the module. Whilst the pre-processing stage of the project was a challenge, I felt the project benefited overall from decisions made at this stage. With more time I would look at further classification techniques such as SVM and extend my analysis using Neutral Networks, as this achieved promising results. While this project was undertaken to practice Classification Machine Learning approaches, the results highlight the importance of data analysis for public policy.

Read the <a href="https://github.com/izzyStewart/predicting-fatalities-in-US-Police-shootings/blob/master/Police_shootings_project_report.pdf">full report</a> and findings from this project.

<h2>Instructions</h2>

<b>Project code:</b> R

<b>R code files</b>
- preprocess_data-cleaning.R (to be run first)
- predictive_analysis.R

