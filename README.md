The original datasets including hurricane characteristics (ian.xlsx) and power outage data (Ian.csv) are in my google drive in this link https://drive.google.com/drive/folders/1RkxZu23qWTWTRnUgQNJANqFvnhPQRsxE?usp=drive_link



HURRICANE IAN 

Among the 19 hurricanes that made landfall between 2017 and 2023, Hurricane Ian (2022) has been selected for the landfall in Florida USA.


The data we have contains up to 90 days of power outage after hurricane landfall. This is to monitor the recovery of the power overtime to see which county recover faster and why. The main aim is to find the correlation between the duration of power restoration and socioeconomic factors. 

Clusters analysis- K-means clustering, hierarchical clustering, and DBSCAN.

Find correlation between power outage (spatial lag y) and the factors (x) under consideration. This is characterized by the association between the values of x in a county and the spatial lag of y (the average value of neighbouring counties), where x is one of the vulnerability variables and y is a measure of power outage.


Centred auto logistic models

Fit two centred auto logistic regression models to examine the association between each of vulnerability measures and the binary dependent variable measuring whether a county experienced a severe outage, on both the relative and absolute scales

Perform sensitivity analysis at the end to check model robustness

Choose range of recovery time (Within the hurricane impact date or outside)
