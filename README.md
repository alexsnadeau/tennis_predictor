# Tennis prediction using XGBoost and Random Forests
Predicting Wimbledon using XGBoost + Building random forest from scratch to predict tennis match results.

## Where to Start
Start by installing all the dependencies needed for the project:
```Shell
conda env create -f environment.yml
conda activate tennisAI
```
Start by going through notebooks 0 (0.CleanData.ipynb) to 3 (3.Predict.ipynb).
My previous attempt at this project is also contained in DataAnalysis.ipynb. I corrected the data leakage there, however, I still would recommend just using notebooks 0 to 3, and forgetting about DataAnalysis.ipynb.

If you want to directly train, you can run ```train.py```.
Otherwise, if you want to explore the final results, just look at ```wimbledonFinalResults2025.ipynb```.

## Why you should not bet using my model?
Firstly, I'm just a youtuber and a CS student. Also, this was a two/three week long project.
Bookmakers are cracked, and they have the best models, which they keep a secret. I doubt that my model will ever be able to compete with them.
That being said, I think this is a fun project about how you can use Machine Learning to do some pretty cool things. Also, this model could be improved in a lot of ways, which I will briefly explain below.

I hope you enjoyed!