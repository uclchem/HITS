# HITs
This repository stores the code to produce the History Independent Tracers by Holdship & Viti as well as the website we built to allow observers to make use of it.

## Website
If you clone this repo, you can host the transition selector website on your own machine. Simply navigate to `website` and run the following
```
    export FLASK_APP="target_finder"
    flask run
```

## Reproducing the HITs paper
All the code to produce the HITs paper is in `Python/`. The notebooks were working documents I used to explore the data and the scripts need to be run to produce the data set. You need your own version of UCLCHEM, compiled to python. The scripts are run in the following order:

| Script | Purpose |
|--------|---------|
| history-grid.py | Produces the initial abundances for each of the different histories for all densities.|
| actual-grid.py | Runs the physical parameter grid with all histories. The grid is defined in this script and the densities must match those run in `history-grid.py`. |
| summarize-grid.py | Largely exists because I ran the code on a server but is required because the summary data it produces is used in later steps. |
| radex-grid.py | You need to run notebook `1 - Grid Results` before this script. That will tell you what your HITs are and which have collisional data. You then list them in this script and run it to produce the line intensities of all transitions|
|generate_mutual_info_table.py | Computes the mutual information between all transitions/ratios and the physical parameters.|

After running those scripts, notebook `2 - Tracers.ipynb` allows you to examine your features, rank them by mutual information and test how well they perform in a random forest model.

Other scripts are unnecessary but useful for making plots.

| Script | Purpose |
|--------|---------|
|get_examples.py | Helps generate Fig.1 in the paper, gets some example models to plot.|
|time_to_forget.py | Creates a table with how long each HIT took to forget its history in each case, used to produce the time to forget statistic in Table 3.|
|plot_histories.py | Doesn't actually appear in the paper but you can plot the abundances of different species from the various histories.|