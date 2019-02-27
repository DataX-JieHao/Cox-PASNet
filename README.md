# Cox-PASNet
### Cox-PASNet is a pathway-based sparse deep neural network for survival analysis. 
# Get Started
## Example Datasets
To get started, you need to download example datasets from URLs as below:

[Train data](http://datax.kennesaw.edu/Cox_PASNet/train.csv) 

[Validation data](http://datax.kennesaw.edu/Cox_PASNet/validation.csv)

[Test data](http://datax.kennesaw.edu/Cox_PASNet/test.csv)

[Pathway Mask data](http://datax.kennesaw.edu/Cox_PASNet/pathway_mask.csv)

[Entire data](http://datax.kennesaw.edu/Cox_PASNet/entire_data.csv)

## Training, Validation and Evaluation of Cox-PASNet
Run.py: to train the model with the inputs from train.csv. Hyperparmeters are optimized by grid search automatically with validation.csv. C-index is used to evaluate the model performance with test.csv.
## Interpretation
Run_for_Interpret.py: to interpret the model with entire_data.csv.
