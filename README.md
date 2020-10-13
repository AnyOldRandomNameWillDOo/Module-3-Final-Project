# Predicting Functional Status Of Water Wells In Tanzania
## Using machine learning models to identify wells that are non functional or need repair

**Author**: Vivienne DiFrancesco

The contents of this repository detail an analysis of classification of Tanzanian water wells as either functional, non functional, or needs repair. This analysis is detailed in hopes of making the work accessible and replicable.

## Business problem:

The purpose of this project is to use machine learning classification models to predict the functional status of water wells in Tanzania. The different status groups for classification are functional, non functional, and functional but needs repair. The hope is that by predicting the functional status of a well, access to water could be improved across Tanzania.

## Data
The data used for this project is from the Data Driven website where, at the time of completing this project, it is an active competition. The dataset contains nearly 60,000 records of water wells across Tanzania. Each record has information that includes various location data, technical specifications of the well, information about the water, etc. The website provides a list of the features contained and a brief description of each. The link to the website to obtain the data for yourself is: https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/


## Methods
The approach for this project was to create many different model types to see what performs the best and to compare and contrast the different types of models. The way the data was preprocessed with feature engineering, filling missing values, and scaling was done with the goal of increasing accuracy of the models. The OSEMiN process is the overarching structure of this project. 

For each type of model, a model was first trained and fitted with default parameters as a base. Then, key parameters were chosen to tune using sklearn GridSearchCV and the best parameters were used to run the model. Finally, the tuned parameters were used to fit the same model using the dataset after SMOTE had been performed. Performance was compared after the tuning to the base model for each type, as well as between different model types.

## Results

### Here are examples of how to embed images from your sub-folder


#### Visual 1
![graph1](./images/visual1.png)
> Sentence about visualization.

#### Visual 2
![graph2](./images/visual2.png)
> Sentence about visualization.

## Recommendations:

More of your own text here

## Limitations & Next Steps

More of your own text here


### For further information
Please review the narrative of our analysis in [our jupyter notebook](./main_notebook.ipynb) or review our [presentation](./SampleProjectSlides.pdf)

For any additional questions, please contact **email, email, email)


##### Repository Structure:

Here is where you would describe the structure of your repoistory and its contents, for exampe:

```

├── README.md                       <- The top-level README for reviewers of this project.
├── main_notebook.ipynb             <- narrative documentation of analysis in jupyter notebook
├── presentation.pdf                <- pdf version of project presentation
└── images
    └── images                          <- both sourced externally and generated from code

```
