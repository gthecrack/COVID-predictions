# COVID-predictions

## PROJECT GOALS

The goal of this project is to answer the question 'Can we predict which inpatient will need intensive care unit (ICU)?' based on parameters presented by the patients when admitted in to the ICU. Due to the time available for the project the aim is to answer the question for those patients in which their parameters were obtained during the 0-2 hour time window since their admittance.


There are 4 groups of features which contain these parameters and are the following:

Demographics ( No–:3 )

    Percentil Age.
    Above 65 years old.
    Gender.

Comorbities ( No–:9 )

    The features were created based on the historical ICD-10 codes of each patient using the Charlson and Elixhauser range of comorbid conditions (https://pubmed.ncbi.nlm.nih.gov/16224307/ https://pubmed.ncbi.nlm.nih.gov/9431328/). Here, we have chosen the comorbid groups related to serious adverse outcomes in COVID-19.

Vital Signs ( No–:36 )

    Diastolic blood pressure.
    Systolic blood pressure.
    Heart rate.
    Respiratory rate.
    Temperature.
    Oxygen saturation.

Laboratory ( No–:180 )

    There are 36 laboratorys types.

The outcome variable is ICU admission.
The following features were created for each vital signs and time window:

    mean
    median
    min
    max
    amplitude (diff): max-min
    relative amplitude (rel): amplitude/median

The following features were created for each laboratory exam and time window:

    mean
    median
    min
    max
    amplitude (diff): max-min

## HOW DOES IT WORK?

#1 Cleaning the dataset 

The first thing done was to fill the missing values of the dataset with the neighboring windows as the dataset author points out and clean the columns for easier use when training models later on.

The second thing was to isolate those patients from the 0-2 hour window in order to get a more concrete sample size. 

The last thing was to normalize the data  to have 2 different approaches when training models (with and without normalized data)

#2 Data modelling

The cleaned data was first modelled with these simple individual models:

  -k-nearesr
  -logis
  -svm-linear
  -svm-rbf
  -randomforest
  
The following metrics were obtain for each of the models in order to get a first overview of the potential results:

  -Accuracy
	-Precision
	-Recall
	-F1Score
  
Later on an open source machine learning library called PyCaret was used in order to process the data in a more advanced form. The column 'ICU NEW' was made the target for the model since it tells us if the patient went into the ICU after the window we are studying to obtain the groundtruth value.

PyCaret's recommended experiment workflow is to use compare_models() right after setup to evaluate top performing models and finalize a few candidates for continued experimentation. As such, the function that actually allows to you create a model is unimaginatively called create_model(). This function creates a model and scores it using stratified cross-validation. Similar to compare_models(), the output prints a score grid that shows Accuracy, Recall, Precision, F1 and Kappa by fold.

It was decided to work with the top 5 models as our candidate models.

    Extra Trees Classifier('et')
    CatBoost Classifier('catboost')
    Random Forest Classifier('rf')
    Logistic Regression('lr')
    Extreme Gradient Boosting('xgboost')
    
Each of these models were used to predict a portion of the patients need to be admitted into the ICU since those were not used for the modelling with varying results of accuracy.

## Future steps

The curation and cleaning process of the dataset can be aproached differently and the use of different models could also give another insight that could refine which parameters should be monitored more closely.
