
# Genetic Variant Classifications
---

## Introduction

Accompanying slides:

[Slides](https://docs.google.com/presentation/d/1UX6y4Z6ekvBifDRAp537xLp6KJlYovjIj7jO5LheU3o/edit?usp=sharing)

The [ClinVar](https://www.kaggle.com/kevinarvai/clinvar-conflicting) dataset is a public resource containing annotations about human genetic variants. These variants are (usually manually) classified by clinical laboratories on a categorical spectrum ranging from benign, likely benign, uncertain significance, likely pathogenic, and pathogenic. Variants that have conflicting classifications (from laboratory to laboratory) can cause confusion when clinicians or researchers try to interpret whether the variant has an impact on the disease of a given patient.

Conflicting classifications are when two of any of the following three categories are present for one variant, two submissions of one category are not considered conflicting.

* Likely Benign or Benign
* VUS (uncertain significance)
* Likely Pathogenic or Pathogenic

## Question

Given a set of variant features, we are going to try and identify whether that variant is likely to posses a conflicting classification or not using machine learning as opposed to manual classification.

Conflicting classification has been assigned to a CLASS column. It is a binary representation of whether or not a variant has conflicting classifications, where 0 represents consistent classifications and 1 represents conflicting classifications.

## Models

The machine learning models I will be experimenting with to solve this problem are listed below:

* Logistic Regression
* Random Forest
* Gradient Boosting

## Conclusion

While achieving an AUC level of 62% using Random Forest is not immediately impressive, I think applying machine learning to the field of genomics is a step in the right direction.  While I have no domain knowledge in this specific area, I imagine it would be beneficial to be able to predict at better than random, whether or not a genetic variation will result in conflicting classifications.  Given this additional knowledge, labs or clinicians could add additional levels of inspection to overcome the increased likelihood of a conflicting classification.

## Challenges I faced

The most challenging aspect of this problem was preprocessing the data and interpreting results on an imbalanced dataset.  

* There were quite a few categorical variables that needed to be converted to a binary form.  

* Some of the categorical variables also consisted of thousands of unique values.  

* Every row consisted of NaN values, which meant imputing the data was a necessity.

* I had no domain knowledge in the area, so feature engineering was not an option, which likely hindered my ability to improve the accuracy of my models
* The dataset was imbalanced so using accuracy as a performance metric won’t work.  I had to examine the ROC, AUC, precision, recall and f1-score.
