# Unsupervised NLP
----

This was my first attempt at using clustering algorithms.  I decided to try different permutations of unsupervised(clustering), and supervised techniques to gauge how they compared and how they worked in conjunction with one another.

For this NLP project, I decided to use the 20 Newsgroups dataset.

[The dataset can be found here](http://qwone.com/~jason/20Newsgroups/)

The 20 Newsgroups dataset is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups.  The data is organized into 20 different newsgroups, each corresponding to a different topic. Some of the newsgroups are very closely related to each other, while others are highly unrelated.

I chose 10 of the most closely related topics and I am going to try and build a classification model to predict which newsgroup a data point belongs to.

Clustering suggests groupings based on patterns in the data. These groupings are typically formed on the basis of similarity. For example, books written by the same author could be clustered together. Clustering can uncover patterns in the data that can be used for inference or for feature engineering purposes. The biggest difference between clustering and classification if that clustering can work without labels.

## Results: Supervised vs Unsupervised

Clustering suggests groupings based on patterns in the data. These groupings are typically formed on the basis of similarity. For example, books written by the same author would be clustered together. Clustering can uncover patterns in the data that can be used for inference or for feature engineering purposes. The biggest difference between clustering and classification if that clustering can work without labels.

With a classification problem, you typically have a set of predefined classes and you want to know which class a new object belongs to. With NLP problems, you’re typically dealing with a wide dataset given the corpus of words from the documents. This can make clustering a computationally expensive process.

It boils down to what you’re trying to accomplish. If you simply want a model to accurately predict what type of genre a book falls under and you have a labeled dataset to train on, then I would go with a classification model. If you’re intent is to provide book recommendation, then clustering could potentially be beneficial because it would allow you to see what books are similar to others. The two do not have to be mutually exclusive.

For this use case and with this dataset, I had a difficult time getting a clustering algorithm to perform well but I was able to achieve a test accuracy of 82% using logistic regression.
