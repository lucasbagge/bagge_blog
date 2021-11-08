---
title: Autoencoder for outlier detection
author: R package build
date: '2021-11-08'
slug: autoencoder-for-outlier-detection
categories:
  - Machine learning
tags:
  - machine learning
  - autoencoder
  - unsupervised
  - deep learning
subtitle: ''
summary: ''
authors: []
lastmod: '2021-11-08T20:54:29+01:00'
featured: no
draft: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---



## Introduction

In this post we will train an autoencoder to detect credit card fraud. We will also demonstrate how to train Keras models in the cloud using CloudML.

The basis of our model will be the Kaggle Credit Card Fraud Detection dataset, which was collected during a research collaboration of Worldline and the Machine Learning Group of ULB (Université Libre de Bruxelles) on big data mining and fraud detection.

The dataset contains credit card transactions by European cardholders made over a two day period in September 2013. There are 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for only 0.172% of all transactions.

## Data

Reading the data from Kaggle with `data.table::fread` that make the process much
faster.


```r
df <-
  data.table::fread("https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv")
```

The input variables consist of only numerical values which are the result of a PCA transformation. In order to preserve confidentiality, no more information about the original features was provided. The features V1, …, V28 were obtained with PCA. There are however 2 features (Time and Amount) that were not transformed. Time is the seconds elapsed between each transaction and the first transaction in the dataset. Amount is the transaction amount and could be used for cost-sensitive learning. The Class variable takes value 1 in case of fraud and 0 otherwise.

## Autoencoder

Since only 0.172% of the observations are frauds, we have a highly unbalanced classification problem. With this kind of problem, traditional classification approaches usually don’t work very well because we have only a very small sample of the rarer class.

An autoencoder is a neural network that is used to learn a representation (encoding) for a set of data, typically for the purpose of dimensionality reduction. For this problem we will train an autoencoder to encode non-fraud observations from our training set. Since frauds are supposed to have a different distribution then normal transactions, we expect that our autoencoder will have higher reconstruction errors on frauds then on normal transactions. This means that we can use the reconstruction error as a quantity that indicates if a transaction is fraudulent or not.

If you want to learn more about autoencoders, a good starting point is this video from Larochelle on YouTube and Chapter 14 from the Deep Learning book by Goodfellow et al.

## Visualization

For an autoencoder to work well we have a strong initial assumption: that the distribution of variables for normal transactions is different from the distribution for fraudulent ones. Let’s make some plots to verify this. Variables were transformed to a [0,1] interval for plotting.


```r
df %>%
  gather(variable, value, -Class) %>%
  ggplot(aes(y = as.factor(variable),
             fill = as.factor(Class),
             x = percent_rank(value))) +
  geom_density_ridges()
```

```
## Picking joint bandwidth of 0.0309
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-2-1.png" width="672" />

We can see that distributions of variables for fraudulent transactions are very different then from normal ones, except for the Time variable, which seems to have the exact same distribution.

## Preprocessing

Before the modeling steps we need to do some preprocessing. We will split the dataset into train and test sets and then we will Min-max normalize our data (this is done because neural networks work much better with small input values). We will also remove the Time variable as it has the exact same distribution for normal and fraudulent transactions.

Based on the Time variable we will use the first 200,000 observations for training and the rest for testing. This is good practice because when using the model we want to predict future frauds based on transactions that happened before.


```r
df_train <- df %>% filter(row_number(Time) <= 200000) %>% select(-Time)
df_test <- df %>% filter(row_number(Time) > 200000) %>% select(-Time)
```

### feature engering


```r
rec <- 
  recipe(Class ~ ., data = df_train) %>% 
  step_range(all_numeric()) %>% 
  prep()
```



```r
x_train <-
  bake(rec, new_data = NULL) %>% 
  select(-Class) %>% 
  as.matrix()

x_test <- 
  bake(rec, new_data = df_test) %>% 
  select(-Class) %>% 
  as.matrix()

y_train <- df_train$Class
y_test <- df_test$Class
```

### Modelling autoencoder


```r
model <- 
  keras_model_sequential()

model %>%
  layer_dense(units = 15, activation = "tanh", input_shape = ncol(x_train)) %>%
  layer_dense(units = 10, activation = "tanh") %>%
  layer_dense(units = 15, activation = "tanh") %>%
  layer_dense(units = ncol(x_train))

summary(model)
```

```
## Model: "sequential"
## ________________________________________________________________________________
## Layer (type)                        Output Shape                    Param #     
## ================================================================================
## dense_3 (Dense)                     (None, 15)                      450         
## ________________________________________________________________________________
## dense_2 (Dense)                     (None, 10)                      160         
## ________________________________________________________________________________
## dense_1 (Dense)                     (None, 15)                      165         
## ________________________________________________________________________________
## dense (Dense)                       (None, 29)                      464         
## ================================================================================
## Total params: 1,239
## Trainable params: 1,239
## Non-trainable params: 0
## ________________________________________________________________________________
```

### complie


```r
model %>%
  compile(
  loss = "mean_squared_error", 
  optimizer = "adam")
```

## Training the model

We can now train our model using the fit() function. Training the model is reasonably fast (~ 14s per epoch on my laptop). We will only feed to our model the observations of normal (non-fraudulent) transactions.

We will use callback_model_checkpoint() in order to save our model after each epoch. By passing the argument save_best_only = TRUE we will keep on disk only the epoch with smallest loss value on the test set. We will also use callback_early_stopping() to stop training if the validation loss stops decreasing for 5 epochs.


```r
checkpoint <-
  callback_model_checkpoint(
  filepath = "model.hdf5", 
  save_best_only = TRUE, 
  period = 1,
  verbose = 1
)
```

```
## Warning in callback_model_checkpoint(filepath = "model.hdf5", save_best_only =
## TRUE, : The period argument is deprecated since TF v1.14 and will be ignored.
## Use save_freq instead.
```

```r
early_stopping <- callback_early_stopping(patience = 5)

model %>% fit(
  x = x_train[y_train == 0,], 
  y = x_train[y_train == 0,], 
  epochs = 100, 
  batch_size = 32,
  validation_data = list(x_test[y_test == 0,], x_test[y_test == 0,]), 
  callbacks = list(checkpoint, early_stopping)
)
```

### Making predictions


```r
pred_train <- predict(model, x_train)
mse_train <- apply((x_train - pred_train)^2, 1, sum)

pred_test <- predict(model, x_test)
mse_test <- apply((x_test - pred_test)^2, 1, sum)
```



```r
library(Metrics)
```

```
## 
## Attaching package: 'Metrics'
```

```
## The following objects are masked from 'package:yardstick':
## 
##     accuracy, mae, mape, mase, precision, recall, rmse, smape
```

```r
auc(y_train, mse_train)
```

```
## [1] 0.9542684
```

```r
auc(y_test, mse_test)
```

```
## [1] 0.95842
```

>To use the model in practice for making predictions we need to find a threshold k for the MSE, then if if MSE>k we consider that transaction a fraud (otherwise we consider it normal). To define this value it’s useful to look at precision and recall while varying the threshold k.


```r
possible_k <- seq(0, 0.5, length.out = 100)
precision <- sapply(possible_k, function(k) {
  predicted_class <- as.numeric(mse_test > k)
  sum(predicted_class == 1 & y_test == 1)/sum(predicted_class)
})

qplot(possible_k, precision, geom = "line") + 
  labs(x = "Threshold", y = "Precision") 
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-10-1.png" width="672" />


```r
recall <- sapply(possible_k, function(k) {
  predicted_class <- as.numeric(mse_test > k)
  sum(predicted_class == 1 & y_test == 1)/sum(y_test)
})
qplot(possible_k, recall, geom = "line") + 
  labs(x = "Threshold", y = "Recall")
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-11-1.png" width="672" />


```r
cost_per_verification <- 1

lost_money <- sapply(possible_k, function(k) {
  predicted_class <- as.numeric(mse_test > k)
  sum(cost_per_verification * predicted_class + (predicted_class == 0) * y_test * df_test$Amount) 
})

qplot(possible_k, lost_money, geom = "line") + labs(x = "Threshold", y = "Lost Money")
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-12-1.png" width="672" />


```r
possible_k[which.min(lost_money)]
```

```
## [1] 0.06565657
```
