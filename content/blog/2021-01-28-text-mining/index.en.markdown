---
title: Text mining
author: R package build
date: '2021-01-28'
slug: text-mining
categories: []
tags: []
subtitle: ''
summary: ''
authors: []
lastmod: '2021-01-28T22:36:54+01:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---



## Introduction

In this new series of post I will explain and make  Deel learning models to 
predict outcomes from predictors in text data. Thease models are different then
orther supervised learnings algoritmes such as *regularized lenear models*,
*support vector machines* and *naive bayes* because they are deep meaning they
use multiple layers to learn how to map from input features to output outcomes. 
Where the former use a shallow (single) mapping.

In this first post on Deep learning we wil explore a **densely connected neural 
network**. This type of model is not the one that is gonna review the best
performance on text data, but it is a great *null* model to start learning
about deep learning models. I will go though the different components in 
a deep learning model and what software to use.

The deep learning models is gonna be compared to a Naive Bayes model and look
which of those two get the best result.

For the analysis I am gonna use the data on 
[kaggle](https://www.kaggle.com/nltkdata/movie-review?select=movie_review.csv)
that is part of a competition I am participated in.

## Neural network

A neural network idea is based on the human brain where we the brain receive
an input from a sensor and goes though different layers before being
executed og end up in the brain. 

The input which is the data can be stored in different dimensions. They 
are called **tensors** and is a key concept and that is way Google´s `TensorFlow`
was named efter them.

So a tensor is basically data that can be in different dimensions and it is
defined by three key attributes:

- Number of axes (rank) - where axes is a *dimension*.
- Shape - This is an integer vector that describes how many dimensions the tensor
  has. In R you can see the dimensions with `dim()`.
- Data types - this is the type of dta contained in the tensor and should be
  a `integer` or `double`.
  
The math and calculations behind tensors relies a lot on **Linear Algebra** which
I will make another post for explaining.

The last step in this short introduction int the method in getting the right 
output. Here there is different optimization methods but one that is
used a lot is **gradient descent**:

![Gradient descent](gd.png)

So the picture explain the essens of what the optimization methods does. You
start of and goes to the final point but you eyees a blinded so under the
way to the final point you have to optimize the path.

### Anatomy of a neural network

So in a neural network we have the following objects:

- Layers tha are combined into a model/network.
- The input data and corresponding target.
- A loss function which defines the feedback signal used for learning.
- The optimizer which determines how the learning are proceeded.

There are diffent types of layers and a simple layer is one that is stored in
2D tensors of shape and they are processed by **densely connected layers**.

So we first need to know which laer




```r
moviereview <- read_csv("movie_review.csv") %>%
  mutate(tag = as.factor(ifelse(tag == "pos", 1, 0)))
moviereview %>% head()
```

```
## # A tibble: 6 x 6
##   fold_id cv_tag html_id sent_id text                                      tag  
##     <dbl> <chr>    <dbl>   <dbl> <chr>                                     <fct>
## 1       0 cv000    29590       0 "films adapted from comic books have had… 1    
## 2       0 cv000    29590       1 "for starters , it was created by alan m… 1    
## 3       0 cv000    29590       2 "to say moore and campbell thoroughly re… 1    
## 4       0 cv000    29590       3 "the book ( or \" graphic novel , \" if … 1    
## 5       0 cv000    29590       4 "in other words , don't dismiss this fil… 1    
## 6       0 cv000    29590       5 "if you can get past the whole comic boo… 1
```


```r
moviereview %>%
  ggplot(aes(nchar(text))) +
  geom_histogram(binwidth = 1, alpha = 0.8) +
  labs(
    x = "Number of characters per campaign blurb",
    y = "Number of campaign blurbs"
  )
```

<img src="{{< blogdown/postref >}}index.en_files/figure-html/unnamed-chunk-2-1.png" width="2400" />


```r
moviereview_split <- moviereview %>% initial_split(strata = tag)

train <- training(moviereview_split)
test <- testing(moviereview_split)
```

## machine learing



```r
complaints_rec <- recipe(tag ~ text, data = train) %>%
  step_tokenize(text) %>%
  textrecipes::step_stopwords(text) %>%
  step_tokenfilter(text) %>%
  step_tfidf(text)
```


```r
complaint_prep <- prep(complaints_rec)
```


```r
complaint_wf <- workflow() %>%
  add_recipe(complaints_rec)
```



```r
library(discrim)
nb_spec <- naive_Bayes() %>%
  set_mode("classification") %>%
  set_engine("naivebayes")
```


```r
nb_fit <- complaint_wf %>%
  add_model(nb_spec) %>%
  fit(data = train)
```


```r
set.seed(234)
complaints_folds <- vfold_cv(train)
```


```r
nb_wf <- workflow() %>%
  add_recipe(complaints_rec) %>%
  add_model(nb_spec)
```


```r
nb_rs <- fit_resamples(
  nb_wf,
  complaints_folds,
  control = control_resamples(save_pred = TRUE)
)
```


```r
nb_rs_metrics <- collect_metrics(nb_rs)
nb_rs_predictions <- collect_predictions(nb_rs)
```


```r
nb_rs_metrics
```

```
## # A tibble: 2 x 6
##   .metric  .estimator  mean     n std_err .config             
##   <chr>    <chr>      <dbl> <int>   <dbl> <chr>               
## 1 accuracy binary     0.538    10 0.00277 Preprocessor1_Model1
## 2 roc_auc  binary     0.565    10 0.00242 Preprocessor1_Model1
```


```r
nb_rs_predictions %>%
  group_by(id) %>%
  roc_curve(truth = tag, .pred_0) %>%
  autoplot() +
  labs(
    color = NULL,
    title = "Receiver operator curve for US Consumer Finance Complaints",
    subtitle = "Each resample fold is shown in a different color"
  )
```

<img src="{{< blogdown/postref >}}index.en_files/figure-html/unnamed-chunk-14-1.png" width="2400" />


```r
nb_rs_predictions %>%
  filter(id == "Fold01") %>%
  conf_mat(tag, .pred_class) %>%
  autoplot(type = "heatmap")
```

<img src="{{< blogdown/postref >}}index.en_files/figure-html/unnamed-chunk-15-1.png" width="2400" />


## deep learning


```r
moviereview_split <- moviereview %>%
  mutate(tag = as.numeric(tag)) %>%
  initial_split(strata = tag)

train <- training(moviereview_split)
test <- testing(moviereview_split)
```



```r
train %>%
  mutate(n_words = tokenizers::count_words(text)) %>%
  ggplot(aes(n_words)) +
  geom_bar() +
  labs(
    x = "Number of words per campaign blurb",
    y = "Number of campaign blurbs"
  )
```

<img src="{{< blogdown/postref >}}index.en_files/figure-html/unnamed-chunk-17-1.png" width="2400" />


```r
library(textrecipes)

max_words <- 2000
max_length <- 50

movie_rec <- recipe(~text, data = train) %>%
  step_tokenize(text) %>%
  step_tokenfilter(text, max_tokens = max_words) %>%
  step_sequence_onehot(text, sequence_length = max_length)
```



```r
movie_prep <- prep(movie_rec)
mov_train <- bake(movie_prep, new_data = NULL, composition = "matrix")
```


```r
library(keras)

dense_model <- keras_model_sequential() %>%
  layer_embedding(
    input_dim = max_words + 1,
    output_dim = 12,
    input_length = max_length
  ) %>%
  layer_flatten() %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")
dense_model
```

```
## Model
## Model: "sequential"
## ________________________________________________________________________________
## Layer (type)                        Output Shape                    Param #     
## ================================================================================
## embedding (Embedding)               (None, 50, 12)                  24012       
## ________________________________________________________________________________
## flatten (Flatten)                   (None, 600)                     0           
## ________________________________________________________________________________
## dense_1 (Dense)                     (None, 32)                      19232       
## ________________________________________________________________________________
## dense (Dense)                       (None, 1)                       33          
## ================================================================================
## Total params: 43,277
## Trainable params: 43,277
## Non-trainable params: 0
## ________________________________________________________________________________
```


```r
dense_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)
```


```r
dense_history <- dense_model %>%
  fit(
    x = mov_train,
    y = train$tag,
    batch_size = 512,
    epochs = 20,
    validation_split = 0.25,
    verbose = FALSE
  )
```



```r
plot(dense_history)
```

<img src="{{< blogdown/postref >}}index.en_files/figure-html/unnamed-chunk-23-1.png" width="2400" />


```r
set.seed(234)
mov_val <- validation_split(train, strata = tag)
```


```r
mov_analysis <- bake(movie_prep,
  new_data = analysis(mov_val$splits[[1]]),
  composition = "matrix"
)
dim(mov_analysis)
```

```
## [1] 36407    50
```


```r
mov_assess <- bake(movie_prep,
  new_data = assessment(mov_val$splits[[1]]),
  composition = "matrix"
)
dim(mov_assess)
```

```
## [1] 12134    50
```


```r
state_analysis <- analysis(mov_val$splits[[1]]) %>% pull(tag)
state_assess <- assessment(mov_val$splits[[1]]) %>% pull(tag)
```


```r
dense_model <- keras_model_sequential() %>%
  layer_embedding(
    input_dim = max_words + 1,
    output_dim = 12,
    input_length = max_length
  ) %>%
  layer_flatten() %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

dense_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)
```


```r
val_history <- dense_model %>%
  fit(
    x = mov_analysis,
    y = state_analysis,
    batch_size = 512,
    epochs = 10,
    validation_data = list(mov_assess, state_assess),
    verbose = FALSE
  )

val_history
```

```
## 
## Final epoch (plot to see history):
##         loss: -10,591
##     accuracy: 0.4911
##     val_loss: -12,336
## val_accuracy: 0.4911
```


```r
plot(val_history)
```

<img src="{{< blogdown/postref >}}index.en_files/figure-html/unnamed-chunk-30-1.png" width="2400" />



```r
library(dplyr)

keras_predict <- function(model, baked_data, response) {
  predictions <- predict(model, baked_data)[, 1]

  tibble(
    .pred_1 = predictions,
    .pred_class = if_else(.pred_1 < 0.5, 0, 1),
    state = response
  ) %>%
    mutate(across(
      c(state, .pred_class), ## create factors
      ~ factor(.x, levels = c(1, 0))
    )) ## with matching levels
}
```


```r
val_res <- keras_predict(dense_model, mov_assess, state_assess)
val_res
```

```
## # A tibble: 12,134 x 3
##    .pred_1 .pred_class state
##      <dbl> <fct>       <fct>
##  1       1 1           <NA> 
##  2       1 1           <NA> 
##  3       1 1           <NA> 
##  4       1 1           <NA> 
##  5       1 1           <NA> 
##  6       1 1           <NA> 
##  7       1 1           <NA> 
##  8       1 1           <NA> 
##  9       1 1           <NA> 
## 10       1 1           <NA> 
## # … with 12,124 more rows
```


```r
metrics(val_res, state, .pred_class)
```

```
## # A tibble: 2 x 3
##   .metric  .estimator .estimate
##   <chr>    <chr>          <dbl>
## 1 accuracy binary             1
## 2 kap      binary           NaN
```


```r
val_res %>%
  conf_mat(state, .pred_class) %>%
  autoplot(type = "heatmap")
```

<img src="{{< blogdown/postref >}}index.en_files/figure-html/unnamed-chunk-34-1.png" width="2400" />
