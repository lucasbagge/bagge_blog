---
title: Applied tidymodels
author: R package build
date: '2021-05-14'
slug: applied-tidymodels
categories:
  - Machine learning
  - tidymodels
tags:
  - tidymodels
  - ml
  - housing pricing
  - knn
subtitle: 'How to use machine learning in practice with tidymodels as backend api'
summary: ''
authors: [Lucas Bagge]
lastmod: '2021-05-14T22:16:30+02:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---





## Applied Machine learning

This post will explore how to predict houses prices where we read in the packages where we use the package `AmesHousing` where the used data is avaialbe.


```r
ames <- make_ames()
```

## The Modeling Process

Common steps during model building are:

- estimating model parameters (i.e. training models)

- determining the values of tuning parameters that cannot be directly calculated from the data

- model selection (within a model type) and model comparison (between types)

- calculating the performance of the final model that will generalize to new data

Many books and courses portray predictive modeling as a short sprint. A better analogy would be a marathon or campaign (depending on how hard the problem is).

![](ml-process.png)
In this tutorial we will go thoug each of the steps so we get a clear understanding on what need to be done.

Another view of the process is as follow:

![](wf.png)

All of this will be much clearer after this post.

There are a few suffixes that we'll use for certain types of objects:

We are going to introduce a few suffixes for the objects we are gonna use.

- _mod for a parsnip model specification
- _fit for a fitted model
- _rec for a recipe
- _wfl for a workflow
- _tune for a tuning object
- _res for a general result

## Data splitting

We typically split data into training and test data sets:

- Training Set: these data are used to estimate model parameters and to pick the values of the complexity parameter(s) for the model.

- Test Set: these data can be used to get an independent assessment of model efficacy. They should not be used during model training.

There are a few different ways to do the split: 
- simple random sampling 
- stratified sampling based on the outcome, by date, or methods that focus on the distribution of the predictors.

- Classification: This would mean sampling within the classes to preserve the distribution of the outcome in the training and test sets.

- Regression: Determine the quartiles of the data set and sample within those artificial groups.

The tidymodels framework make it easy for us to make this split between the traning and testing test.

We are gonna use the function `initial_split` from `rsample`.

> The function initial_split creates a split af the data into a traning and testing set.
initial_split(data, prop = 3/4, strata = NULL, breaks = 4, ...)
Here the function argument is very informativ but the strata is a variable that is used to conduct stratified sampling to create the resamples.

Let us run it in R.


```r
set.seed(4595)

data_split <- initial_split(ames, strata = "Sale_Price")

ames_train <- 
  data_split %>% 
  training()

ames_test  <-
  data_split %>% 
  testing()
```

Let us take a look at the `data_split` object.


```r
data_split
```

```
## <Analysis/Assess/Total>
## <2199/731/2930>
```

What we see is the amount we use as traning - 2199 and the number we use for testing - 731.

## Modelling

It is extreme easy to build model in R specially when you use the tidymodels framwork. 

The process is like a cooking recipe and these steps are gonna be introduced in the following.

### Parsnip in Action

The package  [parsnip](https://parsnip.tidymodels.org/) is the one handling the 
transition from model to model. In this process it is easy to swift between models.

Some of the models type available in parsnip is the following:

- lm() isn't the only way to perform linear regression

- glmnet for regularized regression

- stan for Bayesian regression

- keras for regression using tensorflow

- spark for large data sets

When we are gonna use parsnip at build the model it can be done with 3 steps: 

1) Create a specification


```r
spec <- 
  linear_reg()
```

2) Set the engine


```r
spec <- 
  linear_reg() %>% 
  set_engine("stan",
             chains = 4, 
             iter = 1000)
```

3) Fit the model


```r
model <- 
  linear_reg() %>% 
  set_engine("stan", 
             chains = 4, 
             iter = 1000) %>%
  fit(log10(Sale_Price) ~ Longitude + Latitude, data = ames_train)
coef(model$fit)
```

```
## (Intercept)   Longitude    Latitude 
## -306.283736   -2.029816    2.887020
```

So with this 3 steps you can quickly build models.
The argument regarding the engine can be shfit out quickly for another model or other specification. 

Let us try to shift to another model where we replace the specificaion with `nearest_neighbor` and the engine to `kknn`. Here we are gonna combined the: specification, engine and model fitting into one code chunck.


```r
fit_knn <- 
  nearest_neighbor(neighbors = 5) %>%
  set_mode("regression") %>% 
  set_engine("kknn") %>% 
  fit(log10(Sale_Price) ~ Longitude + Latitude, data = ames_train)
fit_knn
```

```
## parsnip model object
## 
## Fit time:  50ms 
## 
## Call:
## kknn::train.kknn(formula = log10(Sale_Price) ~ Longitude + Latitude,     data = data, ks = min_rows(5, data, 5))
## 
## Type of response variable: continuous
## minimal mean absolute error: 0.06753097
## Minimal mean squared error: 0.009633708
## Best kernel: optimal
## Best k: 5
```

### Preprocessing and Feature  (FE)

Now we have gotten an idea of the power and process of building models in tidymodels we are now gonna touch new methodss to turn our models into more effective models. We are talking about feature engineering where it is maninly the predictors we focus on. In this process one might make operastions that transform the predictors, alternate encodings of a variable and elimination of preditcot (often used in unsipervised).

There can be many reason why to do FE:

Some models (K-NN, SVMs, PLS, neural networks) require that the predictor variables have the same units. Centering and scaling the predictors can be used for this purpose.

Other models are very sensitive to correlations between the predictors and filters or PCA signal extraction can improve the model.

As we'll see in an example, changing the scale of the predictors using a transformation can lead to a big improvement.

In other cases, the data can be encoded in a way that maximizes its effect on the model. Representing the date as the day of the week can be very effective for modeling public transportation data.

Many models cannot cope with missing data so imputation strategies might be necessary.

Development of new features that represent something important to the outcome (e.g. compute distances to public transportation, university buildings, public schools, etc.)

One common procedure for modeling is to create numeric representations of categorical data. This is usually done via dummy variables: a set of binary 0/1 variables for different levels of an R factor.

For example, the Ames housing data contains a predictor called Alley with levels: 'Gravel', 'No_Alley_Access', 'Paved'.

Most dummy variable procedures would make two numeric variables from this predictor that are 1 when the observation has that level, and 0 otherwise.

A zero-variance predictor that has only a single value (zero) would be the result.

Many models (e.g. linear/logistic regression, etc.) would find this numerically problematic and issue a warning and NA values for that coefficient. Trees and similar models would not notice.

There are two main approaches to dealing with this:

Run a filter on the training set predictors prior to running the model and remove the zero-variance predictors.

Recode the factor so that infrequently occurring predictors (and possibly new values) are pooled into an "other" category.

However, model.matrix() and the formula method are incapable of helping you.

Recipes are an alternative method for creating the data frame of predictors for a model. They allow for a sequence of steps that define how data should be handled.

Recall the previous part where we used the formula log10(Sale_Price) ~ Longitude + Latitude? These steps are:

- Assign Sale_Price to be the outcome

- Assign Longitude and Latitude as predictors

- Log transform the outcome


### Recipes and Categorical Predictors

The FE depend on the form of the predictors. If we have a categorical predictor we like mentioned before make it to a dummy variable.

> The function recipe is in the package recipes. It is a descriptiong of waht steps should be applied to the data so it is ready for analysis
recipe(x, ...)



```r
recipe(
  Sale_Price ~ Longitude + Latitude + Neighborhood, 
  data = ames_train) %>%
  step_log(Sale_Price, base = 10) %>%
  step_other(Neighborhood, threshold = 0.05) %>%
  step_dummy(all_nominal()) %>% prep %>% juice()
```

```
## # A tibble: 2,199 x 11
##    Longitude Latitude Sale_Price Neighborhood_College_Creek Neighborhood_Old_To…
##        <dbl>    <dbl>      <dbl>                      <dbl>                <dbl>
##  1     -93.6     42.1       5.24                          0                    0
##  2     -93.6     42.1       5.39                          0                    0
##  3     -93.6     42.1       5.28                          0                    0
##  4     -93.6     42.1       5.29                          0                    0
##  5     -93.6     42.1       5.33                          0                    0
##  6     -93.6     42.1       5.28                          0                    0
##  7     -93.6     42.1       5.37                          0                    0
##  8     -93.6     42.1       5.28                          0                    0
##  9     -93.6     42.1       5.25                          0                    0
## 10     -93.6     42.1       5.26                          0                    0
## # … with 2,189 more rows, and 6 more variables: Neighborhood_Edwards <dbl>,
## #   Neighborhood_Somerset <dbl>, Neighborhood_Northridge_Heights <dbl>,
## #   Neighborhood_Gilbert <dbl>, Neighborhood_Sawyer <dbl>,
## #   Neighborhood_other <dbl>
```

Let us build a recipe. It is normal convention in the tidymodels terminology to calle the recipe for rec.

```r
rec <-
  recipe(Sale_Price ~ Longitude + Latitude + Neighborhood, 
         data = ames_train) %>%
  step_log(Sale_Price, base = 10) %>%
  step_other(Neighborhood, threshold = 0.05) %>%
  step_dummy(all_nominal())
```


In the recipes packages there are three important functions:

![](recipe.png)

So each of the functions has a special focus regarding the tranings or testing set.

Now that we have a preprocessing specification, let's run it on the training set to prepare the recipe:


```r
mod_rec_trained <- 
  rec %>% 
  prep(training = ames_train)
```

Here, the "training" is to determine which levels to lump together and to enumerate the factor levels of the Neighborhood variable.

Now that the recipe has been prepared, we can extract the processed training set from it, with all of the steps applied. To do that, we use juice().

> In prep the steps are being estimated and applied in the traning set. juice() will return the results of a recipe where all steps have been applied.


```r
mod_rec_trained %>% juice()
```

```
## # A tibble: 2,199 x 11
##    Longitude Latitude Sale_Price Neighborhood_College_Creek Neighborhood_Old_To…
##        <dbl>    <dbl>      <dbl>                      <dbl>                <dbl>
##  1     -93.6     42.1       5.24                          0                    0
##  2     -93.6     42.1       5.39                          0                    0
##  3     -93.6     42.1       5.28                          0                    0
##  4     -93.6     42.1       5.29                          0                    0
##  5     -93.6     42.1       5.33                          0                    0
##  6     -93.6     42.1       5.28                          0                    0
##  7     -93.6     42.1       5.37                          0                    0
##  8     -93.6     42.1       5.28                          0                    0
##  9     -93.6     42.1       5.25                          0                    0
## 10     -93.6     42.1       5.26                          0                    0
## # … with 2,189 more rows, and 6 more variables: Neighborhood_Edwards <dbl>,
## #   Neighborhood_Somerset <dbl>, Neighborhood_Northridge_Heights <dbl>,
## #   Neighborhood_Gilbert <dbl>, Neighborhood_Sawyer <dbl>,
## #   Neighborhood_other <dbl>
```

This is what you'd pass on to `fit()` your model.

After model fitting, you'll eventually want to make predictions on new data. But first, you have to reapply all of the pre-processing steps on it. To do that, use `bake()`.

> bake() takes a trained recipe and applies the operations to a data set to create a design matrix.


```r
mod_rec_trained %>%  bake(new_data = ames_test)
```

```
## # A tibble: 731 x 11
##    Longitude Latitude Sale_Price Neighborhood_College_Creek Neighborhood_Old_To…
##        <dbl>    <dbl>      <dbl>                      <dbl>                <dbl>
##  1     -93.6     42.1       5.33                          0                    0
##  2     -93.6     42.1       5.02                          0                    0
##  3     -93.6     42.1       5.27                          0                    0
##  4     -93.6     42.1       5.60                          0                    0
##  5     -93.6     42.1       5.28                          0                    0
##  6     -93.6     42.1       5.17                          0                    0
##  7     -93.6     42.1       5.02                          0                    0
##  8     -93.7     42.1       5.46                          0                    0
##  9     -93.7     42.1       5.44                          0                    0
## 10     -93.7     42.1       5.33                          0                    0
## # … with 721 more rows, and 6 more variables: Neighborhood_Edwards <dbl>,
## #   Neighborhood_Somerset <dbl>, Neighborhood_Northridge_Heights <dbl>,
## #   Neighborhood_Gilbert <dbl>, Neighborhood_Sawyer <dbl>,
## #   Neighborhood_other <dbl>
```

this is passed on to `predict()`

### Juice and bake clarification

One of the things that I found confusion was the difference between juice and bake.
So lets us make sure we are all 100% aligned. 

- juice() is used to get the pre-processed training set (basically for free)
- bake() is used to pre-process a *new* data set

- prep() - training is the entire training set, used to estimate parameters in each step (like means or standard deviations).

- bake() - new_data is data to apply the pre-processing to, using the same estimated parameters from when prep() was called on the training set.

### Interactions effects 

An **interaction** between two predictors indicates that the relationship between the predictors and the outcome cannot be describe using only one of the variables.

For example, let's look at the relationship between the price of a house and the year in which it was built. The relationship appears to be slightly nonlinear, possibly quadratic:


```r
ames_train %>%
  ggplot(aes(x = Year_Built, y = Sale_Price)) + 
  geom_point(alpha = 0.4) +
  scale_y_log10() + 
  geom_smooth(method = "loess") +
  my_theme()
```

```
## `geom_smooth()` using formula 'y ~ x'
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-15-1.png" width="672" />

However... what if we separate this trend based on whether the property has air conditioning or not.


```r
ames_train %>%
  group_by(Central_Air) %>%
  summarise(n = n()) %>%
  mutate(percent = n / sum(n) * 100)
```

```
## # A tibble: 2 x 3
##   Central_Air     n percent
##   <fct>       <int>   <dbl>
## 1 N             141    6.41
## 2 Y            2058   93.6
```


```r
# to get robust linear regression model
ames_train %>%
  ggplot(aes(x = Year_Built, y = Sale_Price)) + 
  geom_point(alpha = 0.4) +
  scale_y_log10() + 
  facet_wrap(~ Central_Air, nrow = 2) +
  geom_smooth(method = "lm") +
  my_theme()
```

```
## `geom_smooth()` using formula 'y ~ x'
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-17-1.png" width="672" />

It appears as though the relationship between the year built and the sale price is somewhat different for the two groups.

When there is no AC, the trend is perhaps flat or slightly decreasing.
With AC, there is a linear increasing trend or is perhaps slightly quadratic with some outliers at the low end.


```r
mod1 <- lm(log10(Sale_Price) ~ Year_Built + Central_Air,                          
           data = ames_train)
mod2 <- lm(log10(Sale_Price) ~ Year_Built + Central_Air + Year_Built:Central_Air, 
           data = ames_train)
anova(mod1, mod2)
```

```
## Analysis of Variance Table
## 
## Model 1: log10(Sale_Price) ~ Year_Built + Central_Air
## Model 2: log10(Sale_Price) ~ Year_Built + Central_Air + Year_Built:Central_Air
##   Res.Df    RSS Df Sum of Sq      F   Pr(>F)    
## 1   2196 42.741                                 
## 2   2195 41.733  1    1.0075 52.993 4.64e-13 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

We first create the dummy variables for the qualitative predictor (Central_Air) then use a formula to create the interaction using the : operator in an additional step:

> step_interact() creates interactions between variables. 


```r
interact_rec <-
  recipe(Sale_Price ~ Year_Built + Central_Air, data = ames_train) %>%
  step_log(Sale_Price) %>%
  step_dummy(Central_Air) %>%
  step_interact(~ starts_with("Central_Air"):Year_Built)

interact_rec %>%
  prep(training = ames_train) %>%
  juice() %>%
  # select a few rows with different values
  slice(153:157)
```

```
## # A tibble: 5 x 4
##   Year_Built Sale_Price Central_Air_Y Central_Air_Y_x_Year_Built
##        <int>      <dbl>         <dbl>                      <dbl>
## 1       1915       11.9             1                       1915
## 2       1912       12.0             1                       1912
## 3       1920       11.7             1                       1920
## 4       1963       11.6             0                          0
## 5       1930       10.9             0                          0
```

## Recipe and models

Let us look at the predictor **Longitude**:


```r
ggplot(ames_train, 
       aes(x = Longitude, y = Sale_Price)) + 
  geom_point(alpha = .5) + 
  geom_smooth(
    method = "lm", 
    formula = y ~ splines::bs(x, 5), 
    se = FALSE
  ) + 
  scale_y_log10() +
  my_theme()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-20-1.png" width="672" />


Splines add nonlinear versions of the predictor to a linear model to create smooth and flexible relationships between the predictor and outcome.

Let us  have a look at the **Latitude**:


```r
ggplot(ames_train, 
       aes(x = Latitude, y = Sale_Price)) + 
  geom_point(alpha = .5) + 
  geom_smooth(
    method = "lm", 
    formula = y ~ splines::ns(x, df = 5), 
    se = FALSE
  ) + 
  scale_y_log10() +
  my_theme()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-21-1.png" width="672" />

## Linear Models Again

We'll add neighborhood in as well and a few other house features.

Our plots suggests that the coordinates can be helpful but probably require a nonlinear representation. We can add these using B-splines with 5 degrees of freedom.

Two numeric predictors are very skewed and could use a transformation (Lot_Area and Gr_Liv_Area).


```r
ames_rec <- 
  recipe(
    Sale_Price ~ Bldg_Type + Neighborhood + Year_Built + 
      Gr_Liv_Area + Full_Bath + Year_Sold + Lot_Area +
      Central_Air + Longitude + Latitude,
    data = ames_train) %>%
  step_log(Sale_Price, base = 10) %>%
  step_BoxCox(Lot_Area, Gr_Liv_Area) %>%
  step_other(Neighborhood, threshold = 0.05)  %>%
  step_dummy(all_nominal()) %>%
  step_interact(~ starts_with("Central_Air"):Year_Built) %>%
  step_ns(Longitude, Latitude, deg_free = 5)
```

Now we will create the linear model:


```r
lm_mod <- 
  linear_reg() %>% 
  set_engine("lm")
```


Apply the recipe:


```r
ames_rec_prep <-
  prep(ames_rec)

ames_rec_prep %>% 
  juice() %>% 
  head() %>% 
  kableExtra::kable()
```

<table>
 <thead>
  <tr>
   <th style="text-align:right;"> Year_Built </th>
   <th style="text-align:right;"> Gr_Liv_Area </th>
   <th style="text-align:right;"> Full_Bath </th>
   <th style="text-align:right;"> Year_Sold </th>
   <th style="text-align:right;"> Lot_Area </th>
   <th style="text-align:right;"> Sale_Price </th>
   <th style="text-align:right;"> Bldg_Type_TwoFmCon </th>
   <th style="text-align:right;"> Bldg_Type_Duplex </th>
   <th style="text-align:right;"> Bldg_Type_Twnhs </th>
   <th style="text-align:right;"> Bldg_Type_TwnhsE </th>
   <th style="text-align:right;"> Neighborhood_College_Creek </th>
   <th style="text-align:right;"> Neighborhood_Old_Town </th>
   <th style="text-align:right;"> Neighborhood_Edwards </th>
   <th style="text-align:right;"> Neighborhood_Somerset </th>
   <th style="text-align:right;"> Neighborhood_Northridge_Heights </th>
   <th style="text-align:right;"> Neighborhood_Gilbert </th>
   <th style="text-align:right;"> Neighborhood_Sawyer </th>
   <th style="text-align:right;"> Neighborhood_other </th>
   <th style="text-align:right;"> Central_Air_Y </th>
   <th style="text-align:right;"> Central_Air_Y_x_Year_Built </th>
   <th style="text-align:right;"> Longitude_ns_1 </th>
   <th style="text-align:right;"> Longitude_ns_2 </th>
   <th style="text-align:right;"> Longitude_ns_3 </th>
   <th style="text-align:right;"> Longitude_ns_4 </th>
   <th style="text-align:right;"> Longitude_ns_5 </th>
   <th style="text-align:right;"> Latitude_ns_1 </th>
   <th style="text-align:right;"> Latitude_ns_2 </th>
   <th style="text-align:right;"> Latitude_ns_3 </th>
   <th style="text-align:right;"> Latitude_ns_4 </th>
   <th style="text-align:right;"> Latitude_ns_5 </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:right;"> 1958 </td>
   <td style="text-align:right;"> 6.456179 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 2010 </td>
   <td style="text-align:right;"> 18.93439 </td>
   <td style="text-align:right;"> 5.235528 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 1958 </td>
   <td style="text-align:right;"> 0.0000418 </td>
   <td style="text-align:right;"> 0.4678536 </td>
   <td style="text-align:right;"> 0.4360063 </td>
   <td style="text-align:right;"> 0.1530795 </td>
   <td style="text-align:right;"> -0.0569811 </td>
   <td style="text-align:right;"> 0.0000000 </td>
   <td style="text-align:right;"> 0.1366082 </td>
   <td style="text-align:right;"> 0.5589852 </td>
   <td style="text-align:right;"> 0.2253646 </td>
   <td style="text-align:right;"> 0.0790420 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 1968 </td>
   <td style="text-align:right;"> 6.824563 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2010 </td>
   <td style="text-align:right;"> 18.09966 </td>
   <td style="text-align:right;"> 5.387390 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 1968 </td>
   <td style="text-align:right;"> 0.0000000 </td>
   <td style="text-align:right;"> 0.4021351 </td>
   <td style="text-align:right;"> 0.4763868 </td>
   <td style="text-align:right;"> 0.1732659 </td>
   <td style="text-align:right;"> -0.0517878 </td>
   <td style="text-align:right;"> 0.0001229 </td>
   <td style="text-align:right;"> 0.1975667 </td>
   <td style="text-align:right;"> 0.5685334 </td>
   <td style="text-align:right;"> 0.1981186 </td>
   <td style="text-align:right;"> 0.0356585 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 1997 </td>
   <td style="text-align:right;"> 6.619025 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2010 </td>
   <td style="text-align:right;"> 18.82719 </td>
   <td style="text-align:right;"> 5.278525 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 1997 </td>
   <td style="text-align:right;"> 0.3684027 </td>
   <td style="text-align:right;"> 0.6049363 </td>
   <td style="text-align:right;"> 0.0143487 </td>
   <td style="text-align:right;"> 0.0112095 </td>
   <td style="text-align:right;"> -0.0064332 </td>
   <td style="text-align:right;"> 0.0000000 </td>
   <td style="text-align:right;"> 0.0017056 </td>
   <td style="text-align:right;"> 0.0832997 </td>
   <td style="text-align:right;"> 0.3901965 </td>
   <td style="text-align:right;"> 0.5247983 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 1998 </td>
   <td style="text-align:right;"> 6.606687 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2010 </td>
   <td style="text-align:right;"> 17.72790 </td>
   <td style="text-align:right;"> 5.291147 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 1998 </td>
   <td style="text-align:right;"> 0.3680764 </td>
   <td style="text-align:right;"> 0.6052355 </td>
   <td style="text-align:right;"> 0.0144074 </td>
   <td style="text-align:right;"> 0.0112031 </td>
   <td style="text-align:right;"> -0.0064295 </td>
   <td style="text-align:right;"> 0.0000000 </td>
   <td style="text-align:right;"> 0.0019644 </td>
   <td style="text-align:right;"> 0.0934633 </td>
   <td style="text-align:right;"> 0.3877505 </td>
   <td style="text-align:right;"> 0.5168219 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 2001 </td>
   <td style="text-align:right;"> 6.461595 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2010 </td>
   <td style="text-align:right;"> 15.50052 </td>
   <td style="text-align:right;"> 5.329398 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 2001 </td>
   <td style="text-align:right;"> 0.1713910 </td>
   <td style="text-align:right;"> 0.7464424 </td>
   <td style="text-align:right;"> 0.0722385 </td>
   <td style="text-align:right;"> 0.0231940 </td>
   <td style="text-align:right;"> -0.0133111 </td>
   <td style="text-align:right;"> 0.0000000 </td>
   <td style="text-align:right;"> 0.0000076 </td>
   <td style="text-align:right;"> -0.0980697 </td>
   <td style="text-align:right;"> 0.4326474 </td>
   <td style="text-align:right;"> 0.6654146 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 1992 </td>
   <td style="text-align:right;"> 6.426012 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2010 </td>
   <td style="text-align:right;"> 15.55210 </td>
   <td style="text-align:right;"> 5.282169 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 1992 </td>
   <td style="text-align:right;"> 0.1725111 </td>
   <td style="text-align:right;"> 0.7459434 </td>
   <td style="text-align:right;"> 0.0716869 </td>
   <td style="text-align:right;"> 0.0230209 </td>
   <td style="text-align:right;"> -0.0132118 </td>
   <td style="text-align:right;"> 0.0000000 </td>
   <td style="text-align:right;"> 0.0020818 </td>
   <td style="text-align:right;"> 0.0977687 </td>
   <td style="text-align:right;"> 0.3867111 </td>
   <td style="text-align:right;"> 0.5134384 </td>
  </tr>
</tbody>
</table>


```r
lm_fit <- 
  lm_mod %>% 
  fit(Sale_Price ~ ., 
      data = juice(ames_rec_prep))

glance(lm_fit$fit)
```

```
## # A tibble: 1 x 12
##   r.squared adj.r.squared  sigma statistic p.value    df logLik    AIC    BIC
##       <dbl>         <dbl>  <dbl>     <dbl>   <dbl> <dbl>  <dbl>  <dbl>  <dbl>
## 1     0.802         0.799 0.0800      303.       0    29  2448. -4834. -4657.
## # … with 3 more variables: deviance <dbl>, df.residual <int>, nobs <int>
```

Now we can apply the train recipe to our testing set.


```r
ames_test_processed <-
  bake(ames_rec_prep, ames_test, all_predictors())
```

Let us try to make some predictions:

> At the same time I am gonna use `yardstick` to collect metrices to look at how well the model is performing.


```r
predict(lm_fit, new_data = ames_test_processed) %>% 
  bind_cols(bake(ames_rec_prep, new_data = ames_test) %>% select(Sale_Price)) %>% 
  metrics(Sale_Price, .pred)
```

```
## # A tibble: 3 x 3
##   .metric .estimator .estimate
##   <chr>   <chr>          <dbl>
## 1 rmse    standard      0.0769
## 2 rsq     standard      0.800 
## 3 mae     standard      0.0555
```


The workflows package enables a handy type of object that can bundle pre-processing and models together.

## Resampling 



These are additional data splitting schemes that are applied to the training set and are used for estimating model performance.

They attempt to simulate slightly different versions of the training set. These versions of the original are split into two model subsets:

- The analysis set is used to fit the model (analogous to the training set).
- Performance is determined using the assessment set.

This process is repeated many times.

![](resampling.png)

All resampling methods repeat this process multiple times:

![](wf-resampling.png)

The final resampling estimate is the average of all of the estimated metrics (e.g. RMSE, etc).

### V-Fold Cross-Validation

Here, we randomly split the training data into V distinct blocks of roughly equal size (AKA the "folds").

We leave out the first block of analysis data and fit a model.

This model is used to predict the held-out block of assessment data.

We continue this process until we've predicted all V assessment blocks

![](vfold.png)

The final performance is based on the hold-out predictions by averaging the statistics from the V blocks.

V is usually taken to be 5 or 10 and leave-one-out cross-validation has each sample as a block.

Repeated CV can be used when training set sizes are small. 5 repeats of 10-fold CV averages for 50 sets of metrics.


### Resampling results

The goal of resampling is to produce a single estimate of performance for a model.

Even though we end up estimating V models (for V-fold CV), these models are discarded after we have our performance estimate.

Resampling is basically an emprical simulation system used to understand how well the model would work on new data.

### Cross-Validating Using {rsample}

rsample has a number of resampling functions built in. One is vfold_cv(), for performing V-Fold cross-validation like we've been discussing.


```r
set.seed(2453)
cv_splits <- vfold_cv(ames_train) #10-fold is default
cv_splits
```

```
## #  10-fold cross-validation 
## # A tibble: 10 x 2
##    splits             id    
##    <list>             <chr> 
##  1 <split [1979/220]> Fold01
##  2 <split [1979/220]> Fold02
##  3 <split [1979/220]> Fold03
##  4 <split [1979/220]> Fold04
##  5 <split [1979/220]> Fold05
##  6 <split [1979/220]> Fold06
##  7 <split [1979/220]> Fold07
##  8 <split [1979/220]> Fold08
##  9 <split [1979/220]> Fold09
## 10 <split [1980/219]> Fold10
```


Each individual split object is similar to the initial_split() example.

Use **analysis()** to extract the resample's data used for the fitting process.

Use **assessment()** to extract the resample's data used for the performance process.


```r
cv_splits$splits[[1]]
```

```
## <Analysis/Assess/Total>
## <1979/220/2199>
```


```r
cv_splits$splits[[1]] %>% 
  analysis() %>%
  dim()
```

```
## [1] 1979   74
```


```r
cv_splits$splits[[1]] %>% 
  assessment() %>%
  dim()
```

```
## [1] 220  74
```

### K-Nearest Neighbors Model

K-nearest neighbors stores the training set (including the outcome).

When a new sample is predicted, K training set points are found that are most similar to the new sample being predicted.

The predicted value for the new sample is some summary statistic of the neighbors, usually:

- the mean for regression, or
- the mode for classification.

Let's try a 5-neighbor model on the Ames data.

### Resampling a 5-NN model


```r
knn_mod <- 
  nearest_neighbor(neighbors = 5) %>% 
  set_engine("kknn") %>% 
  set_mode("regression")

knn_wfl <- 
  workflow() %>% 
  add_model(knn_mod) %>% 
  add_formula(log10(Sale_Price) ~ Longitude + Latitude)

# If we were compute this model with the training set:
fit(knn_wfl, data = ames_train)
```

```
## ══ Workflow [trained] ══════════════════════════════════════════════════════════
## Preprocessor: Formula
## Model: nearest_neighbor()
## 
## ── Preprocessor ────────────────────────────────────────────────────────────────
## log10(Sale_Price) ~ Longitude + Latitude
## 
## ── Model ───────────────────────────────────────────────────────────────────────
## 
## Call:
## kknn::train.kknn(formula = ..y ~ ., data = data, ks = min_rows(5,     data, 5))
## 
## Type of response variable: continuous
## minimal mean absolute error: 0.06753097
## Minimal mean squared error: 0.009633708
## Best kernel: optimal
## Best k: 5
```

Let's repeat that but using each of the 10 analysis sets:


```r
knn_res <- 
  cv_splits %>%
  mutate(
    workflows = map(
      splits, 
      ~ fit(knn_wfl, data = analysis(.x))
    )
  ) 
knn_res
```

```
## #  10-fold cross-validation 
## # A tibble: 10 x 3
##    splits             id     workflows 
##    <list>             <chr>  <list>    
##  1 <split [1979/220]> Fold01 <workflow>
##  2 <split [1979/220]> Fold02 <workflow>
##  3 <split [1979/220]> Fold03 <workflow>
##  4 <split [1979/220]> Fold04 <workflow>
##  5 <split [1979/220]> Fold05 <workflow>
##  6 <split [1979/220]> Fold06 <workflow>
##  7 <split [1979/220]> Fold07 <workflow>
##  8 <split [1979/220]> Fold08 <workflow>
##  9 <split [1979/220]> Fold09 <workflow>
## 10 <split [1980/219]> Fold10 <workflow>
```

Let us  look at the predictions for each model. 

> map2_dfr is path of the purrr familiy over loop functions.


```r
knn_pred <- 
  map2_dfr(
    knn_res$workflows, 
    knn_res$splits,     
    ~ predict(.x, assessment(.y)),         
    .id = "fold"
  )                                   
prices <- 
  map_dfr(
    knn_res$splits,  
    ~ assessment(.x) %>% select(Sale_Price)
  ) %>% 
  mutate(Sale_Price = log10(Sale_Price))

rmse_estimates <-
  knn_pred %>%
  bind_cols(prices) %>% 
  group_by(fold) %>% 
  do(rmse = rmse(., Sale_Price, .pred)) %>% 
  unnest(cols = c(rmse)) 
mean(rmse_estimates$.estimate)
```

```
## [1] 0.09851353
```

So this is a lot of work but there is a way around this.

### Easy resampling using the {tune} package

There is a fit_resamples() function in the tune package that does all of this for you.

> fit_resamples computes a set of metrices across samples.


```r
easy_eval <-
  fit_resamples(knn_wfl, 
                resamples = cv_splits, 
                control = control_resamples(save_pred = TRUE))
```


```r
collect_metrics(easy_eval)
```

```
## # A tibble: 2 x 6
##   .metric .estimator   mean     n std_err .config             
##   <chr>   <chr>       <dbl> <int>   <dbl> <chr>               
## 1 rmse    standard   0.0985    10 0.00298 Preprocessor1_Model1
## 2 rsq     standard   0.698     10 0.0153  Preprocessor1_Model1
```

So instead of collect the result ourselve we can use the tune functions fit_resamples to do the hard work for us. 

## Model Tuning

### Tuning Parameters

There are some models with parameters that cannot be directly estimated from the data.

For example:

- The number of neighbors in a K-NN models.

- The depth of a classification tree.

- The link function in a generalized linear model (e.g. logit, probit, etc).

- The covariance structure in a linear mixed model.

### Tuning Parameters and Overfitting

Overfitting occurs when a model inappropriately picks up on trends in the training set that do not generalize to new samples.

When this occurs, assessments of the model based on the training set can show good performance that does not reproduce in future samples.

For example, K = 1 neighbors is much more likely to overfit the data than larger values since they average more values.

Also, how would you evaluate this model by re-predicting the training set? Those values would be optimistic since one of your neighbors is always you.

Unsurprisingly, we will evaluate a tuning parameter by fitting a model on one set of data and assessing it with another.

Grid search uses a pre-defined set of candidate tuning parameter values and evaluates their performance so that the best values can be used in the final model.

We'll use resampling to do this. If there are B resamples and C tuning parameter combinations, we end up fitting B×C models (but these can be done in parallel).

tune has more general functions for tuning models. There are two main strategies used:

- Grid search (as shown above) where all of the candidate models are known at the start. We pick the best of these.

- Iterative search where each iteration finds novel tuning parameter values to evaluate.

Both have their advantages and disadvantages. At first, we will focus on grid search.

Usually combinatorial representation of vectors of tuning parameter values. Note that:

The number of values don't have to be the same per parameter.

The values can be regular on a transformed scale (e.g. log-10 for penalty).

Quantitative and qualitative parameters can be combined.

As the number of parameters increase, the curse of dimensionality kicks in.

Thought to be really inefficient but not in all cases (see the sub-model trick and multi_predict()).

Bad when performance plateaus over a range of one or more parameters.

### About Regular Grids

Usually combinatorial representation of vectors of tuning parameter values. Note that:

The number of values don't have to be the same per parameter.

The values can be regular on a transformed scale (e.g. log-10 for penalty).

Quantitative and qualitative parameters can be combined.

As the number of parameters increase, the curse of dimensionality kicks in.

Thought to be really inefficient but not in all cases (see the sub-model trick and multi_predict()).

Bad when performance plateaus over a range of one or more parameters.


```r
glmn_param <-
  parameters(penalty(), mixture())
```


```r
glmn_grid <- 
  grid_regular(glmn_param, levels = c(10, 5))
glmn_grid %>% slice(1:4)
```

```
## # A tibble: 4 x 2
##         penalty mixture
##           <dbl>   <dbl>
## 1 0.0000000001        0
## 2 0.00000000129       0
## 3 0.0000000167        0
## 4 0.000000215         0
```

### Tagging Tuning parameters

To tune the model, the first step is to tag the parameters that will be optimized. tune::tune() can be used to do this:


```r
knn_mod <- 
  nearest_neighbor(neighbors = tune(),
                   weight_func = tune()) %>% 
  set_engine("kknn") %>% 
  set_mode("regression")

wf <- 
  workflow() %>% 
  add_recipe(ames_rec) %>% 
  add_model(knn_mod)
```


`parameter()` can detect these arguments:


```r
parameters(knn_mod)
```

```
## Collection of 2 parameters for tuning
## 
##   identifier        type    object
##    neighbors   neighbors nparam[+]
##  weight_func weight_func dparam[+]
```

We can also name the parameter:


```r
nearest_neighbor(neighbors = tune("K"),
                 weight_func = tune("weights")) %>% 
  set_engine("kknn") %>% 
  set_mode("regression") %>% 
  parameters()
```

```
## Collection of 2 parameters for tuning
## 
##  identifier        type    object
##           K   neighbors nparam[+]
##     weights weight_func dparam[+]
```

This mainly comes in handy when the same parameter type shows up in two different places (an example is shown later).

Recipe steps can also have tune() in their arguments.

### Grid search

Let's tune the model over a regular grid and optimize neighbors and weight_func. First we make the grid, then call tune_grid():



```r
set.seed(522)
knn_grid <- knn_mod %>% 
  parameters() %>% 
  grid_regular(levels = c(15, 5))
ctrl <- control_grid(verbose = TRUE)

wf <- 
  workflow() %>% 
  add_model(knn_mod) %>% 
  add_recipe(ames_rec)

knn_tune <- tune_grid(
  object = wf,
  resamples = cv_splits,
  grid = knn_grid,
  control = ctrl
)
```

### Resampled Performance Estimates

To get the overall resampling estimate (averaged over folds) for each parameter combination:


```r
show_best(knn_tune, metric = "rmse", maximize = FALSE)
```

```
## Warning: The `maximize` argument is no longer needed. This value was ignored.
```

```
## # A tibble: 5 x 8
##   neighbors weight_func .metric .estimator   mean     n std_err .config         
##       <int> <chr>       <chr>   <chr>       <dbl> <int>   <dbl> <chr>           
## 1         8 triangular  rmse    standard   0.0816    10 0.00281 Preprocessor1_M…
## 2         7 triangular  rmse    standard   0.0817    10 0.00278 Preprocessor1_M…
## 3         6 triangular  rmse    standard   0.0817    10 0.00279 Preprocessor1_M…
## 4         9 triangular  rmse    standard   0.0818    10 0.00283 Preprocessor1_M…
## 5        11 triangular  rmse    standard   0.0819    10 0.00299 Preprocessor1_M…
```


```r
best <-
  select_best(knn_tune, metric = "rmse", maximize = FALSE)
```

```
## Warning: The `maximize` argument is no longer needed. This value was ignored.
```

```r
best
```

```
## # A tibble: 1 x 3
##   neighbors weight_func .config              
##       <int> <chr>       <chr>                
## 1         8 triangular  Preprocessor1_Model53
```


```r
ames_rec_final <- 
  prep(ames_rec)
knn_mod_final <-
  finalize_model(knn_mod, best)
knn_mod_final
```

```
## K-Nearest Neighbor Model Specification (regression)
## 
## Main Arguments:
##   neighbors = 8
##   weight_func = triangular
## 
## Computational engine: kknn
```

We can now create a final workflow with the best model and here we can use the function `last_fit`.

> the function last_fit(), takes the splits and emulate the process on the whole data but evalutate the results on the test set. 


```r
final_wf <- 
  workflow() %>%
  add_recipe(ames_rec_final) %>%
  add_model(knn_mod_final)

final_res <- 
  final_wf %>%
  last_fit(data_split)
```

```
## ! train/test split: preprocessor 1/1, model 1/1 (predictions): some 'x' values beyond bounda...
```

```r
final_res %>%
  collect_metrics()
```

```
## # A tibble: 2 x 4
##   .metric .estimator .estimate .config             
##   <chr>   <chr>          <dbl> <chr>               
## 1 rmse    standard      0.0867 Preprocessor1_Model1
## 2 rsq     standard      0.746  Preprocessor1_Model1
```


WeYou can also extract the test set predictions themselves using the `collect_predictions()` function. 

> collect_predictions() is used to obtain and format results produced by tuning functions.


```r
final_res %>% 
  collect_predictions()
```

```
## # A tibble: 731 x 5
##    id               .pred  .row Sale_Price .config             
##    <chr>            <dbl> <int>      <dbl> <chr>               
##  1 train/test split  5.19     1       5.33 Preprocessor1_Model1
##  2 train/test split  5.07     2       5.02 Preprocessor1_Model1
##  3 train/test split  5.22    12       5.27 Preprocessor1_Model1
##  4 train/test split  5.42    18       5.60 Preprocessor1_Model1
##  5 train/test split  5.23    21       5.28 Preprocessor1_Model1
##  6 train/test split  5.13    24       5.17 Preprocessor1_Model1
##  7 train/test split  4.94    31       5.02 Preprocessor1_Model1
##  8 train/test split  5.36    40       5.46 Preprocessor1_Model1
##  9 train/test split  5.38    42       5.44 Preprocessor1_Model1
## 10 train/test split  5.33    44       5.33 Preprocessor1_Model1
## # … with 721 more rows
```


## Fitting and using your final model

The previous section evaluated the model trained on the training data using the testing data. But once you’ve determined your final model, you often want to train it on your full dataset and then use it to predict the response for new data.

If you want to use your model to predict the response for new observations, you need to use the fit() function on your workflow and the dataset that you want to fit the final model on (e.g. the complete training + testing dataset).


```r
final_model <-
  fit(final_wf, ames)
```

And now when we get new data we can make an prediction on the sale price.
