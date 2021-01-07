---
title: Cargo classification
author: Lucas Bagge
date: '2021-01-07'
slug: cargo-classification
categories:
  - Classification
tags:
  - Ørsted
  - Logistic regression
subtitle: ''
summary: ''
authors: []
lastmod: '2021-01-07T10:01:17+01:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---







## Introduction

Amagerværket has two units 1 and 4 which produces heat and power using
respectively wood pellets and wood chips as fuel. The fuel arrives to the plant
by vessel and it is possible to track the arrivals through Copenhagen Malmö (CM)
Port's website. However, it is not directly possible to identify if the cargo of
a given ship is either wood pellets or wood chips. Consequently, the objective
is to build a model capable of classifying the vessels.

With the newly inaugurated wood chips fired unit 4 at Amagerværket and
expectedly using well above 1 Mt of wood chips per year, the plant has
suddenly---and by a substantial margin---become the largest player in the market
for wood chips delivered by vessel. To maintain market visibility and have the
best possible information to act in the biomass market, it is of importance to
BIO Markets to have information about the arrival and consumption of biomass at
Amagerværket.

The objective of this analysis is

> build a model that accurately and automatically classify vessels arriving to
> Amagerværket into groups according to cargo type

## The data set

For building our model we need to collect the [cmport
data](https://www.cmport.com/terminals/ships-in-port/). Here we have used web
scraping techniques to extract the data that is essential for us to build our
model.

The site is being updated on a regular basis, so our script that scrapes the
data is running three times a day.

As a first analysis task we need to collect information about the port and if
there were information on how we could classify the cargoes. We got great insight 
from colleagues that have tracked the cargoes for a long time and they gave us 
the information that `quay`, and `arriving from` could help us
differentiated between the cargoes.

We need to explore the data for understanding it better and look at the outcome
variable and the predictor we are considering being of most used for us.


```r
gross_feature_set %>% kbl() %>% kable_styling()
```

<table class="table" style="margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;"> feature </th>
   <th style="text-align:left;"> description </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> eta </td>
   <td style="text-align:left;"> Estimated time of arrival </td>
  </tr>
  <tr>
   <td style="text-align:left;"> etd </td>
   <td style="text-align:left;"> Estimated time of departure </td>
  </tr>
  <tr>
   <td style="text-align:left;"> name </td>
   <td style="text-align:left;"> Vessel name </td>
  </tr>
  <tr>
   <td style="text-align:left;"> company </td>
   <td style="text-align:left;"> The shipping company carrying out the charter </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from </td>
   <td style="text-align:left;"> The location from the vessel is arriving (not necessarily the loading port) </td>
  </tr>
  <tr>
   <td style="text-align:left;"> nationality </td>
   <td style="text-align:left;"> The country under whose laws the vessel is registered or licensed </td>
  </tr>
  <tr>
   <td style="text-align:left;"> vessel_type </td>
   <td style="text-align:left;"> The types os the vessel arriving </td>
  </tr>
  <tr>
   <td style="text-align:left;"> purpose </td>
   <td style="text-align:left;"> Give us the purpose of the vessel </td>
  </tr>
  <tr>
   <td style="text-align:left;"> call_no </td>
   <td style="text-align:left;"> CM Port call number </td>
  </tr>
  <tr>
   <td style="text-align:left;"> length </td>
   <td style="text-align:left;"> Length of the vessel </td>
  </tr>
  <tr>
   <td style="text-align:left;"> width </td>
   <td style="text-align:left;"> Width of the vessel </td>
  </tr>
  <tr>
   <td style="text-align:left;"> brt </td>
   <td style="text-align:left;"> Gross Register Tonnage (GRT) of the vessel </td>
  </tr>
  <tr>
   <td style="text-align:left;"> imo </td>
   <td style="text-align:left;"> The international Maritime Organization number </td>
  </tr>
  <tr>
   <td style="text-align:left;"> quay </td>
   <td style="text-align:left;"> The quay at Amagerværket to which the vessel is assigned </td>
  </tr>
</tbody>
</table>

The available dataset consists of `\(N = 129\)` 
observations of vessel arrivals---also named port calls or just calls---to
Amagerværket each with 129 features. In the
current context not every feature is important, but those deemed important for
our model are:

* `cargo`: Specifies if the vessel carries wood chips (WC) or wood pellets (WP)
* `arriving_from`: The port/city where the cargo is being loaded
* `quay`: The quay at Amagerværket where the cargo is discharged

The `cargo` variable is our outcome variable and the one we want to classify.

### The cargo

Here we can see that there is 105 ships with WC and WP with 24. We don´t have an
equal amount of both cargoes and that is something that need to be considered in
the preprocessing step.


```r
vessels %>% 
  count(cargo) %>%   
  ggplot(aes(x = cargo, y = n, fill = cargo)) +
  geom_col() +
  geom_text(aes(label = n), nudge_y = 3) +
  labs(y = NULL, x = NULL,
       title = "Dependent variable based on cargo",
       subtitle = "Number of cargoes containing either WP or WC")
```

<img src="/post/2021-01-07-cargo-classification/index.en_files/figure-html/unnamed-chunk-3-1.svg" width="672" />

### The quay at Amagerværket

One important predictor is `quay`. It indicate where the vessel anchor at the
port. Here we have four possibilities:

- c835
- c836
- c837
- c838


```r
tbl <- addmargins(table(vessels$cargo, vessels$quay))

tbl %>%
  as.data.frame() %>%
  pivot_wider(names_from = Var2, values_from = Freq) %>%
  kbl(col.names = c("", colnames(tbl)),
      caption = "Cross tabulation of cargo against quay.") %>%
  kable_styling()
```

From looking at the pattern in the data and talking to colleagues there is
some of the quay that is a clear indicator of if the vessel is a WP or WC.


```r
vessels %>%
  count(quay) %>%
  ggplot(aes(x = quay, y = n, fill = quay)) +
  geom_col(position = "dodge") +
  geom_text(aes(label = n), nudge_y = 3) +
  labs(y = NULL, x = NULL,
       title = "The Quay is a possible predictor", 
       subtitle = "Number of cargoes arriving to the different quay")
```

<img src="/post/2021-01-07-cargo-classification/index.en_files/figure-html/unnamed-chunk-5-1.svg" width="672" />

Here `c838` is the quay where WP is being delivered and `c836` plus `c836` is
mainly WC. 


```r
vessels %>% 
  count(quay, cargo) %>% 
  ggplot(aes(reorder_within(quay, n, cargo), n, fill = quay)) +
  facet_grid(rows = vars(cargo), scales = 'free_y') +
  geom_col(position = "dodge") +
  geom_text(aes(label = n), nudge_y = 1.9) +
  coord_flip() +
  scale_x_reordered() +
  labs(title = "The quay indicate what the cargo is delivering", 
       subtitle = "Number of cargoes arriving groped by WC or WP",
       y = "", x = "") +
  theme(legend.title = element_blank())
```

<img src="/post/2021-01-07-cargo-classification/index.en_files/figure-html/unnamed-chunk-6-1.svg" width="672" />

From the above plots it is clear that we can from the quay seperat what the 
vessel is delivering. For c837 this can be considered a joker. It lies between 
the end quay, so our hypothesis is that it can be considered as a waiting
quay where it can either be WC or WP.

### The loading port of the cargo

Another important predictor we want to include in the models framework is 
`arriving_from`. This indicate where the vessel is comming from.
Here we want to make some investigation if we can use this as a predictor. 


```r
vessels %>% 
  group_by(arriving_from, cargo) %>% 
  count(arriving_from, cargo)  %>% 
  ungroup() %>% 
  mutate(across(where(is.character), as.factor)) %>% 
  ggplot(aes(reorder_within(arriving_from, n, cargo), n, fill = cargo)) +
  geom_col(position = "dodge") +
  geom_text(aes(label = n), nudge_y = 0.9) +
  facet_wrap(~cargo, scales = "free_y") +
  coord_flip() +
  scale_x_reordered() +
  scale_fill_manual(values=c("#4099DA", "#E85757")) + 
  labs(title = "Arrival port",
       subtitle = "Number of cargoes grouped by arriving from",
       y = "", x = "") +
  theme(legend.title = element_blank())
```

<img src="/post/2021-01-07-cargo-classification/index.en_files/figure-html/unnamed-chunk-7-1.svg" width="672" />


## Methodology

For solving this supervised classifications task there is different models 
that can be used and we did try out a logistic regression, KNN and decision tree 
model. In the end we went with the logistic regression for classification. 

In the following section we want to work though the essentiel idea behind the
method.

### The logistic regression model

Logistic regression is a very popular model for classification problems and is
one of the most commonly used Machine Learning algorithms. It is easy
to implement and is often used in settings where the dataset is small or as a
baseline for any binary classification problem.

The method is used when the outcome variable is dichotomous in nature. It is a
special case of the general linear regression model when the outcome variable is
binary.

Let `\(y\)` be our binary target with `\(y = 1\)` if the vessel carries WC (the WC
class) and `\(y = 0\)` if the cargo is WP (the WP class). Further, let `\(x\)` be a
`\(M\)`-vector of features of the observed port calls. Then, given some available
information `\(x\)`, our interest lies in identifying the probability of a WC vessel
and by implication also the probability of a WP vessel. Hence,
$$
\Pr(y = 1 \mid x) = \pi^c(x) = 1 - \Pr(y = 0 \mid x) = 1 - \pi^p(x)
$$
where `\(\pi^c(x)\)` and `\(\pi^p(x)\)` are the conditional probabilities of observing
respectively a WC or a WP vessel. The probability mass function for this
Bernoulli distribution is
$$
p(y \mid x) = \pi^c(x) \cdot \bigl(1 - \pi^c(x)\bigr)^{1-y}.
$$

In order to proceed from here, one has to make assumptions about the structure
of the probability `\(\pi^c(x)\)`. The assumption used in the logistic regression
model is to proceed with a logistic function defined by
$$
\sigma(z) = \frac{1}{1 + \exp(-z)} = \frac{\exp(z)}{\exp(z) + 1},
$$
with
$$
z = \phi(x)' w
$$
and finally impose
$$
\pi^c(x) = \sigma(\phi(x)' w).
$$

This equation tells that the crux of the logistic regression model is the
mapping of `\(z = \phi(x)' w\)` to a "probability" through the logistic function.

The logistic function gives an S shape curvature. It takes any real vauled
number and map it into a value between 0 and 1. Here we have in mind of how the
probability should be interpreted. If we get an value 0.75 this mean that there
is 75 percent change that a cargo is WC (if WC is baseline).


```r
curve({exp(x) / (1 + exp(x))}, -10, 10, xlab = "", ylab = "",
      main = "The logistic function mapping z to the unit interval")
abline(v = 0, lty = "dashed")
```

<img src="/post/2021-01-07-cargo-classification/index.en_files/figure-html/unnamed-chunk-8-1.svg" width="672" />

For estimated the parameter logistic regression relies on Maximum likelihood
estimation (MLE). By this estimation technique the parameter is determined by
maximizing the log likelihood function. Here we need to use a optimization
algorithm for finding `\(w\)`. Often one uses Gradient acent for this purpose.

The coefficents should be read as log of odds so we predicts the probability of
occurrence of a binary event.

## Analysis & results

For implement the logistic regression we are gonna use the software program `R`
and it machine learning framework `tidymodels`. tidymodels has its own unique way
of structuring a modelling problem and we follows that workflow.


### Model formulation and data preprocessing

We split our data set in two sets:

* the *training set*: This is the bigger part of the data which we use to
  train/estimate of our models
* the *test set*: This data we hold out to use for evaluation of our trained
  models

  Further, we divide our training set into five folds to use for evaluation of
  the models using cross-validation.[^mdl-eval-cv]

  [^mdl-eval-cv]: We never use a model's fit to the training set to evaluate the
  model's performance.
  

```r
set.seed(4595)

df <- vessels %>% select(cargo, arriving_from, quay)
df_split <- initial_split(df, prop = 0.75, strata = cargo)
df_train <- training(df_split)
df_test <- testing(df_split)
df_train_folds <- vfold_cv(df_train, v = 5, strata = "cargo") # FIXME 
```

Two model specifications are formulated. One where the cargo is predicted using
the quay, and one model where the cargo is predicted using the information about
the port where the vessel is arriving from.


```r
fml_1 <- formula(cargo ~ quay)
fml_2 <- formula(cargo ~ arriving_from)
```

In the prepoceesion step we gonna use `recipe` which is part of tidymodels. Here
we gonna specify what we want to make of changes to the predictors and handling 
other issues. In that step we also could implment a method for solving the
imbalanced dataset problem, but because it wont give a better result we don´t 
add this step'. 


```r
lr_1_rec <-
  recipe(fml_1, data = df_train) %>% step_dummy(all_nominal(), -all_outcomes())  
lr_2_rec <-
  recipe(fml_2, data = df_train) %>% step_dummy(all_nominal(), -all_outcomes())  
```


### Logistic regression model training

We train a logistic regression model and use five-fold cross-validation to
evaluate the model fit. Here we gonna use the recipe from earlier and that and
the model to `workflow` which is handling the information so we easy can make
predictions. 


```r
lr_mod <- logistic_reg() %>% set_engine("glm")
lr_wfl_1 <- workflow() %>% add_model(lr_mod) %>% add_recipe(lr_1_rec)
lr_wfl_2 <- workflow() %>% add_model(lr_mod) %>% add_recipe(lr_2_rec)
lr_wfls <- list(quay = lr_wfl_1, arriving_from = lr_wfl_2)

lr_fits_rs <-
  lr_wfls %>%
  imap(~ {
    fit_resamples(
      .x,
      resamples = df_train_folds,
      control = control_resamples(save_pred = TRUE)
    )
  })
```

```
## 
## Attaching package: 'rlang'
```

```
## The following objects are masked from 'package:purrr':
## 
##     %@%, as_function, flatten, flatten_chr, flatten_dbl, flatten_int,
##     flatten_lgl, flatten_raw, invoke, list_along, modify, prepend,
##     splice
```

```
## The following object is masked from 'package:magrittr':
## 
##     set_names
```

```
## 
## Attaching package: 'vctrs'
```

```
## The following object is masked from 'package:tibble':
## 
##     data_frame
```

```
## The following object is masked from 'package:dplyr':
## 
##     data_frame
```

```
## ! Fold1: preprocessor 1/1, model 1/1 (predictions): prediction from a rank-defici...
```

```
## ! Fold2: preprocessor 1/1, model 1/1 (predictions): prediction from a rank-defici...
```

```
## ! Fold3: preprocessor 1/1, model 1/1 (predictions): prediction from a rank-defici...
```

```
## ! Fold4: preprocessor 1/1, model 1/1 (predictions): prediction from a rank-defici...
```

```
## ! Fold5: preprocessor 1/1, model 1/1 (predictions): prediction from a rank-defici...
```

For the evaluation of the model fit we use the model *accuracy* and the
*Receiver Operating characteristic (ROC)* curve which plots the True Positive
Rate (the sensitivity) against the False Positive Rate (1 - specificity/true
negative rate). These measures are defined 

$$
`\begin{aligned}
\mathrm{accuracy} &= \frac{\mathrm{TP} + \mathrm{TN}}{N} \\
\text{sensitivity} &= \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}} \\
\text{specificity} &= \frac{\mathrm{TN}}{\mathrm{TN} + \mathrm{FP}}
\end{aligned}`
$$

where `\(\mathrm{TP}\)` is number of true WP (true positive), `\(\mathrm{TN}\)` is the
number of true WC (true negative), `\(\mathrm{FN}\)` is the number of predicted WCs
which really were WPs (false negative) and `\(\mathrm{FP}\)` is the number of
falsely predicted WP vessels (false positive).

The other measure we are gonna use is the *classification accuracy*. It is
the fraction of predictions our model got right.

Figure <a href="#fig:lr-train-roc-curves">1</a> shows the trade-off between TRP and 1-FP.
Classifiers that give curves closer to the top-left corner indicate a better 
performance. Here we see that some gives a perfect prediction. 


```r
lr_fits_rs %>%
  imap(~ {
    collect_predictions(.x) %>%
      group_by(id) %>%
      roc_curve(truth = cargo, .pred_WP) %>%
      autoplot() + ggtitle(.y) + theme_orsted(base_family = "")
  }) %>%
  reduce(`+`)
```

<div class="figure">
<img src="/post/2021-01-07-cargo-classification/index.en_files/figure-html/lr-train-roc-curves-1.svg" alt="ROC curves." width="672" />
<p class="caption">Figure 1: ROC curves.</p>
</div>

As a final clearification we can look at Table <a href="#tab:lr-train-mdl-accu">1</a>  that
shows us a clear picture of our cross validation and can conclude that we will
use quay af a predictor.


```r
lr_fits_rs %>%
  imap_dfr(~ relocate(mutate(collect_metrics(.x), model = .y), model)) %>%
  kbl(caption = "Model accuracy.") %>% kable_styling()
```

<table class="table" style="margin-left: auto; margin-right: auto;">
<caption>Table 1: Model accuracy.</caption>
 <thead>
  <tr>
   <th style="text-align:left;"> model </th>
   <th style="text-align:left;"> .metric </th>
   <th style="text-align:left;"> .estimator </th>
   <th style="text-align:right;"> mean </th>
   <th style="text-align:right;"> n </th>
   <th style="text-align:right;"> std_err </th>
   <th style="text-align:left;"> .config </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> quay </td>
   <td style="text-align:left;"> accuracy </td>
   <td style="text-align:left;"> binary </td>
   <td style="text-align:right;"> 0.9377778 </td>
   <td style="text-align:right;"> 5 </td>
   <td style="text-align:right;"> 0.0254830 </td>
   <td style="text-align:left;"> Preprocessor1_Model1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> quay </td>
   <td style="text-align:left;"> roc_auc </td>
   <td style="text-align:left;"> binary </td>
   <td style="text-align:right;"> 0.9064931 </td>
   <td style="text-align:right;"> 5 </td>
   <td style="text-align:right;"> 0.0552653 </td>
   <td style="text-align:left;"> Preprocessor1_Model1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from </td>
   <td style="text-align:left;"> accuracy </td>
   <td style="text-align:left;"> binary </td>
   <td style="text-align:right;"> 0.8556140 </td>
   <td style="text-align:right;"> 5 </td>
   <td style="text-align:right;"> 0.0102995 </td>
   <td style="text-align:left;"> Preprocessor1_Model1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from </td>
   <td style="text-align:left;"> roc_auc </td>
   <td style="text-align:left;"> binary </td>
   <td style="text-align:right;"> 0.7748264 </td>
   <td style="text-align:right;"> 5 </td>
   <td style="text-align:right;"> 0.0681452 </td>
   <td style="text-align:left;"> Preprocessor1_Model1 </td>
  </tr>
</tbody>
</table>

### Prediction

Now that we have trained our model on the training data, we test how well it
does on the test data. We are os course certain that quay is the best predictor
but we are gonna show the result for both predictors. 


```r
# FIXME prediction from a rank-deficient fit may be misleading. https://stackoverflow.com/a/26560328.
lr_fits <- lr_wfls %>% map(~ fit(., data = df_train))
```


```r
lr_fits %>%
  imap(~ {
    predict(.x, df_test, type = "class") %>%
      bind_cols(select(df_test, cargo)) %>%
      conf_mat(truth = cargo, estimate = .pred_class) %>%
      autoplot(type = "heatmap") + ggtitle(.y)
  }) %>%
  reduce(`+`)
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type = if (type == :
## prediction from a rank-deficient fit may be misleading
```

<img src="/post/2021-01-07-cargo-classification/index.en_files/figure-html/unnamed-chunk-14-1.svg" width="672" />


```r
lr_fits %>%
  imap_dfr(~ {
    predict(.x, df_test, type = "prob") %>% 
      bind_cols(select(df_test, cargo)) %>%
      roc_curve(truth = cargo, .pred_WP) %>%
      mutate(model = .y)
  }) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) + 
  geom_path(lwd = 1.5, alpha = 0.8) +
  geom_abline(lty = 3) + 
  coord_equal() + 
  scale_color_orsted_d()
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type = if (type == :
## prediction from a rank-deficient fit may be misleading
```

<img src="/post/2021-01-07-cargo-classification/index.en_files/figure-html/unnamed-chunk-15-1.svg" width="672" />


```r
lr_pred_prob %>% roc_auc(truth = cargo, .pred_WP) %>%  kbl() %>% kable_styling()
```

From the above result we see that the quay as a predictor has correctly predicted
every data point in our test set.

## Conclusion

From our analysis we have created a logistic regression model that predicts if 
the upcomming vessel is arriving with WP or WC to CMport. 

Our model is using `quay` as a predictor and this give us good with a very high
accuracy.

## Estimation results {.appendix}


```r
lr_fits[[1]] %>% pull_workflow_fit() %>% tidy() %>% kbl(digits = 2) %>% kable_styling()
```

<table class="table" style="margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;"> term </th>
   <th style="text-align:right;"> estimate </th>
   <th style="text-align:right;"> std.error </th>
   <th style="text-align:right;"> statistic </th>
   <th style="text-align:right;"> p.value </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> (Intercept) </td>
   <td style="text-align:right;"> 4.13 </td>
   <td style="text-align:right;"> 1.01 </td>
   <td style="text-align:right;"> 4.10 </td>
   <td style="text-align:right;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> quay_c836 </td>
   <td style="text-align:right;"> -1.56 </td>
   <td style="text-align:right;"> 1.45 </td>
   <td style="text-align:right;"> -1.08 </td>
   <td style="text-align:right;"> 0.28 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> quay_c837 </td>
   <td style="text-align:right;"> -4.41 </td>
   <td style="text-align:right;"> 1.26 </td>
   <td style="text-align:right;"> -3.49 </td>
   <td style="text-align:right;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> quay_c838 </td>
   <td style="text-align:right;"> -6.61 </td>
   <td style="text-align:right;"> 1.45 </td>
   <td style="text-align:right;"> -4.56 </td>
   <td style="text-align:right;"> 0.00 </td>
  </tr>
</tbody>
</table>


```r
lr_fits[[2]] %>% pull_workflow_fit() %>% tidy() %>% kbl(digits = 2) %>% kable_styling()
```

<table class="table" style="margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;"> term </th>
   <th style="text-align:right;"> estimate </th>
   <th style="text-align:right;"> std.error </th>
   <th style="text-align:right;"> statistic </th>
   <th style="text-align:right;"> p.value </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> (Intercept) </td>
   <td style="text-align:right;"> 20.57 </td>
   <td style="text-align:right;"> 10236.63 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Avedörevaerkets.Havn </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Bandholm..Maribo. </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 13541.79 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Bekkeri </td>
   <td style="text-align:right;"> -41.13 </td>
   <td style="text-align:right;"> 14476.79 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Bremen.Farge </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Cartagena </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Enstedvaerkets.Havn </td>
   <td style="text-align:right;"> -41.13 </td>
   <td style="text-align:right;"> 16185.54 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Eydehavn </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 16185.54 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Gdynia </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 13541.79 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Gent </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 13541.79 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Greifswald..Landkreis </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 13541.79 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Grenå </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 20473.27 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Huelva </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 20473.27 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Köge </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Larvik </td>
   <td style="text-align:right;"> -19.87 </td>
   <td style="text-align:right;"> 10236.63 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Leixoes </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Liepaja </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 11444.90 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Lübeck </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 11356.53 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Lyngdal </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Mandal </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Mersrags </td>
   <td style="text-align:right;"> -20.57 </td>
   <td style="text-align:right;"> 10236.63 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Papenburg </td>
   <td style="text-align:right;"> -41.13 </td>
   <td style="text-align:right;"> 20473.27 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Riga </td>
   <td style="text-align:right;"> -19.55 </td>
   <td style="text-align:right;"> 10236.63 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Rohukuela </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Rostock </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 13541.79 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Santana </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 16185.54 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Skulte </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 16185.54 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Stralsund </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 14476.79 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Szczecin </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Szczecinek </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
   <td style="text-align:right;"> NA </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Tallinn </td>
   <td style="text-align:right;"> -41.13 </td>
   <td style="text-align:right;"> 16185.54 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Törkopp..Drammen. </td>
   <td style="text-align:right;"> -41.13 </td>
   <td style="text-align:right;"> 20473.27 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Ventspils </td>
   <td style="text-align:right;"> -19.18 </td>
   <td style="text-align:right;"> 10236.63 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Virtsu </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 20473.27 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Wismar </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 20473.27 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Övriga.danska.hamnar </td>
   <td style="text-align:right;"> -41.13 </td>
   <td style="text-align:right;"> 20473.27 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> arriving_from_Övriga.estniska.hamnar </td>
   <td style="text-align:right;"> -41.13 </td>
   <td style="text-align:right;"> 20473.27 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 1 </td>
  </tr>
</tbody>
</table>
