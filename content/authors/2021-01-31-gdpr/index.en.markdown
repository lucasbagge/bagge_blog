---
title: GDPR
author: Lucas Bagge
date: '2021-01-31'
slug: gdpr
categories: []
tags: []
subtitle: ''
summary: ''
authors: []
lastmod: '2021-01-31T00:18:25+01:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---




# Introduction

For this first post I will show an example of using newly launched [tidymodels](https://www.tidymodels.org/), 
which is a new packages that make it easiler to make models. To test the packages functionality
I will work on an dataset for GDPR violation and see which gives the biggest fines.

## Explore the date

Our modelling goal here is to understand which GDOR violations gives the higest fines. Before I
will look at the data it is reasonable to understand some of the GDPR articles and
what they are about:

- **Article 5:** principles for processing personal data (legitimate purpose, limited)
- **Article 6:** lawful processing of personal data (i.e. consent, etc)
- **Article 13:** inform subject when personal data is collected
- **Article 15:** right of access by data subject
- **Article 32:** security of processing (i.e. data breaches)


As a first step in analysis gdpr violations and get a understanding of tidymodels
I will explore the data.





```r
gdpr_raw
```

```
## # A tibble: 250 x 11
##       id picture name   price authority date  controller article_violated type 
##    <dbl> <chr>   <chr>  <dbl> <chr>     <chr> <chr>      <chr>            <chr>
##  1     1 https:… Pola…   9380 Polish N… 10/1… Polish Ma… Art. 28 GDPR     Non-…
##  2     2 https:… Roma…   2500 Romanian… 10/1… UTTIS IND… Art. 12 GDPR|Ar… Info…
##  3     3 https:… Spain  60000 Spanish … 10/1… Xfera Mov… Art. 5 GDPR|Art… Non-…
##  4     4 https:… Spain   8000 Spanish … 10/1… Iberdrola… Art. 31 GDPR     Fail…
##  5     5 https:… Roma… 150000 Romanian… 10/0… Raiffeise… Art. 32 GDPR     Fail…
##  6     6 https:… Roma…  20000 Romanian… 10/0… Vreau Cre… Art. 32 GDPR|Ar… Fail…
##  7     7 https:… Gree… 200000 Hellenic… 10/0… Telecommu… Art. 5 (1) c) G… Fail…
##  8     8 https:… Gree… 200000 Hellenic… 10/0… Telecommu… Art. 21 (3) GDP… Fail…
##  9     9 https:… Spain  30000 Spanish … 10/0… Vueling A… Art. 5 GDPR|Art… Non-…
## 10    10 https:… Roma…   9000 Romanian… 09/2… Inteligo … Art. 5 (1) a) G… Non-…
## # … with 240 more rows, and 2 more variables: source <chr>, summary <chr>
```


How are they distributed?


```r
gdpr_raw %>%
  ggplot(aes(
    price + 1
  )) +
  geom_histogram() +
  scale_x_log10(labels = scales::dollar_format(prefix = "€")) +
  labs(
    x = "GDPR fine (EUR)",
    y = "GDPR violations"
  ) +
  theme_minimal()
```

<img src="{{< blogdown/postref >}}index.en_files/figure-html/unnamed-chunk-3-1.png" width="2400" />

We will know make the data tidy:


```r
gdp_tidy <- gdpr_raw %>%
  # add nes variable and drops existing ones.
  transmute(id,
    price,
    country = name,
    article_violated,
    articles = str_extract_all(
      article_violated,
      # Add regex to get numbers
      # Have a look to see that some didt get becasue of missing space
      "Art. [:digit:]+|Art.[:digit:]+"
    )
  ) %>%
  # more engerring
  # more violations gives bigger fines
  # Beacuse we know it is an integer we can say map_int
  mutate(total_articles = map_int(articles, length)) %>%
  # by unnest we are gonna spread the articles.
  unnest(articles) %>%
  # We can count the number of articles.
  # There is 27 articles and is to many for our modelling
  # this wee need to faut
  # count(articles, sort = TRUE)
  add_count(articles) %>%
  filter(n > 10) %>%
  select(-n)
# Kow we have a row for each violations and not
gdp_tidy
```

```
## # A tibble: 304 x 6
##       id  price country article_violated                 articles total_articles
##    <dbl>  <dbl> <chr>   <chr>                            <chr>             <int>
##  1     2   2500 Romania Art. 12 GDPR|Art. 13 GDPR|Art. … Art. 13               4
##  2     2   2500 Romania Art. 12 GDPR|Art. 13 GDPR|Art. … Art. 5                4
##  3     2   2500 Romania Art. 12 GDPR|Art. 13 GDPR|Art. … Art. 6                4
##  4     3  60000 Spain   Art. 5 GDPR|Art. 6 GDPR          Art. 5                2
##  5     3  60000 Spain   Art. 5 GDPR|Art. 6 GDPR          Art. 6                2
##  6     5 150000 Romania Art. 32 GDPR                     Art. 32               1
##  7     6  20000 Romania Art. 32 GDPR|Art. 33 GDPR        Art. 32               2
##  8     7 200000 Greece  Art. 5 (1) c) GDPR|Art. 25 GDPR  Art. 5                2
##  9     9  30000 Spain   Art. 5 GDPR|Art. 6 GDPR          Art. 5                2
## 10     9  30000 Spain   Art. 5 GDPR|Art. 6 GDPR          Art. 6                2
## # … with 294 more rows
```

How are the articles distributed over per article? 


```r
library(ggbeeswarm)
gdp_tidy %>%
  mutate(
    articles = str_replace_all(articles, "Art. ", "Article "),
    articles = fct_reorder(articles, price)
  ) %>%
  ggplot(aes(
    articles, price + 1,
    color = articles, fill = articles
  )) +
  geom_quasirandom() +
  geom_boxplot(alpha = 0.2, outlier.colour = NA) +
  scale_y_log10(labels = scales::dollar_format(prefix = "£")) +
  labs(
    x = NULL, y = "GDPR fine (EUR)",
    title = "GDPR fines levied by article",
    subtitle = "For 250 violations in 25 countries"
  ) +
  theme_minimal() +
  theme(legend.position = "none")
```

<img src="{{< blogdown/postref >}}index.en_files/figure-html/unnamed-chunk-5-1.png" width="2400" />

From the beeshart I can see that ther is a bug difference in the size of the fine and
the different articles. Article 15 is accosiated with the smallest fine and for
articles 5 and 6 you get a highere fine.

Know I will prefer the data for the modelling.


```r
# Back to one row to on violation.
gdpr_violation <- gdp_tidy %>%
  mutate(value = 1) %>%
  select(-article_violated) %>%
  # article and value is gonna be maded wider
  pivot_wider(
    names_from = articles, values_from = value,
    values_fn = list(value = max),
    values_fill = list(value = 0)
  ) %>%
  janitor::clean_names()
gdpr_violation
```

```
## # A tibble: 219 x 9
##       id  price country total_articles art_13 art_5 art_6 art_32 art_15
##    <dbl>  <dbl> <chr>            <int>  <dbl> <dbl> <dbl>  <dbl>  <dbl>
##  1     2   2500 Romania              4      1     1     1      0      0
##  2     3  60000 Spain                2      0     1     1      0      0
##  3     5 150000 Romania              1      0     0     0      1      0
##  4     6  20000 Romania              2      0     0     0      1      0
##  5     7 200000 Greece               2      0     1     0      0      0
##  6     9  30000 Spain                2      0     1     1      0      0
##  7    10   9000 Romania              2      0     1     1      0      0
##  8    11 195407 Germany              3      0     0     0      0      1
##  9    12  10000 Belgium              1      0     1     0      0      0
## 10    13 644780 Poland               1      0     0     0      1      0
## # … with 209 more rows
```

With this data we are ready to take the next step and start building our model.

## Build model

One key step before building the model is the prepocess the data.


```r
library(tidymodels)
# Data preprocessing
gdpr_rec <- recipe(price ~ ., data = gdpr_violation) %>%
  update_role(id, new_role = "id") %>%
  step_log(price, base = 10, offset = 1, skip = TRUE) %>% # skip: i tilfælde vi ikke har et outcome varibale
  # There is to many country
  step_other(country, other = "Other") %>%
  step_dummy(all_nominal()) %>%
  # fjerner ting som ingen varians har.
  step_zv(all_predictors())
# We set a recipe to model our data.
gdpr_prep <- prep(gdpr_rec)
# For train the model use prep
gdpr_prep
```

```
## Data Recipe
## 
## Inputs:
## 
##       role #variables
##         id          1
##    outcome          1
##  predictor          7
## 
## Training data contained 219 data points and no missing data.
## 
## Operations:
## 
## Log transformation on price [trained]
## Collapsing factor levels for country [trained]
## Dummy variables from country [trained]
## Zero variance filter removed no terms [trained]
```

The modelling process in tidymodels is build on a recipe like when you are baking.
It makes sence because you can see both process as a algoritme, which is just some
steps you are doing to complete your task.

- The first thing we need to specify is what are our model going to be and what
data is it going to use.
- For the next part we need to identify what we consider predictor or outcome. Here it
is important that we tell the recipe that `id` is not either but we want to keep it.
- The next step is to take the log of `price`, which is the amount of the fine.
- We only want to look at the important `country`, so the other we collapses with `Other`. 
- At last we create an indicator variable and remove variable with zero variance.

Until know we haven´t don anything but only defined a lot of stuff. With `prep`
we evaluted our data.

Here we introduced the `workflow()` which can be associates with lego blocks. Here
we can have both the recipe and model (our model is a OLS model). 


```r
# How does our data look like know?
juice(gdpr_prep)
```

```
## # A tibble: 219 x 14
##       id total_articles art_13 art_5 art_6 art_32 art_15 price country_Czech.R…
##    <dbl>          <int>  <dbl> <dbl> <dbl>  <dbl>  <dbl> <dbl>            <dbl>
##  1     2              4      1     1     1      0      0  3.40                0
##  2     3              2      0     1     1      0      0  4.78                0
##  3     5              1      0     0     0      1      0  5.18                0
##  4     6              2      0     0     0      1      0  4.30                0
##  5     7              2      0     1     0      0      0  5.30                0
##  6     9              2      0     1     1      0      0  4.48                0
##  7    10              2      0     1     1      0      0  3.95                0
##  8    11              3      0     0     0      0      1  5.29                0
##  9    12              1      0     1     0      0      0  4.00                0
## 10    13              1      0     0     0      1      0  5.81                0
## # … with 209 more rows, and 5 more variables: country_Germany <dbl>,
## #   country_Hungary <dbl>, country_Romania <dbl>, country_Spain <dbl>,
## #   country_Other <dbl>
```


```r
# workflow, contain object that let you carrie stuff around.
gdpr_workflow <- workflow() %>%
  add_recipe(gdpr_rec) %>%
  add_model(linear_reg() %>%
    set_engine("lm"))
# I want a model wark flow and get a linear model.
# prepoccesor + model
#
gdpr_workflow
```

```
## ══ Workflow ════════════════════════════════════════════════════════════════════
## Preprocessor: Recipe
## Model: linear_reg()
## 
## ── Preprocessor ────────────────────────────────────────────────────────────────
## 4 Recipe Steps
## 
## ● step_log()
## ● step_other()
## ● step_dummy()
## ● step_zv()
## 
## ── Model ───────────────────────────────────────────────────────────────────────
## Linear Regression Model Specification (regression)
## 
## Computational engine: lm
```

When we normally have a model we want to fit it. This is an important aspect in
modelling because this tell us have well our model is. 


```r
gdpr_fit <- gdpr_workflow %>%
  fit(data = gdpr_violation)
(gdpr_fit)
```

```
## ══ Workflow [trained] ══════════════════════════════════════════════════════════
## Preprocessor: Recipe
## Model: linear_reg()
## 
## ── Preprocessor ────────────────────────────────────────────────────────────────
## 4 Recipe Steps
## 
## ● step_log()
## ● step_other()
## ● step_dummy()
## ● step_zv()
## 
## ── Model ───────────────────────────────────────────────────────────────────────
## 
## Call:
## stats::lm(formula = ..y ~ ., data = data)
## 
## Coefficients:
##            (Intercept)          total_articles                  art_13  
##                3.76607                 0.47957                -0.76251  
##                  art_5                   art_6                  art_32  
##               -0.41869                -0.55988                -0.15317  
##                 art_15  country_Czech.Republic         country_Germany  
##               -1.56765                -0.64953                 0.05974  
##        country_Hungary         country_Romania           country_Spain  
##               -0.15532                -0.34580                 0.42968  
##          country_Other  
##                0.23438
```

So we can fit a model or a workflow.

## Explore results

Let us see if we can get anything out of weather country and fines are different.


```r
gdpr_fit %>%
  pull_workflow_fit() %>%
  tidy() %>%
  arrange(estimate) %>%
  kable()
```



|term                   |   estimate| std.error|  statistic|   p.value|
|:----------------------|----------:|---------:|----------:|---------:|
|art_15                 | -1.5676538| 0.4651576| -3.3701564| 0.0008969|
|art_13                 | -0.7625069| 0.4074302| -1.8715031| 0.0626929|
|country_Czech.Republic | -0.6495339| 0.4667470| -1.3916188| 0.1655387|
|art_6                  | -0.5598765| 0.2950382| -1.8976404| 0.0591419|
|art_5                  | -0.4186949| 0.2828869| -1.4800789| 0.1403799|
|country_Romania        | -0.3457980| 0.4325560| -0.7994295| 0.4249622|
|country_Hungary        | -0.1553232| 0.4790037| -0.3242631| 0.7460679|
|art_32                 | -0.1531725| 0.3146769| -0.4867613| 0.6269450|
|country_Germany        |  0.0597408| 0.4189434|  0.1425986| 0.8867465|
|country_Other          |  0.2343787| 0.3551225|  0.6599939| 0.5099950|
|country_Spain          |  0.4296805| 0.3643060|  1.1794494| 0.2395796|
|total_articles         |  0.4795667| 0.1656494|  2.8950705| 0.0041993|
|(Intercept)            |  3.7660677| 0.4089156|  9.2098904| 0.0000000|

The results makes perfectly sense and we see that if you have many GDPR violation
then the fine is bigger.


```r
gdpr_fit %>%
  pull_workflow_fit() %>%
  tidy() %>%
  filter(p.value < 0.05) %>%
  arrange(estimate) %>%
  kable()
```



|term           |   estimate| std.error| statistic|   p.value|
|:--------------|----------:|---------:|---------:|---------:|
|art_15         | -1.5676538| 0.4651576| -3.370156| 0.0008969|
|total_articles |  0.4795667| 0.1656494|  2.895071| 0.0041993|
|(Intercept)    |  3.7660677| 0.4089156|  9.209890| 0.0000000|

If you violates art_15 which is if you dont give the right of access by data subject,
then you get a lower fine.  

What if we relax it a litlle bit?


```r
gdpr_fit %>%
  pull_workflow_fit() %>%
  tidy() %>%
  filter(p.value < 0.1) %>%
  kable()
```



|term           |   estimate| std.error| statistic|   p.value|
|:--------------|----------:|---------:|---------:|---------:|
|(Intercept)    |  3.7660677| 0.4089156|  9.209890| 0.0000000|
|total_articles |  0.4795667| 0.1656494|  2.895071| 0.0041993|
|art_13         | -0.7625069| 0.4074302| -1.871503| 0.0626929|
|art_6          | -0.5598765| 0.2950382| -1.897640| 0.0591419|
|art_15         | -1.5676538| 0.4651576| -3.370156| 0.0008969|

Here we see that violation of article 13, 6 and 15 gives you are smaller fine.

We know want to explore the result even more and try to predict. Here tidymodels is
really good.


As we have seen from the resulta we have big p-values. But I understand the results
better if I see them visual. I will make prediction with the help of `workflow()`

Let´s create some example new data that we are interested in.


```r
new_data <- crossing(
  country = "Other",
  art_5 = 0:1,
  art_6 = 0:1,
  art_13 = 0:1,
  art_15 = 0:1,
  art_32 = 0:1
) %>%
  mutate(
    total_articles = art_5 +
      art_6 +
      art_13 +
      art_15 +
      art_32,
    id = row_number()
  )
new_data %>%
  head() %>%
  kable()
```



|country | art_5| art_6| art_13| art_15| art_32| total_articles| id|
|:-------|-----:|-----:|------:|------:|------:|--------------:|--:|
|Other   |     0|     0|      0|      0|      0|              0|  1|
|Other   |     0|     0|      0|      0|      1|              1|  2|
|Other   |     0|     0|      0|      1|      0|              1|  3|
|Other   |     0|     0|      0|      1|      1|              2|  4|
|Other   |     0|     0|      1|      0|      0|              1|  5|
|Other   |     0|     0|      1|      0|      1|              2|  6|


The above code are making some new data. With the new data we can make prediction
analysis. In the next step I will calculate the mean and confidence interval.


```r
mean_pred <- predict(gdpr_fit,
  new_data = new_data
)
conf_int_pred <- predict(gdpr_fit,
  new_data = new_data,
  type = "conf_int"
)
gdpr_res <- new_data %>%
  bind_cols(mean_pred) %>%
  bind_cols(conf_int_pred)
gdpr_res
```

```
## # A tibble: 32 x 11
##    country art_5 art_6 art_13 art_15 art_32 total_articles    id .pred
##    <chr>   <int> <int>  <int>  <int>  <int>          <int> <int> <dbl>
##  1 Other       0     0      0      0      0              0     1  4.00
##  2 Other       0     0      0      0      1              1     2  4.33
##  3 Other       0     0      0      1      0              1     3  2.91
##  4 Other       0     0      0      1      1              2     4  3.24
##  5 Other       0     0      1      0      0              1     5  3.72
##  6 Other       0     0      1      0      1              2     6  4.04
##  7 Other       0     0      1      1      0              2     7  2.63
##  8 Other       0     0      1      1      1              3     8  2.96
##  9 Other       0     1      0      0      0              1     9  3.92
## 10 Other       0     1      0      0      1              2    10  4.25
## # … with 22 more rows, and 2 more variables: .pred_lower <dbl>,
## #   .pred_upper <dbl>
```

Know we have the prediction in log euros. We also see we can get the confidence
interval. 

From the res we have that for different amount violation what can we predict
what they have violated. 


```r
gdpr_res %>%
  filter(total_articles == 1) %>%
  pivot_longer(art_5:art_32) %>%
  filter(value > 0) %>%
  mutate(
    name = str_replace_all(name, "art_", "Article "),
    name = fct_reorder(name, .pred)
  ) %>%
  ggplot(aes(name, 10^.pred, color = name)) +
  geom_point(size = 3.5) +
  geom_errorbar(aes(
    ymin = 10^.pred_lower,
    ymax = 10^.pred_upper
  ),
  width = 0.2, alpha = 0.7
  ) +
  theme_minimal() +
  labs(
    x = NULL, y = "Increase in fine (EUR)",
    title = "Predicted fine for each type of GDPR article violation",
    subtitle = "Modeling based on 250 violations in 25 countries"
  ) +
  scale_y_log10(labels = scales::dollar_format(prefix = "€", accuracy = 1)) +
  theme(legend.position = "none")
```

<img src="{{< blogdown/postref >}}index.en_files/figure-html/unnamed-chunk-16-1.png" width="2400" />

Know we have a model over the how expensive each fine are and which will give 
you the highest.
