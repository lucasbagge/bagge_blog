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





```r
kickstarter <- read_csv("kickstarter.csv 22.56.12.gz")
moviereview <- read_csv("movie_review.csv")
kickstarter %>% head()
```

```
## # A tibble: 6 x 3
##   blurb                                                         state created_at
##   <chr>                                                         <dbl> <date>    
## 1 Exploring paint and its place in a digital world.                 0 2015-03-17
## 2 Mike Fassio wants a side-by-side photo of me and Hazel eatin…     0 2014-07-11
## 3 I need your help to get a nice graphics tablet and Photoshop!     0 2014-07-30
## 4 I want to create a Nature Photograph Series of photos of wil…     0 2015-05-08
## 5 I want to bring colour to the world in my own artistic skill…     0 2015-02-01
## 6 We start from some lovely pictures made by us and we decide …     0 2015-11-18
```

```r
moviereview %>% head()
```

```
## # A tibble: 6 x 6
##   fold_id cv_tag html_id sent_id text                                      tag  
##     <dbl> <chr>    <dbl>   <dbl> <chr>                                     <chr>
## 1       0 cv000    29590       0 "films adapted from comic books have had… pos  
## 2       0 cv000    29590       1 "for starters , it was created by alan m… pos  
## 3       0 cv000    29590       2 "to say moore and campbell thoroughly re… pos  
## 4       0 cv000    29590       3 "the book ( or \" graphic novel , \" if … pos  
## 5       0 cv000    29590       4 "in other words , don't dismiss this fil… pos  
## 6       0 cv000    29590       5 "if you can get past the whole comic boo… pos
```


```r
kickstarter %>%
  ggplot(aes(nchar(blurb))) +
  geom_histogram(binwidth = 1, alpha = 0.8) +
  labs(
    x = "Number of characters per campaign blurb",
    y = "Number of campaign blurbs"
  )
```

<img src="{{< blogdown/postref >}}index.en_files/figure-html/unnamed-chunk-2-1.png" width="2400" />

```r
moviereview %>%
  ggplot(aes(nchar(text))) +
  geom_histogram(binwidth = 1, alpha = 0.8) +
  labs(
    x = "Number of characters per campaign blurb",
    y = "Number of campaign blurbs"
  )
```

<img src="{{< blogdown/postref >}}index.en_files/figure-html/unnamed-chunk-2-2.png" width="2400" />


```r
moviereview %>%
  count(nchar(text), sort = TRUE)
```

```
## # A tibble: 541 x 2
##    `nchar(text)`     n
##            <int> <int>
##  1            84   455
##  2            85   448
##  3            82   444
##  4            94   441
##  5           106   432
##  6            86   431
##  7           102   429
##  8            89   426
##  9            99   426
## 10           112   426
## # … with 531 more rows
```


```r
moviereview_split <- moviereview %>% initial_split()

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

<img src="{{< blogdown/postref >}}index.en_files/figure-html/unnamed-chunk-5-1.png" width="2400" />


```r
library(textrecipes)

max_words <- 2000
max_length <- 50

movie_rec <- recipe(~text, data = train) %>%
  step_tokenize(text) %>%
  step_tokenfilter(text, max_tokens = max_words) %>%
  step_sequence_onehot(text, sequence_length = max_length)

movie_rec
```

```
## Data Recipe
## 
## Inputs:
## 
##       role #variables
##  predictor          1
## 
## Operations:
## 
## Tokenization for text
## Text filtering for text
## Sequence 1 hot encoding for text
```

