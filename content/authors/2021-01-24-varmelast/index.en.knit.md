---
title: Varmelast
author: Lucas Bagge
date: '2021-01-24'
slug: varmelast
categories: []
tags: []
subtitle: ''
summary: ''
authors: []
lastmod: '2021-01-24T22:50:11+01:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---






# Introduction

The energy use of the different power plant around Copenhagen is an important
indicator of where there can be a shorted of of sources.

The site [varmelast](https://www.varmelast.dk/) offers a view of how the different
plans use the energy sources every hour.

In this post I am gonna build a function to extract the data on the site and
use `timetk` to build different models to forecast the time series.


## Data

We need to get data from [varmelast](https://www.varmelast.dk/) which we do with 
he following function. Unfortunately there is no download bottom so we need to
extract the data by ourselve with the`httr` package.

To make it more generic I am create a function `varme_last` that extract the data for us.
Here I specify the user can decide in what period the data is gonna be extracted.



```r
varme_last <- function(from = "",  to = "") {
  resp <- GET(
    paste0('https://www.varmelast.dk/api/v1/heatdata/historical?from=',
           from,'&to=',to,'&intervalMinutes=60&contextSite=varmelast_dk')
  )
  content <- fromJSON(content(resp, 'text'))
  
  date <- content$times$timestamp
  
  df_list <- content$times$values
  
  date_tibble <- map_df(date, as_tibble)
  
  df_tibble <- map_df(df_list, as_tibble)
  
  df_wide <-
    df_tibble %>%
    select(-valueError) %>%
    pivot_wider(names_from =  key,
                values_from = value) %>%
    unnest()
  
  df <-
    df_wide %>% cbind(date_tibble) %>%
    mutate(date = ymd_hms(value),
           across(where(is.numeric), round, 2)) %>% 
    janitor::clean_names() %>%
    select(-value) %>%
    rename(
      Affaldsenergianlæg = be_vl_affald_ef,
      Kraftvarmeanlæg = be_vl_kraftv_ef,
      'Spidslast gas' = be_vl_spids_gas_ef,
      'Spidslast olie' = be_vl_spids_olie_ef,
      'Spidslast træpiller' = be_vl_bio_ef,
      'CO2 - Udledning' = be_vl_total_fak,
      'Lokal produktion' = local
    )
  df
}
```

With this function I can extract the data from all of the energy sources and
go as far back in time as possible. The site is relative new so we have to
see how far we can go back. 

As a start we can see if we can get data from *2020-01-01*.









