---
title: Varmelast time series
author: Lucas Bagge
date: '2021-01-24'
slug: varmelast-time-series
categories:
  - Time series
tags:
  - prophet
  - arima
  - xgboost
subtitle: ''
summary: ''
authors: []
lastmod: '2021-01-24T22:05:11+01:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---
<script src="{{< blogdown/postref >}}index.en_files/kePrint/kePrint.js"></script>
<link href="{{< blogdown/postref >}}index.en_files/lightable/lightable.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index.en_files/kePrint/kePrint.js"></script>
<link href="{{< blogdown/postref >}}index.en_files/lightable/lightable.css" rel="stylesheet" />
<script src="{{< blogdown/postref >}}index.en_files/htmlwidgets/htmlwidgets.js"></script>
<script src="{{< blogdown/postref >}}index.en_files/pymjs/pym.v1.js"></script>
<script src="{{< blogdown/postref >}}index.en_files/widgetframe-binding/widgetframe.js"></script>





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
varme_last <- function(from = "", to = "") {
  resp <- GET(
    paste0(
      "https://www.varmelast.dk/api/v1/heatdata/historical?from=",
      from, "&to=", to, "&intervalMinutes=60&contextSite=varmelast_dk"
    )
  )
  content <- fromJSON(content(resp, "text"))

  date <- content$times$timestamp

  df_list <- content$times$values

  date_tibble <- map_df(date, as_tibble)

  df_tibble <- map_df(df_list, as_tibble)

  df_wide <-
    df_tibble %>%
    select(-valueError) %>%
    pivot_wider(
      names_from = key,
      values_from = value
    ) %>%
    unnest()

  df <-
    df_wide %>%
    cbind(date_tibble) %>%
    mutate(
      date = ymd_hms(value),
      across(where(is.numeric), round, 2)
    ) %>%
    janitor::clean_names() %>%
    select(-value) %>%
    rename(
      Affaldsenergianlæg = be_vl_affald_ef,
      Kraftvarmeanlæg = be_vl_kraftv_ef,
      "Spidslast gas" = be_vl_spids_gas_ef,
      "Spidslast olie" = be_vl_spids_olie_ef,
      "Spidslast træpiller" = be_vl_bio_ef,
      "CO2 - Udledning" = be_vl_total_fak,
      "Lokal produktion" = local
    )
  df
}
```

With this function I can extract the data from all of the energy sources and
go as far back in time as possible. The site is relative new so we have to
see how far we can go back. 

As a start we can see if we can get data from *2020-01-01*.


```r
data <- varme_last(from = "2020-01-01", to = today())
```


```r
data %>%
  dplyr::glimpse()
```

```
## Rows: 2,391
## Columns: 17
## $ be_eo_ctr_eff         <dbl> 759.52, 745.04, 733.10, 730.36, 749.44, 793.04,…
## $ Affaldsenergianlæg    <dbl> 318.29, 323.38, 310.82, 313.69, 315.18, 307.19,…
## $ `Spidslast træpiller` <dbl> 0.32, 0.27, 0.14, 0.18, 0.35, 0.12, 0.10, 0.37,…
## $ be_vl_evo_ef          <dbl> 0.00, 5.73, 0.15, 0.00, 0.00, 0.97, 9.27, 0.28,…
## $ Kraftvarmeanlæg       <dbl> 798.53, 794.04, 787.56, 790.41, 804.35, 782.26,…
## $ `Spidslast gas`       <dbl> 44.77, 44.61, 44.51, 44.36, 44.63, 45.27, 45.37…
## $ `Spidslast olie`      <dbl> 2.79, 3.04, 2.80, 2.79, 2.79, 2.81, 2.86, 2.80,…
## $ `CO2 - Udledning`     <dbl> 10.88, 12.15, 10.96, 10.90, 10.80, 11.16, 12.80…
## $ biogas                <dbl> 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,…
## $ dap_veks_forbrug_eff  <dbl> 288.88, 286.80, 292.58, 298.93, 307.32, 315.64,…
## $ geoterm               <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
## $ ind_ovs               <dbl> 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,…
## $ `Lokal produktion`    <dbl> 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,…
## $ pump                  <dbl> 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,…
## $ sol                   <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
## $ total                 <dbl> 1164.72, 1171.06, 1145.97, 1151.43, 1167.31, 11…
## $ date                  <dttm> 2020-10-15 22:00:00, 2020-10-15 23:00:00, 2020…
```
From the above view from `glimse` we have **2246** observation ans **17**
features.


```r
data %>%
  summarise(
    date_min = min(date),
    date_max = max(date)
  ) %>%
  knitr::kable() %>%
  kableExtra::kable_styling()
```

<table class="table" style="margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;"> date_min </th>
   <th style="text-align:left;"> date_max </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> 2020-10-15 22:00:00 </td>
   <td style="text-align:left;"> 2021-01-23 13:00:00 </td>
  </tr>
</tbody>
</table>

So the earlist date is *2020-10-15*. SO the site is new pretty new.

Let us see if there is an na values in some of the features.


```r
data %>%
  summarise_all(funs(sum(is.na(.)))) %>%
  knitr::kable() %>%
  kableExtra::kable_styling()
```

<table class="table" style="margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:right;"> be_eo_ctr_eff </th>
   <th style="text-align:right;"> Affaldsenergianlæg </th>
   <th style="text-align:right;"> Spidslast træpiller </th>
   <th style="text-align:right;"> be_vl_evo_ef </th>
   <th style="text-align:right;"> Kraftvarmeanlæg </th>
   <th style="text-align:right;"> Spidslast gas </th>
   <th style="text-align:right;"> Spidslast olie </th>
   <th style="text-align:right;"> CO2 - Udledning </th>
   <th style="text-align:right;"> biogas </th>
   <th style="text-align:right;"> dap_veks_forbrug_eff </th>
   <th style="text-align:right;"> geoterm </th>
   <th style="text-align:right;"> ind_ovs </th>
   <th style="text-align:right;"> Lokal produktion </th>
   <th style="text-align:right;"> pump </th>
   <th style="text-align:right;"> sol </th>
   <th style="text-align:right;"> total </th>
   <th style="text-align:right;"> date </th>
  </tr>
 </thead>
<tbody>
  <tr>
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
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 0 </td>
  </tr>
</tbody>
</table>

There is no na values. So it seems that there is a automatic  transition
between the power plants going on so the data is pretty consistent. 

Lets us have a general view of the data


```r
test <- data %>%
  pivot_longer(
    cols = -c(date),
    names_to = "metric",
    values_to = "value"
  )
frameWidget(
  hchart(test, "line", hcaes(x = as.character(date), y = "value", group = "metric")) %>%
    hc_title(text = "Energy use of Copenhagen Power Plants") %>%
    hc_subtitle(text = "The data goes from 2020-10-15 to 2021-01-18") %>%
    hc_xAxis(
      title = list(text = ""),
      labels = list(enabled = FALSE)
    ) %>%
    hc_yAxis(title = list(text = "MJ/s"))
)
```

```{=html}
<div id="htmlwidget-1" style="width:100%;height:1500px;" class="widgetframe html-widget"></div>
<script type="application/json" data-for="htmlwidget-1">{"x":{"url":"index.en_files/figure-html//widgets/widget_unnamed-chunk-6.html","options":{"xdomain":"*","allowfullscreen":false,"lazyload":false}},"evals":[],"jsHooks":[]}</script>
```

As can be seem from the `highcharter` plot there is a lot of sources that
could be interesting to model on. I am gonna chosse `Kraftvarmeanlæg` to
work with and build several models on.
