---
title: 'Varmelast'
author: Lucas Bagge
date: '2021-01-24'
slug: varmelast
categories: 
- Time Series
tags: 
- Xgboost
- machine learning
- prophet
subtitle: 
summary: 'Learn how to use ´timetk´ to build time series models.'
authors: [Lucas Bagge]
lastmod: '2021-01-24T22:50:11+01:00'
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
<script src="{{< blogdown/postref >}}index.en_files/kePrint/kePrint.js"></script>
<link href="{{< blogdown/postref >}}index.en_files/lightable/lightable.css" rel="stylesheet" />






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
  content <- fromJSON(httr::content(resp, 'text'))
  
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


```r
data <- varme_last(from = "2020-01-01", to = today())
```

```
## Warning: Values are not uniquely identified; output will contain list-cols.
## * Use `values_fn = list` to suppress this warning.
## * Use `values_fn = length` to identify where the duplicates arise
## * Use `values_fn = {summary_fun}` to summarise duplicates
```

```
## Warning: `cols` is now required when using unnest().
## Please use `cols = c(`BE-EO-CTR-EFF`, `BE-VL-AFFALD-EF`, `BE-VL-BIO-EF`, `BE-VL-EVO-EF`, 
##     `BE-VL-KRAFTV-EF`, `BE-VL-SPIDS-GAS-EF`, `BE-VL-SPIDS-OLIE-EF`, 
##     `BE-VL-TOTAL-FAK`, BIOGAS, `DAP-VEKS-FORBRUG-EFF`, GEOTERM, 
##     IND_OVS, LOCAL, PUMP, SOL, TOTAL)`
```


```r
data %>% 
  dplyr::glimpse()
```

```
## Rows: 2,536
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
  summarise(date_min = min(date),
            date_max = max(date)) %>% 
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
   <td style="text-align:left;"> 2021-01-29 14:00:00 </td>
  </tr>
</tbody>
</table>

So the earlist date is *2020-10-15*. SO the site is new pretty new.

Let us see if there is an na values in some of the features.


```r
data %>% summarise_all(funs(sum(is.na(.)))) %>% 
  knitr::kable() %>% 
  kableExtra::kable_styling()
```

```
## Warning: `funs()` is deprecated as of dplyr 0.8.0.
## Please use a list of either functions or lambdas: 
## 
##   # Simple named list: 
##   list(mean = mean, median = median)
## 
##   # Auto named with `tibble::lst()`: 
##   tibble::lst(mean, median)
## 
##   # Using lambdas
##   list(~ mean(., trim = .2), ~ median(., na.rm = TRUE))
## This warning is displayed once every 8 hours.
## Call `lifecycle::last_warnings()` to see where this warning was generated.
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
data %>%
  pivot_longer(cols = -c(date),
               names_to = "metric",
               values_to = "value") %>%
  ggplot(aes(x = date, y = value, colour = metric)) +
  geom_line() +
  my_theme() +
  labs(
    title = "Energy use of Copenhagen Power Plants",
    subtitle = "The data goes from 2020-10-15 to 2021-01-18",
    y = "MJ/s",
    x = "")
```

<img src="{{< blogdown/postref >}}index.en_files/figure-html/unnamed-chunk-7-1.png" width="672" />

As can be seem from the `highcharter` plot there is a lot of sources that
could be interesting to model on. I am gonna chosse `Kraftvarmeanlæg` to
work with and build several models on.

## Model Data

I am gonna create the data that I will build my models around. Here I am gonna use
`date` and `Kraftvarmeanlæg`.


```r
df <- data %>%
  select(date, Kraftvarmeanlæg,
         Affaldsenergianlæg, dap_veks_forbrug_eff
         ) %>%
  rename(value = Kraftvarmeanlæg) %>%
  as_tibble()
```


## Traning and Testing data

As in every model building I am gonna split the data in a traning and testing data.
For building time series models the new packages `timetk` is a great additonal
tool for building such models and it comes with great ways of splitting
the data.


```r
split <- df %>%
  time_series_split(assess = "5 days", cumulative = TRUE)
```

```
## Using date_var: date
```

Lets us get a view of the traning and testing data.


```r
split %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, value, .interactive = FALSE)
```

<img src="{{< blogdown/postref >}}index.en_files/figure-html/unnamed-chunk-10-1.png" width="672" />


From the plot we can see there is a general growing tendency in the data but
there is also a sharply decline in the periods. It would be ideal to have data
for a years so we could se if there was a clear seasonal effects.

From the split we can use `rsample` functions to build the traning and testing data.


```r
train <- training(split)
test <- testing(split)
```


# Modelling

From the traning data we a gonna build multiple models to try to forecast
Kraftvarmeanlæg.

This articles is meant as an introduction for time series so the math and theoretical
behinde each algoritmes is not something I will put a big foucs on. Additional
post will go into deep with each models and the math behinde them.

## Arima


```r
model_fit_arima <- arima_reg() %>%
  set_engine("auto_arima") %>%
  fit(value ~ date,
        train)
```

```
## frequency = 24 observations per 1 day
```

## Prophet

A prophet model is maybe most fomous of being a forecasting model developed by Facebook.


```r
model_fit_prophet <-
  prophet_reg(seasonality_daily = TRUE) %>%
  set_engine("prophet") %>%
  fit(value ~ date, train)
```

```
## Disabling yearly seasonality. Run prophet with yearly.seasonality=TRUE to override this.
```

## ML models

They are more complex then the automated models. This means wee need to add
a workflow (also called pipeline). In the tidymodels framwork we have the
folowwing process:

- Create preprocessing recipe
- Create model specifications
- Use workflow (wf) to combine model spec and preprocessing and fit model

### Preprocessing Recipe

I want to create a preprocessing recipe with `recipe` and use some steps in
creating new features that is gonna be used in the model. These include
time series signature and fourier series.


```r
recipe_spec <- recipe(value ~ date, train) %>%
  step_timeseries_signature(date) %>%
  step_rm(contains("am.pm"), contains("hour"), contains("minute"),
          contains("second"), contains("xts")) %>%
  step_fourier(date, period =  365 / 12, K = 2) %>%
  step_dummy(all_nominal())

recipe_spec %>% prep() %>% juice()
```

```
## # A tibble: 2,416 x 41
##    date                value date_index.num date_year date_year.iso date_half
##    <dttm>              <dbl>          <dbl>     <int>         <int>     <int>
##  1 2020-10-15 22:00:00  799.     1602799200      2020          2020         2
##  2 2020-10-15 23:00:00  794.     1602802800      2020          2020         2
##  3 2020-10-16 00:00:00  788.     1602806400      2020          2020         2
##  4 2020-10-16 01:00:00  790.     1602810000      2020          2020         2
##  5 2020-10-16 02:00:00  804.     1602813600      2020          2020         2
##  6 2020-10-16 03:00:00  782.     1602817200      2020          2020         2
##  7 2020-10-16 04:00:00  794.     1602820800      2020          2020         2
##  8 2020-10-16 05:00:00  723.     1602824400      2020          2020         2
##  9 2020-10-16 06:00:00  723.     1602828000      2020          2020         2
## 10 2020-10-16 07:00:00  728.     1602831600      2020          2020         2
## # … with 2,406 more rows, and 35 more variables: date_quarter <int>,
## #   date_month <int>, date_day <int>, date_wday <int>, date_mday <int>,
## #   date_qday <int>, date_yday <int>, date_mweek <int>, date_week <int>,
## #   date_week.iso <int>, date_week2 <int>, date_week3 <int>, date_week4 <int>,
## #   date_mday7 <int>, date_sin30.42_K1 <dbl>, date_cos30.42_K1 <dbl>,
## #   date_sin30.42_K2 <dbl>, date_cos30.42_K2 <dbl>, date_month.lbl_01 <dbl>,
## #   date_month.lbl_02 <dbl>, date_month.lbl_03 <dbl>, date_month.lbl_04 <dbl>,
## #   date_month.lbl_05 <dbl>, date_month.lbl_06 <dbl>, date_month.lbl_07 <dbl>,
## #   date_month.lbl_08 <dbl>, date_month.lbl_09 <dbl>, date_month.lbl_10 <dbl>,
## #   date_month.lbl_11 <dbl>, date_wday.lbl_1 <dbl>, date_wday.lbl_2 <dbl>,
## #   date_wday.lbl_3 <dbl>, date_wday.lbl_4 <dbl>, date_wday.lbl_5 <dbl>,
## #   date_wday.lbl_6 <dbl>
```

WIth this recipe created we can use this as one of the ingretigens in a Machine
Learning pipeline.

### Elastic Net

Making a **Elastic Net** model is very easy to do here we just need to set the
spec to use `linear_reg()` and `set_engine("glmnet")`.


```r
model_spec_glmnet <- linear_reg(penalty = 0.1, mixture = 0.5) %>%
  set_engine("glmnet")
```

Notice here we have not fitted the model yet as we did in the firsts model. It
is because we are gonna fit the model in our workflow:

- Start with a workflow.
- Add a model spec.
- Add preprocessing.
-- Note here that we remove the date column because ML algorithms dont know
how to deal with date features.
- Fit the workflow.


```r
workflow_fit_glmnet <- workflow() %>%
  add_model(model_spec_glmnet) %>%
  add_recipe(recipe_spec %>%  step_rm(date)) %>%
  fit(train)
```


## New Hybrid models

As a new model I will showcase a **hybrid models** (a combination between
`arima_boost()` and `prophet_boost()`) that combine the two automated algorithms
with ML.

### Prophet Boost

The **Prophet Boost algorithm** combines Prophet with XGBoost to get the best
of the two. The algoritme works as follow:

1) First modeling the univariate series using Prophet
2) Using regressors supplied via the preprocessing recipe and regressing the
Prophet Residual with the XGBoost model

As with the other ML models we set it up in our workflow.


```r
model_spec_prophet_boost <- prophet_boost(seasonality_daily  = TRUE) %>%
  set_engine("prophet_xgboost")

workflow_fit_prophet_boost <- workflow() %>%
  add_model(model_spec_prophet_boost) %>%
  add_recipe(recipe_spec) %>%
  fit(train)
```

```
## Disabling yearly seasonality. Run prophet with yearly.seasonality=TRUE to override this.
```


# The modeltime Workflow

With `modeltime` workflow we can speed up the model evaluation and it is very
useful know we have several time series models. In the next path I will analyze
them and forecast the future with the modeltime workflow.

## Modeltime table

The function `modeltime_table()` organizes the models with IDs and creates
generic descriptions to help us keep track of our models.


```r
model_table <- modeltime_table(
  model_fit_arima,
  model_fit_prophet,
  workflow_fit_glmnet,
  workflow_fit_prophet_boost
)
```

## Calibration

**Model Calibration** is used to quantify error and estimate confidence interval.
Here we gonna use calibration on our testing set with `modeltime_calibrate()`.
When using the function we create two new colums .type and .calibration_data
where the most important column is the former. This includes the actual values,
fitted vaues and residuals for the testing set.


```r
calibration_table <- model_table %>%
  modeltime_calibrate(test)
```

## Forecast (test set)

With the calibrated data we can visualize the testing prediction also called
forecast.

- Use `modeltime_forecast()` to generate the forecast data for the tesring set
  as a tiblle.
- Use `plot_modeltime_forecast()` to visualize the results.



```r
calibration_table %>%
  modeltime_forecast(actual_data = df) %>%
  plot_modeltime_forecast(.interactive = FALSE, .legend_show = TRUE)
```

```
## Using '.calibration_data' to forecast.
```

```
## Warning in max(ids, na.rm = TRUE): no non-missing arguments to max; returning -
## Inf
```

<img src="{{< blogdown/postref >}}index.en_files/figure-html/unnamed-chunk-20-1.png" width="672" />

## Accuracy (test set)

Next we can calculate the testing accuracy to compare the models.

- Use `modeltime_accuracy()` to generate the testing set metric as a tibble
- Use `table_modeltime_accuracy()` to generate a table.


```r
calibration_table %>%
  modeltime_accuracy() %>%
  knitr::kable() %>% 
  kableExtra::kable_styling()
```

<table class="table" style="margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:right;"> .model_id </th>
   <th style="text-align:left;"> .model_desc </th>
   <th style="text-align:left;"> .type </th>
   <th style="text-align:right;"> mae </th>
   <th style="text-align:right;"> mape </th>
   <th style="text-align:right;"> mase </th>
   <th style="text-align:right;"> smape </th>
   <th style="text-align:right;"> rmse </th>
   <th style="text-align:right;"> rsq </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:left;"> ARIMA(2,1,3)(0,0,2)[24] </td>
   <td style="text-align:left;"> Test </td>
   <td style="text-align:right;"> 119.46262 </td>
   <td style="text-align:right;"> 7.522759 </td>
   <td style="text-align:right;"> 4.296762 </td>
   <td style="text-align:right;"> 7.809373 </td>
   <td style="text-align:right;"> 129.38054 </td>
   <td style="text-align:right;"> 0.0989027 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:left;"> PROPHET </td>
   <td style="text-align:left;"> Test </td>
   <td style="text-align:right;"> 222.70503 </td>
   <td style="text-align:right;"> 13.973011 </td>
   <td style="text-align:right;"> 8.010125 </td>
   <td style="text-align:right;"> 15.178063 </td>
   <td style="text-align:right;"> 238.85610 </td>
   <td style="text-align:right;"> 0.0348801 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 3 </td>
   <td style="text-align:left;"> GLMNET </td>
   <td style="text-align:left;"> Test </td>
   <td style="text-align:right;"> 82.51981 </td>
   <td style="text-align:right;"> 5.277460 </td>
   <td style="text-align:right;"> 2.968025 </td>
   <td style="text-align:right;"> 5.356291 </td>
   <td style="text-align:right;"> 94.23809 </td>
   <td style="text-align:right;"> 0.1602888 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 4 </td>
   <td style="text-align:left;"> PROPHET W/ XGBOOST ERRORS </td>
   <td style="text-align:left;"> Test </td>
   <td style="text-align:right;"> 146.76916 </td>
   <td style="text-align:right;"> 9.226815 </td>
   <td style="text-align:right;"> 5.278908 </td>
   <td style="text-align:right;"> 9.705615 </td>
   <td style="text-align:right;"> 162.14158 </td>
   <td style="text-align:right;"> 0.0422390 </td>
  </tr>
</tbody>
</table>

## Analyse result


From the accuracy measures we can see that the est model is GLMNET.

## Refit

Refitting is a best practice before forecasting the future.

- `modeltime_refit()` here we gonna re train the model on the full data.
- `modeltime_forecast()` here we gonna forcecast on the date feature where
we we use the argument h to set the months or years.


```r
calibration_table %>%
  modeltime_refit(df) %>%
  modeltime_forecast(h = "1 month", actual_data = df) %>%
  plot_modeltime_forecast(.interactive = FALSE)
```

```
## frequency = 24 observations per 1 day
```

```
## Disabling yearly seasonality. Run prophet with yearly.seasonality=TRUE to override this.
## Disabling yearly seasonality. Run prophet with yearly.seasonality=TRUE to override this.
```

<img src="{{< blogdown/postref >}}index.en_files/figure-html/unnamed-chunk-22-1.png" width="672" />

Here we can see that the GLMNET seem to  be a little bit off compared to the other
models.
