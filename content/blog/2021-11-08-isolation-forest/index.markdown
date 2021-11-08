---
title: Isolation Forest
author: R package build
date: '2021-11-08'
slug: isolation-forest
categories:
  - Machine learning
tags: []
subtitle: ''
summary: 'A introduction to a useful unsupervised ML model for detecting outliers.'
authors: []
lastmod: '2021-11-08T05:11:53+01:00'
featured: no
draft: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---




## Introduction to `Isolation Forest`

In every modelling project one need to consider the outliers. It is because
models suffer in their performance when they are included. In are normal 
sitting the modelleren would maybe choose to drop them. But what if it
is the outliers we want to spot?

As an example we can take freauds and corruption. If we take som transaction
history of people credit card use and we observe some abnormility then 
it could be viwed as a person who tries to corrupt the system.

The are different methods to spot the outliers but what is important
is how can we try to model the outlier based on different features?

For this purpose we can introduce a unsupervised model called **Isolation Forest**.

## Isolation Forest

The main idea, which is different from other popular outlier detection methods, is that Isolation Forest explicitly identifies anomalies instead of profiling normal data points. Isolation Forest, like any tree ensemble method, is built on the basis of decision trees. In these trees, partitions are created by first randomly selecting a feature and then selecting a random split value between the minimum and maximum value of the selected feature.

In principle, outliers are less frequent than regular observations and are different from them in terms of values (they lie further away from the regular observations in the feature space). That is why by using such random partitioning they should be identified closer to the root of the tree (shorter average path length, i.e., the number of edges an observation must pass in the tree going from the root to the terminal node), with fewer splits necessary.

As with other outlier detection methods, an anomaly score is required for decusin
making and in the case fo IF it is deifned as:

$$
s(x,n) = 2^{- \frac{E(h(x))}{c(n)}}
$$

Each observation are given a acore and here we need to remember the following:

* A score close to 1 indicates anamalies.
* Score much smaller than 0.5 indicates normal observations.
* If all scores are close to 0.5 then the entire sample does not seem to have clearly distinct anamalies.

Now the theoru a in palce let us look at the data and try to modelling the outliers.

## Data

Here I am going to use a famouse data set about Number of NYC taxi passengers. 


```r
data <- read.csv("https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv")
data <-
  data %>%
  as_tibble() %>%
  mutate(timestamp = as_datetime(timestamp))
```

Let us take a look at the data:


```r
data %>%
  ggplot() +
  geom_line(aes(x = timestamp, y = value)) +
  labs(
    x = "",
    y = "",
    title = "NYC Taxi"
  )
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-2-1.png" width="2400" />

It is on half hourly basis but we accrecate it up to whole hourly.


```r
# resample timeseries to hourly
data <-
  data %>%
  group_by(timestamp = floor_date(timestamp, unit = "hour")) %>%
  summarise(value = sum(value, na.rm = TRUE))

data %>%
  head(5) %>%
  kableExtra::kable()
```

<table>
 <thead>
  <tr>
   <th style="text-align:left;"> timestamp </th>
   <th style="text-align:right;"> value </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> 2014-07-01 00:00:00 </td>
   <td style="text-align:right;"> 18971 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 2014-07-01 01:00:00 </td>
   <td style="text-align:right;"> 10866 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 2014-07-01 02:00:00 </td>
   <td style="text-align:right;"> 6693 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 2014-07-01 03:00:00 </td>
   <td style="text-align:right;"> 4433 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 2014-07-01 04:00:00 </td>
   <td style="text-align:right;"> 4379 </td>
  </tr>
</tbody>
</table>

## Modelling

Using `isotree`


```r
mod <-
  isolation.forest(
    data,
    ntrees = 500,
    nthreads = 1
  )

p <- predict(mod, data)
```

Let us see the outliers. 


```r
df <-
  data %>%
  cbind(p)

df %>%
  ggplot() +
  geom_line(aes(x = timestamp, y = value)) +
  geom_point(
    data = df %>% filter(p >= 0.6),
    aes(x = timestamp, y = value),
    color = "red",
    size = 3
  )
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-5-1.png" width="2400" />



## Conclusion

We have tried to modelling outliers the NYC data set. The model
succeed in plotting the most eyeboiling outliers but in also
included som that was actually normal.




