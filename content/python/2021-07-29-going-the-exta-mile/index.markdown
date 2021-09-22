---
title: Going the exta mile
author: R package build
date: '2021-07-29'
slug: going-the-exta-mile
categories: []
tags: []
subtitle: ''
summary: ''
authors: []
lastmod: '2021-07-29T13:32:06+02:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
runtime: shiny
---




## Introduction

For my interview at Frey I would like to show that I am a person that
want to *go the extra mile*. Therefore I have made a short report where 
I have taken three agricultural commodities:

- [Corn](https://finance.yahoo.com/quote/CORN/)
- [Soy](https://finance.yahoo.com/quote/soyb/)
- [Wheat](https://finance.yahoo.com/quote/WEAT/)

From those I have made a interactive report with `shiny` showing;
a historical plot, a summary and a table of the data. 

For trying to forecast the prices I have build a Vector Autoregressive model
and assume the prices a correlated to each other and predicted the next 90
days prices. 

The building of the report has been rushed so I have included a reflection
section with things that could be improved.

## Data

I am using the [Yahoo Fiance](https://finance.yahoo.com/) to extract the data for
the commidities. Here I use the package `tidyquant` to create a connection to
the url.


```r
corn  <- tq_get(
  "CORN", 
  get = "stock.prices",
  from = "2018-01-01")
```

```
## Registered S3 method overwritten by 'tune':
##   method                   from   
##   required_pkgs.model_spec parsnip
```

```r
wheat  <- tq_get(
  "WEAT", 
  get = "stock.prices",
  from = "2018-01-01")

soy <- tq_get(
  "SOYB",
  get = "stock.prices",
  from = "2018-01-01")
```

## The shiny component

Shiny is a way of making a dashboard or a interactive report It consist of
a UI and a Server element. 



```
## Error in fluidPage(titlePanel("Tabsets"), sidebarLayout(sidebarPanel(selectInput("stocks", : could not find function "fluidPage"
```

```
## Error in reactive({: could not find function "reactive"
```

```
## Error in output$plot <- renderHighchart({: object 'output' not found
```

```
## Error in renderPrint({: could not find function "renderPrint"
```

```
## Error in renderTable({: could not find function "renderTable"
```
