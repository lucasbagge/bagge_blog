---
title: Text mining - nøie trust pilot reviews
author: Lucas Bagge
date: '2020-12-29'
slug: text-mining-nøie-trust-pilot-reviews
categories:
  - text mining
tags:
  - tidytext
  - topic models
subtitle: ''
summary: ''
authors: []
lastmod: '2020-12-29T22:49:57+01:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---







## Introduction

In this analysis I am gonna scapes truspilot web page for reviews given by
customer for the skincare firm Nøie.

Here I am gonna use the data to make some topic modelling.

## Web scraping

For sciping the [trutpilot site](https://www.trustpilot.com/review/noie.com)



I am gonna make three functions: `get_ratings`, `get_reviews` and 
`get_reviewer_names` and combine it into a tibble with `get_data` to extract the
data.


```r
get_ratings <- function(html) {
  html %>%
    read_html() %>%
    html_nodes("body") %>%
    html_nodes(".star-rating") %>%
    as.character() %>%
    str_subset("medium") %>%
    str_extract("(\\d stjerne)") %>%
    str_remove(("( stjerne)")) %>%
    unlist()
}

get_reviews <- function(html) {
  html %>%
    read_html() %>%
    html_nodes(".review-content__body") %>%
    html_text() %>%
    str_trim() %>%
    unlist()
}

get_reviewer_names <- function(html) {
  html %>%
    read_html() %>%
    html_nodes(".consumer-information__name") %>%
    html_text() %>%
    str_trim() %>%
    unlist()
}

get_data <- function(html) {
  review <- get_reviews(html)
  names <- get_reviewer_names(html)
  ratings <- get_ratings(html)
  data <- tibble(
    reviewer = names,
    rating = ratings,
    review = review
  )
  data
}

urls <- cbind(c(url1, url2, url3, url4, url5, url6, url7))

url_list <- map(urls, get_data) %>%
  as.list()

data <- do.call(bind_rows, url_list)
```

Let us quick take a look at what I have extracted from the site:


```r
data %>%
  head()
```

```
## # A tibble: 6 x 3
##   reviewer                rating review                                         
##   <chr>                   <chr>  <chr>                                          
## 1 Lars Jensen             5      "Gode produkter og super service. Skru ned for…
## 2 Maria Cirkeline Rasmin… 5      "Super produkter\n                \n        \n…
## 3 Lis                     1      "2 cremer gav endnu mere uren hud + ulovlig ma…
## 4 Trine Holm              5      "Min hud slår altid ud om vinteren.\n         …
## 5 Camilla                 1      "Min hud er værre end nogensinde\n            …
## 6 Andrea Broe             5      "Stor anbefaling\n                \n        \n…
```
We see the following information:

- `reviewer` that is the person that has used the product.
- `rating` what the the reviewer has chosen to give the product on a scale from
  1-5.
- `review` is the comment given by the reviewer and the central aspect for this
  analysis.

In the next section I will drewll into into the preprocessig step for this text
mining task. 

## Loading and preparing the data

From the data we can see that the reviews are in Danish. Here we can use the `happyorsad` package
to compute a sentiment score for each review. Thease score are based on a Danish list of
sentiment words and put toheather by [Finn Årup Nielsen](https://www.dtu.dk/service/telefonbog/person?id=1755&cpid=&tab=1)



```r
df <-
  data %>%
  mutate(sentiment = map_int(review, happyorsad, "da")) %>%
  mutate(review = tolower(review)) %>%
  mutate(review = removeWords(
    review,
    c(
      "så", "3", "kan", "få", "får", "fik", "nøie",
      stopwords("danish")
    )
  ))
```

## Distribution of sentiment scores

In the density plot we see how sentiment scores are distributed with a median
score of 2. This a really good score and it is of interst to find out *why*
Nøie has a this great score and it also svore 4.5 rating out of 5.




```r
df %>%
  ggplot(aes(x = sentiment)) +
  geom_density(size = 1) +
  geom_vline(
    xintercept = median(df$sentiment),
    colour = "indianred", linetype = "dashed", size = 1
  ) +
  ggplot2::annotate("text",
    x = 15, y = 0.06,
    label = paste("median = ", median(df$sentiment)), colour = "indianred"
  ) +
  my_theme() +
  xlim(-40, 40)
```

<img src="/post/2020-12-29-text-mining-nøie-trust-pilot-reviews/index.en_files/figure-html/unnamed-chunk-6-1.png" width="2400" />

In a crude way we can put positive and negative reviews in separate data frames 
perform topic modelling on each in order to explore what reviewers lik and
dislike.

## Topic modelling for positive reviews


```r
df_pos <-
  df %>%
  filter(sentiment > 1) %>%
  unnest_tokens(word, review) %>%
  mutate(word = str_replace(word, "cremen", "creme")) %>%
  mutate(word = str_replace(word, "cremer", "creme")) %>%
  mutate(word = str_replace(word, "cremejeg", "creme")) %>%
  mutate(word = str_replace(word, "cremene", "creme"))
```

Before creating a so called **document term matrix** we need to count the
frequency of each word per document.


```r
words_pos <- df_pos %>%
  count(reviewer, word, sort = TRUE) %>%
  ungroup()
```

We want to use the famouse `Latent Dirichlet Allocation` algorithme for topic
modelling. To use this we need to create our DTM and here we use `tidytext` function
`cast_dtm` to do that. 


```r
reviewDTM_pos <- words_pos %>%
  cast_dtm(reviewer, word, n)
```

LDA assumes that every document is a mixture of topics, and every topic is a 
mixture of words. The k argument is used to specify the desired amount of topics 
that we want in our model. Let´s create a two-topic mode.


```r
reviewLDA_pos <- LDA(reviewDTM_pos, k = 3, control = list(seed = 123))
```

The following table shows how many reviews that are assigned to each topic


```r
tibble(topics(reviewLDA_pos)) %>%
  group_by(`topics(reviewLDA_pos)`) %>%
  count() %>%
  kable() %>%
  kable_styling(
    full_width = FALSE,
    position = "left"
  )
```

<table class="table" style="width: auto !important; ">
 <thead>
  <tr>
   <th style="text-align:right;"> topics(reviewLDA_pos) </th>
   <th style="text-align:right;"> n </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 50 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 29 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 3 </td>
   <td style="text-align:right;"> 30 </td>
  </tr>
</tbody>
</table>

It is also possible to get the per-topic word probabilities or 'beta'


```r
topics_pos <- tidy(reviewLDA_pos, matrix = "beta")
topics_pos
```

```
## # A tibble: 3,336 x 3
##    topic term          beta
##    <int> <chr>        <dbl>
##  1     1 creme     0.0611  
##  2     2 creme     0.0320  
##  3     3 creme     0.0499  
##  4     1 tak       0.000964
##  5     2 tak       0.00569 
##  6     3 tak       0.00600 
##  7     1 hud       0.0216  
##  8     2 hud       0.0229  
##  9     3 hud       0.0275  
## 10     1 produkter 0.0229  
## # … with 3,326 more rows
```

Now we can find the words with the highest beta. Here we choose the top five 
words which we will show in a plot.


```r
top_terms_pos <- topics_pos %>%
  group_by(topic) %>%
  top_n(5, beta) %>%
  ungroup() %>%
  arrange(topic, -beta) %>%
  mutate(order = rev(row_number()))
```


```r
# plot_pos <-
top_terms_pos %>%
  ggplot(aes(order, beta)) +
  ggtitle("Positive review topics") +
  geom_col(show.legend = FALSE, fill = "steelblue") +
  scale_x_continuous(
    breaks = top_terms_pos$order,
    labels = top_terms_pos$term,
    expand = c(0, 0)
  ) +
  facet_wrap(~topic, scales = "free") +
  coord_flip(ylim = c(0, 0.02)) +
  my_theme() +
  theme(axis.title = element_blank())
```

<img src="/post/2020-12-29-text-mining-nøie-trust-pilot-reviews/index.en_files/figure-html/unnamed-chunk-14-1.png" width="2400" />

## Word co-occurrence within reviews




```r
pairs_plot_pos <- word_pairs_pos <-
  df_pos %>%
  pairwise_count(word, reviewer, sort = TRUE) %>%
  filter(n >= 10) %>%
  graph_from_data_frame() %>%
  ggraph(layout = "fr") +
  geom_edge_link(aes(edge_alpha = n, edge_width = n), edge_colour = "steelblue") +
  ggtitle("Positive word pairs") +
  geom_node_point(size = 5) +
  geom_node_text(aes(label = name),
    repel = TRUE,
    point.padding = unit(0.2, "lines")
  ) +
  my_theme() +
  theme(
    axis.title = element_blank(),
    axis.text = element_blank(),
    axis.ticks = element_blank()
  )

grid.arrange(pairs_plot_pos)
```

<img src="/post/2020-12-29-text-mining-nøie-trust-pilot-reviews/index.en_files/figure-html/unnamed-chunk-15-1.png" width="2400" />

