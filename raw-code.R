##############################################################################
############################################################################## 
# PRELIMS ####################################################################
############################################################################## 
############################################################################## 

############################################################################## 
# PACKAGES 
############################################################################## 

library(tidyverse)
library(arrow)
library(tidytext)
library(tokenizers)
library(yardstick)
library(tidymodels)
library(textrecipes)
library(glmnet)
library(textfeatures)

############################################################################## 
# GLOBAL SETTINGS FOR PLOTS 
##############################################################################

# class label colors
POS_COLOR <- "#03a5fc" # bluey color 
NEG_COLOR <- "#fca503" #orangey color 

# how transparent fills
ALPHA <- 0.6


##############################################################################
############################################################################## 
# PRELIMS ####################################################################
############################################################################## 
############################################################################## 

############################################################################## 
# DATA READ IN
##############################################################################

imdb <- arrow::read_parquet("data/imdb-sample.parquet")
str(imdb)

############################################################################## 
# NUM OBS BY SPLIT 
##############################################################################

imdb %>% count(split) 

############################################################################## 
# CLASS DISTRIBUTION BY SPLIT 
##############################################################################

imdb %>% 
  group_by(split, label) %>% 
  summarise(value_counts = n()) %>%
  mutate(`normalized_counts (%)` = round((value_counts / sum(value_counts) * 100), 2)
  )


############################################################################## 
# SPLIT INTO SEPARATE DATAFRAMES 
##############################################################################

train_imdb <- imdb %>% filter(split == "train") %>% dplyr::select(text, label)
dev_imdb <- imdb %>% filter(split == "dev") %>% dplyr::select(text, label)
test_imdb <- imdb %>% filter(split == "test") %>% dplyr::select(text, label)

############################################################################## 
# EXAMPLES 
##############################################################################

# 5 examples of positive reviews 
set.seed(123)
train_imdb %>%
  filter(label == "pos") %>%
  sample_n(3) %>%
  pull(text)

# 5 examples of negative reviews 
set.seed(123)
train_imdb %>%
  filter(label == "neg") %>%
  sample_n(3) %>%
  pull(text)

############################################################################## 
# CLEAN MARK UP ETC - N.B JUST FOR EXAMPLE
##############################################################################

simple_clean <- function(text){
  str_replace_all(text, '\u0085|<br />', '')
}

simple_clean("or something?\u0085<br /><br />")

############################################################################## 
# REVIEW LENGTH 
##############################################################################

# get the review legnths 
review_lengths <- lengths(tokenize_words(train_imdb$text), use.names = F)
# stick into a dataframe
review_lengths_labels_df <- tibble(label = train_imdb$label, number_of_words = review_lengths)
# plot counts by label 
review_lengths_labels_df %>% ggplot(aes(x = number_of_words, fill = label)) +
  geom_histogram(bins = 50, alpha = 0.7)

############################################################################## 
# NEGATION
##############################################################################

list_of_negatives <- c("not", "never", "no", "nowhere", "nobody", "yet", "hardly", "barely")
contracted_negative_pattern <- "n't"

mean_negative_rate <- train_imdb %>% 
  mutate(review_id = 1:nrow(train_imdb)) %>%
  unnest_tokens(words, text) %>%
  mutate(is_negative = (words %in% list_of_negatives)|
           str_detect(words, contracted_negative_pattern)) %>%
  select(review_id, is_negative) %>%
  group_by(review_id) %>%
  summarize(mean_negative_rate = mean(is_negative)) %>%
  mutate(mean_negative_rate = mean_negative_rate * 10^6) %>%
  select(mean_negative_rate)

mean_negative_rate_df <- tibble(mean_negative_rate, label = train_imdb$label)

mean_negative_rate_df %>% ggplot(aes(x = mean_negative_rate, fill = label)) +
  geom_histogram(bins = 50, alpha = 0.7)

############################################################################## 
# CLASS-DISTINGUISHING WORDS 
##############################################################################

pos_neg_ratio_df <- train_imdb %>%
  unnest_tokens(word, text) %>%
  group_by(label) %>%
  count(word, sort = TRUE) %>%
  filter(n > 25) %>%
  pivot_wider(names_from = label, values_from = n) %>%
  mutate(pos = replace_na(pos, 1) + 1, neg = replace_na(neg, 1) + 1) %>%
  mutate(pos_neg_ratio = log(pos/neg)) %>%
  arrange(desc(pos_neg_ratio)) 

head(pos_neg_ratio_df)

pos_neg_ratio_df %>%
  head(10) %>%
  ggplot(aes(x = reorder(word, pos_neg_ratio), y = pos_neg_ratio)) +
  geom_bar(stat = "identity", fill = POS_COLOR, alpha = 0.7) +
  coord_flip() +
  labs(x="", y = "log[count(positive+1)/count(negative+1)]", 
       title = "words associated with +'ve reviews") 

pos_neg_ratio_df %>%
  tail(10) %>%
  ggplot(aes(x = reorder(word, -pos_neg_ratio), y = pos_neg_ratio)) +
  geom_bar(stat = "identity", fill = NEG_COLOR, alpha = 0.7) +
  coord_flip() +
  labs(x="", y = "log[count(positive+1)/count(negative+1]", 
       title = "words associated with -'ve reviews")

##############################################################################
############################################################################## 
# EVAL METRIC ################################################################
############################################################################## 
############################################################################## 

# make up some data and fake predictions  
y_true <- factor(c(1,1,1,1,1,0,0,0,0,0))
y_pred <- factor(c(0,1,1,0,0,1,1,1,1,0))

# if your results are in a dataset 
results <- tibble(y_true, y_pred)
f_meas(results, y_true, y_pred, estimator = "macro")

# as vectors 
f_meas_vec(y_true, y_pred, estimator = "macro")

##############################################################################
############################################################################## 
# LEXICON APPROACH ###########################################################
############################################################################## 
############################################################################## 

liu_lex <- get_sentiments("bing")
head(liu_lex)

liu_lex %>% 
  count(sentiment)

get_sentiment_score <- function(data){
  tokens <- unlist(tokenize_words(data))
  tokens_df <- tibble(word = tokens)
  sentiment_tokens <- inner_join(tokens_df, liu_lex, by = "word")
  sentiment_tokens$sentiment <- recode(sentiment_tokens$sentiment, "positive" = 1, "negative" = 0)
  score <- mean(sentiment_tokens$sentiment) 
  if (is.nan(score)){
    return(sample(c("pos", "neg"), 1))
  }
  else if (score > 0.5){
    return("pos")
  } 
  else {
    return("neg")
  }
}

train_lexicon_preds <- sapply(train_imdb$text, get_sentiment_score, USE.NAMES = FALSE) 
train_lexicon_result <- f_meas_vec(factor(train_imdb$label), factor(train_lexicon_preds), 
                                   estimator = "macro")
sprintf("Train macro F1 using lexicon approach: %.4f", train_lexicon_result)

dev_lexicon_preds <- sapply(dev_imdb$text, get_sentiment_score, USE.NAMES = FALSE) 
dev_lexicon_result <- f_meas_vec(factor(dev_imdb$label), factor(dev_lexicon_preds), 
                                 estimator = "macro")
sprintf("Dev  macro F1 using lexicon approach: %.4f", dev_lexicon_result)

##############################################################################
############################################################################## 
# TRADITIONAL MACHINE LEARNING ###############################################
############################################################################## 
############################################################################## 


############################################################################## 
# PREPROCESSING RECIPES 
############################################################################## 

# -----------------------------------------------------------------------------
# unigrams --------------------------------------------------------------------
# -----------------------------------------------------------------------------

# write recipe 
unigram_rec <- recipe(label ~ text, data = train_imdb) %>%
  step_tokenize(text, engine = "tokenizers", 
                token = "words",
                options = list(lowercase = TRUE, strip_punct = FALSE)) %>%
  step_tokenfilter(text, max_tokens = 2000) %>%
  step_tf(text) %>%
  step_normalize(all_predictors()) 

# return an updated recipe with the estimates
unigram_prep <- prep(unigram_rec)

# -----------------------------------------------------------------------------
# bigrams ---------------------------------------------------------------------
# -----------------------------------------------------------------------------

# write recipe 
bigram_rec <- recipe(label ~ text, data = train_imdb) %>%
  step_tokenize(text, engine = "tokenizers", 
                token = "words", 
                options = list(lowercase = TRUE, strip_punct = FALSE)) %>%
  step_ngram(text, min_num_tokens = 2, num_tokens = 2) %>%
  step_tokenfilter(text, max_tokens = 2000) %>%
  step_tf(text) %>%
  step_normalize(all_predictors())

# return an updated recipe with the estimates
bigram_prep <- prep(bigram_rec)


# -----------------------------------------------------------------------------
# unigrams + bigrams ----------------------------------------------------------
# -----------------------------------------------------------------------------

# write recipe 
unigram_bigram_rec <- recipe(label ~ text, data = train_imdb) %>%
  step_tokenize(text, engine = "tokenizers", 
                token = "words", 
                options = list(lowercase = TRUE, strip_punct = FALSE)) %>%
  step_ngram(text, min_num_tokens = 1, num_tokens = 2) %>%
  step_tokenfilter(text, max_tokens = 2000) %>%
  step_tf(text) %>%
  step_normalize(all_predictors())

# return an updated recipe with the estimates
unigram_bigram_prep <- prep(unigram_bigram_rec)

# -----------------------------------------------------------------------------
# unigrams (length normalized) ------------------------------------------------
# -----------------------------------------------------------------------------

# write recipe 
unigram_len_norm_rec <- recipe(label ~ text, data = train_imdb) %>%
  step_tokenize(text, engine = "tokenizers", 
                token = "words",
                options = list(lowercase = TRUE, strip_punct = FALSE)) %>%
  step_tokenfilter(text, max_tokens = 2000) %>%
  step_tf(text, weight_scheme = "term frequency") %>%
  step_normalize(all_predictors()) 

# return an updated recipe with the estimates
unigram_len_norm_prep <- prep(unigram_len_norm_rec)

# -----------------------------------------------------------------------------
# unigrams (binary) -----------------------------------------------------------
# -----------------------------------------------------------------------------

# write recipe 
unigram_binary_rec <- recipe(label ~ text, data = train_imdb) %>%
  step_tokenize(text, engine = "tokenizers", 
                token = "words",
                options = list(lowercase = TRUE, strip_punct = FALSE)) %>%
  step_tokenfilter(text, max_tokens = 2000) %>%
  step_tf(text, weight_scheme = "binary") %>%
  step_mutate_at(all_predictors(), fn  = as.numeric)  %>%
  step_normalize(all_predictors()) 

# return an updated recipe with the estimates
unigram_binary_prep <- prep(unigram_binary_rec)

# -----------------------------------------------------------------------------
# unigrams (tfidf) -----------------------------------------------------------
# -----------------------------------------------------------------------------

# write recipe 
unigram_tfidf_rec <- recipe(label ~ text, data = train_imdb) %>%
  step_tokenize(text, engine = "tokenizers", 
                token = "words",
                options = list(lowercase = TRUE, strip_punct = FALSE)) %>%
  step_tokenfilter(text, max_tokens = 2000) %>%
  step_tfidf(text) %>%
  step_normalize(all_predictors()) 

# return an updated recipe with the estimates
unigram_tfidf_prep <- prep(unigram_tfidf_rec)

# -----------------------------------------------------------------------------
# unigrams + linguistic features ----------------------------------------------
# -----------------------------------------------------------------------------

# write recipe 
unigram_lingfeats_rec <- recipe(label ~ text, data = train_imdb) %>%
  step_textfeature(text, keep_original_cols = TRUE) %>%
  step_tokenize(text, engine = "tokenizers", 
                token = "words",
                options = list(lowercase = TRUE, strip_punct = FALSE)) %>%
  step_tokenfilter(text, max_tokens = 2000) %>%
  step_tf(text) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors()) 

# return an updated recipe with the estimates
unigram_lingfeats_rep <- prep(unigram_lingfeats_rec)


############################################################################## 
# MODEL SET UPS 
############################################################################## 

# -----------------------------------------------------------------------------
# unigrams --------------------------------------------------------------------
# -----------------------------------------------------------------------------

unigram_spec <- logistic_reg(penalty = 0.1, mixture = 0, engine = "glmnet")

unigram_wf <- workflow() %>%
  add_recipe(unigram_rec) %>%
  add_model(unigram_spec)


# -----------------------------------------------------------------------------
# bigrams ---------------------------------------------------------------------
# -----------------------------------------------------------------------------

bigram_spec <- logistic_reg(penalty = 0.1, mixture = 0, engine = "glmnet")

bigram_wf <- workflow() %>%
  add_recipe(bigram_rec) %>%
  add_model(bigram_spec)

# -----------------------------------------------------------------------------
# unigrams + bigrams ----------------------------------------------------------
# -----------------------------------------------------------------------------

unigram_bigram_spec <- logistic_reg(penalty = 0.1, mixture = 0, engine = "glmnet")

unigram_bigram_wf <- workflow() %>%
  add_recipe(unigram_bigram_rec) %>%
  add_model(unigram_bigram_spec)


# -----------------------------------------------------------------------------
# unigrams (length normalized) ------------------------------------------------
# --------------------------------------------------------------------------

unigram_len_norm_spec <- logistic_reg(penalty = 0.1, mixture = 0, engine = "glmnet")

unigram_len_norm_wf <- workflow() %>%
  add_recipe(unigram_len_norm_rec) %>%
  add_model(unigram_len_norm_spec)


# -----------------------------------------------------------------------------
# unigrams (binary) -----------------------------------------------------------
# -----------------------------------------------------------------------------

unigram_binary_spec <- logistic_reg(penalty = 0.1, mixture = 0, engine = "glmnet")

unigram_binary_wf <- workflow() %>%
  add_recipe(unigram_binary_rec) %>%
  add_model(unigram_binary_spec)


# -----------------------------------------------------------------------------
# unigrams (tfidf) -----------------------------------------------------------
# -----------------------------------------------------------------------------

unigram_tfidf_spec <- logistic_reg(penalty = 0.1, mixture = 0, engine = "glmnet")

unigram_tfidf_wf <- workflow() %>%
  add_recipe(unigram_tfidf_rec) %>%
  add_model(unigram_tfidf_spec)



# -----------------------------------------------------------------------------
# unigrams + linguistic features ----------------------------------------------
# -----------------------------------------------------------------------------

unigram_lingfeats_spec <- logistic_reg(penalty = 0.1, mixture = 0, engine = "glmnet")

unigram_lingfeats_wf <- workflow() %>%
  add_recipe(unigram_lingfeats_rec) %>%
  add_model(unigram_lingfeats_spec)



############################################################################## 
# MODEL TRAINING 
############################################################################## 

unigram_model <- fit(unigram_wf, train_imdb)

bigram_model <- fit(bigram_wf, train_imdb)

unigram_bigram_model <- fit(unigram_bigram_wf, train_imdb)

unigram_len_norm_model <- fit(unigram_len_norm_wf, train_imdb)

unigram_binary_model <- fit(unigram_binary_wf, train_imdb)

unigram_tfidf_model <- fit(unigram_tfidf_wf, train_imdb)

unigram_lingfeats_model <- fit(unigram_lingfeats_wf, train_imdb)



############################################################################## 
# MODEL EVALUATION 
##############################################################################

get_results <- function(model, y_true, newdata){
  y_pred <- predict(model, newdata)
  result <- f_meas_vec(factor(y_true), y_pred$.pred_class, 
             estimator = "macro")
  return(result)
}

unigram_res <- get_results(unigram_model, dev_imdb$label, dev_imdb)

bigram_res <- get_results(bigram_model, dev_imdb$label, dev_imdb)

unigram_bigram_res <- get_results(unigram_bigram_model, dev_imdb$label, dev_imdb)

unigram_len_norm_res <- get_results(unigram_len_norm_model, dev_imdb$label, dev_imdb)

unigram_binary_res <- get_results(unigram_binary_model, dev_imdb$label, dev_imdb)

unigram_tfidf_res <- get_results(unigram_tfidf_model, dev_imdb$label, dev_imdb)

unigram_lingfeats_res <- get_results(unigram_lingfeats_model, dev_imdb$label, dev_imdb)

experiment_names_vec <- c("unigram (raw counts)",
                      "bigram (raw counts)",
                      "unigram + bigram (raw counts)",
                      "unigram (normalized counts)",
                      "unigram (binary)",
                      "unigram (tfidf)",
                      "unigram (raw counts) + linguistic features")

experiment_results_vec <- c(unigram_res, bigram_res, unigram_bigram_res, 
        unigram_len_norm_res, unigram_binary_res, 
        unigram_tfidf_res, unigram_lingfeats_res) * 100
          
experiment_results_df <- tibble(experiment = experiment_names_vec, macro_f1 = experiment_results_vec)


##############################################################################
############################################################################## 
# NEURAL NETS  ###############################################################
############################################################################## 
############################################################################## 

############################################################################## 
# PREPROCESSING RECIPE 
############################################################################## 

# get some pretrained embeddings 
glove_embeddings <- read_delim("data/glove6b100d.txt", delim = "\t")

# write recipe 
embeds_ffnn_rec <- recipe(label ~ text, data = train_imdb) %>%
  step_tokenize(text, engine = "tokenizers", 
                token = "words",
                options = list(lowercase = TRUE, strip_punct = TRUE)) %>%
  step_word_embeddings(text, embeddings = glove_embeddings ) %>%
  step_normalize(all_predictors()) 

# return an updated recipe with the estimates
embeds_ffnn_prep <- prep(embeds_ffnn_rec )

############################################################################## 
# MODEL SET UPS 
############################################################################## 

embeds_spec <- mlp(mode = "classification", 
                        hidden_units = 2,
                        epochs = 25,
                        # dropout = 0.3,
                        activation = "relu") %>%
  set_engine('keras', validation_split = 0.05)

embeds_wf <- workflow() %>%
  add_recipe(embeds_rec) %>%
  add_model(embeds_spec)

############################################################################## 
# MODEL TRAINING 
############################################################################## 


embeds_model <- fit(embeds_wf, train_imdb)

############################################################################## 
# MODEL EVALUATION 
##############################################################################

embeds_res <- get_results(embeds_model, dev_imdb$label, dev_imdb)
embeds_res 

##############################################################################
############################################################################## 
# OVERALL ASSESSMENT #########################################################
############################################################################## 
############################################################################## 

experiment_names_vec <- c("lexicon",
                          "unigram (raw counts) in L2 logit",
                          "bigram (raw counts) in L2 logit",
                          "unigram + bigram (raw counts) in L2 logit",
                          "unigram (normalized counts) in L2 logit",
                          "unigram (binary) in L2 logit",
                          "unigram (tfidf) in L2 logit",
                          "unigram (raw counts) + linguistic features in L2 logit",
                          "glove embeds in ffnn in L2 logit")

experiment_results_vec <- c(dev_lexicon_result, unigram_res, bigram_res, unigram_bigram_res, 
                            unigram_len_norm_res, unigram_binary_res, 
                            unigram_tfidf_res, unigram_lingfeats_res, embeds_res) * 100

experiment_results_df <- tibble(experiment = experiment_names_vec, macro_f1 = experiment_results_vec)

experiment_results_df %>% 
  arrange(desc(macro_f1))


############################################################################## 
# MODEL COMPARISON
##############################################################################

unigram_binary_preds <- predict(unigram_binary_model,dev_imdb)

model_is_correct <- tibble(lexicon_is_correct = dev_lexicon_preds == dev_imdb$label,
                           unigram_binary_is_correct = unigram_binary_preds  == dev_imdb$label)

model_is_correct_ct <- table(model_is_correct)

mcnemar.test(model_is_correct_ct)


############################################################################## 
# QUALITATIVE ERROR ANALYSIS
##############################################################################

set.seed(123)

tibble(true_label = dev_imdb$label,
       predicted_label = unigram_binary_preds$.pred_class,
       is_correct = model_is_correct$unigram_binary_is_correct,
       text = dev_imdb$text) %>% 
  filter(!is_correct) %>%
  sample_n(5) %>% 
  View()


