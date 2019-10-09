library(tidyverse)
library(reshape2)
library(RColorBrewer)
library(ggplot2)
library(dplyr)
library(gplots)
library(cluster)
library(ggdendro)

utils = new.env()

#### DATA READ-IN ####

# This function combines the summary file (generated during training)
#   and the evaluation file (generated after training) in order to
#   create a dataframe that contains all training hyperparameters and
#   accuracies on each word in the test/training/validation dataset
utils$combine = function (results_folder, words_file = "final_eval-agg.csv") {
  # TODO handle the fact that data is stored in different directories
  
  accuracy_file = paste(results_folder, words_file, sep="/")
  summary_file = paste(results_folder, "summary.csv", sep="/")
  
  d <- read.csv(accuracy_file)
  d$len_fac = as.factor(d$len)
  
  # melt so that we have one model-word pairing per line
  # columns that define each model
  model_cols = grep("^X[[:digit:]]+.[[:digit:]]+$", names(d), value=T)
  d = melt(d, measure.vars = model_cols)
  # separate model columns into condition/runs
  d = transform(d, model = colsplit(as.character(variable), pattern = "\\.", names = c('condition', 'run')))
  # remove X from the beginning of each condition
  d$model.condition = sub('.', '', d$model.condition)
  
  # read in information about conditions
  summary_data <- read.csv(summary_file)
  
  # truncate summary file data to just the essentials
  #   removing the loss/perplexity/time/learning rate columns, since
  #   we're not interested in timecourse data
  drop_cols = grep("(_(loss|ppl|time|lr)|epoch)$", names(summary_data), value=T)
  summary_data = subset(summary_data, subset = epoch == 1)
  summary_data = summary_data[, !(names(summary_data) %in% drop_cols)]
  
  combined_data <- merge(d, summary_data, by.x = c("model.condition", "model.run"), by.y = c("condition", "run"))
  
  # make negative to match ngram results
  combined_data$value = -combined_data$value
  combined_data['ngram'] = FALSE
  
  # rename columns for clarity
  names(combined_data)[names(combined_data) == 'model.condition'] <- 'condition'
  names(combined_data)[names(combined_data) == 'model.run'] <- 'run'
  
  return(combined_data)
}

# function to aggregate each model run by averaging
utils$agg = function (dataframe) {
  agg_hyper = dataframe %>% 
    group_by(run, condition) %>% 
    mutate(avg=mean(value))
  # exclude by-word variables
  by_words <- c("word","len", "len_fac", "value")
  agg_hyper = agg_hyper[ , !(names(agg_hyper) %in% by_words)]
  agg_hyper = unique(agg_hyper)
  
  return (agg_hyper)
}

# function to summarize stats by average accuracy
utils$summarize_acc <- function (combined_data) {
  summary_stats <- combined_data %>%
    group_by(run, condition, run_cond, phonol_emb) %>%
    summarise_at(vars(value), mean)
  names(summary_stats)[names(summary_stats) == 'value'] <- 'avg_acc'
 
  return(summary_stats) 
}

utils$get_best <- function (acc_summary, phonol_emb_cond = "True") {
  subset_stats = subset(acc_summary, phonol_emb == phonol_emb_cond)
  best_info = subset_stats[which.max(subset_stats$avg_acc),]
  best_model = paste(best_info['condition'], best_info['run'], sep="-")
}

# function to read output from ngram testing
utils$ngram_results <- function (ngram_file) {
  # read in ngram data
  ngram <- read.csv(ngram_file)
  ngram$len_fac = as.factor(ngram$len)
  
  # change of base
  logprobs = ngram
  logprobs[,3:11] = ngram[,3:11] / log(2.71828182845904, base=10)
  summary(logprobs)
  
  # melt into similar format as the other results
  logprobs['ngram'] = TRUE
  keep_cols = grep("^(X3gram|split_name|word|len.*|ngram)$", names(logprobs), value=T)
  ngram_subset = subset(logprobs, select = keep_cols)
  colnames(ngram_subset)[colnames(ngram_subset)=="X3gram"] <- "value"
  
  return(ngram_subset)
}

#### PLOTTING ####

# function that plots a folder automatically
# split by patience
utils$plot_folder_var <- function (folder,
                                   var = "patience",
                                   about = '',
                                   ngram_mean = FALSE,
                                   words_file = "final-words.csv") {
  # process and plot data from test data only
  test_only = subset(utils$combine(folder, words_file), split_name == "test")
  # summarize means over words
  agg_test = test_only %>% 
    group_by(run, condition) %>% 
    summarize(avg=mean(value))
  # import hyperparameter from the summary csv
  agg_hyper = merge(agg_test, read.csv(paste(folder, "summary.csv", sep="/")),
                             by.x = c("condition", "run"),
                             by.y = c("condition", "run"))
  
  # generate plot
  result_plot <- ggplot(agg_hyper, aes(x=as.factor(unlist(agg_hyper[var])), y=avg)) +
      geom_boxplot(aes(fill=phonol_emb)) +
      labs(y = "Average natural log likelihood \nover words",
           title="Average accuracies per model",
           x = "Patience",
           subtitle=about) + 
      ylim(limits = c(-29.2,-22)) +
      NULL
  
  if (ngram_mean != FALSE)
    result_plot = result_plot + geom_abline(intercept = ngram_mean, slope = 0)
  
  return(result_plot)
}

# wrapper function that plots a folder automatically
utils$plot_folder <- function (folder, about = '', ngram_mean = FALSE, split = "test",
                               color_var = 'phonol_emb', color_lab = NA, words_file = "final-words.csv") {
  return(utils$plot_split(utils$combine(folder, words_file), about, ngram_mean,
                          split,
                          color_var,
                          color_lab))
}

# function that selects a specific split in the data frame
utils$plot_split <- function (dataframe, about = '', ngram_mean = FALSE, split = "test",
                              color_var = 'phonol_emb', color_lab = NA) {
  split_filtered = subset(dataframe, split_name == split)
  return(utils$plot_data(split_filtered, about, ngram_mean,
                         color_var,
                         color_lab))
}
  

# funtion that plots a dataframe
utils$plot_data <- function (dataframe, about = '', ngram_mean = FALSE,
                             color_var = 'phonol_emb',
                             color_lab = NA,
                             plot_type = 'box',
                             split = NA
                              ) {
  # which data to plot?
  if (!is.na(split)) {
    df = subset(dataframe, split_name == split)
    
    # set caption automatically, including number of words
    if (about == '') {
      n_words <- nrow(subset(df, run_cond == "0-0", select = word))
      about = paste0('Word set: ',split,' (',n_words,' words)')
    }
  } else {
    df = dataframe
  }
  
  agg_hyper = utils$agg(dataframe)
  
  # generate plot
  result_plot <- ggplot(agg_hyper, aes(x=as.factor(unlist(agg_hyper["phonol_emb"])), y=avg)) +
    # ylim(limits = c(-27,-22)) +
    NULL
  
  # point or box
  if (plot_type == 'box') {
    result_plot = result_plot +
      geom_boxplot(aes(fill=as.factor(unlist(agg_hyper[color_var])))) +
      scale_fill_manual(values = brewer.pal(n=3, name="PuOr")[c(1,3)])
  } else {
    result_plot = result_plot +
      geom_point(aes(color=as.factor(unlist(agg_hyper[color_var])))) +
      scale_color_manual(values = brewer.pal(n=3, name="PuOr")[c(1,3)])
  }
  
  # aesthetic stuff
  result_plot = result_plot +
    labs(y = "Average natural log likelihood \nover all words in set",
                                   x = "Condition",
                                   title="Average log likelihoods per model",
                                   fill = color_lab,
                                   subtitle=about) + 
    scale_x_discrete(breaks = c("True", "False"),
                     labels= c("Feature-aware", "Feature-naive")) +
    theme(axis.text = element_text(size = 13)) +
    theme(text = element_text(size = 13))
  
  # remove legend if redundant
  if (color_var == "phonol_emb") {
    result_plot = result_plot + theme(legend.position = "none")
  }
  
  if (ngram_mean != FALSE) {
    result_plot = result_plot +
                  geom_abline(intercept = ngram_mean, slope = 0) + # add line
                  ylim(limits = c(NA, ceiling(ngram_mean))) # set limits so the line will fit
  }
    
  return(result_plot)
}
utils$plot_timecourse <- function (combined_df, summary_df, split = NA) {
  if (is.na(split)) {
    lab_st = ""
    lab_title = "Learning rate across epochs"
    lab_y = "learning rate"
    y_measure = summary_df$current_lr
  } else {
    lab_title = "Loss function across epochs"
    lab_y = "Average per-phone natural log probability"
    n_words <- nrow(subset(combined_df, split_name == split & run_cond == "0-0", select = word))
    lab_st = paste0("Word set: ", split, " (", n_words, " words)")
    if (split == "valid") {
      y_measure <- summary_df$val_loss
    } else {
      y_measure <- summary_df$train_loss
    }
  }
  ggplot(data = summary_df, aes(x=epoch, y=y_measure, group=interaction(condition,run), colour=as.factor(phonol_emb))) +
    geom_line() +
    labs(y = lab_y,
         color = "Phonological\nembedding",
         title = lab_title,
         subtitle = lab_st)
}

#### STATS FUNCTIONS ####

utils$wilcox <- function (df, test_var, agg = TRUE) {
  if (agg == TRUE) {
    data = utils$agg(df)
    names(data)[names(data) == 'avg'] <- 'value'
  } else {
    data = df
  }

  # check factor levels
  levs = levels(as.factor(unlist(data[test_var])))
  if (length(levs) != 2) {
    print("Sorry! Test cannot be performed, variable doesn't have 2 levels")
  } else {
    # split up the data
    set1 <- aggreg[aggreg["phonol_emb"] == levs[1],]
    set2 <- aggreg[aggreg["phonol_emb"] == levs[2],]

    print(wilcox.test(set1$value, set2$value))

    if (mean(set1$value) > mean(set2$value)) {
      print(paste(levs[1],"is better",sep = " "))
    } else {
      print(paste(levs[2],"is better",sep = " "))
    }

    return(wilcox.test(set1$value, set2$value))
    # TODO split up the data
  }
}

#### CLUSTERING FUNCTIONS #####

utils$ssd <- function(x) {
  return (sum (x - mean(x) )^2)
}
utils$condition_number <- function(summary_acc_table, variable, value) {
  s = subset(summary_acc_table, select = summary_acc_table$model.condition)
  return(s)
}
# create a heat map using the specified data
utils$create_heat <- function(embs, emb_name, title = "") {
  # turn into matrix
  model_emb <- t(as.matrix(as.data.frame(embs[,emb_name])))
  
  # create heat map
  heatmap.2(model_emb,
            hclustfun = agnes,
            trace = "none",
            col = colorRampPalette(brewer.pal(9,'PuOr'))(n=40),
            main = title,
            xlab = "Embedding Dimension",
            labRow = gsub("\n", "</s>", rownames(model_emb)),
            ylab = "Phonetic Segment",
            cexRow = 1.9,
            srtCol = 45,
            # par(cex.lab = 3)
            cexCol = 1.9
  )
}

# Function to get a set of (trained or untrained) embeddings from a results folder
utils$get_embeddings <- function(results_folder, trained) {
  
  if (trained == TRUE) {
    when = "after"
  } else {
    when = "before"
  }

  # read in all embedding files from folder
  embedding_files = list.files(results_folder, pattern = paste0("emb-",when,".txt"), full.names=TRUE, recursive="TRUE")

  # read embeddings into matrix
  embs <- sapply(embedding_files, function(x) {
    dt <- t(read.csv(x))
    return(as.list(as.data.frame(dt)))
  })

  # determine feature set based on dimensions of embs
  # TODO: change this for set/unsets
  if (length(embs[[1,1]]) == 34) {
    features = c('duration', 'laryngeal', 'nasal', 'suprasegmental', 'manner_V', 'manner_nasal', 'manner_stop', 'manner_fric', 'manner_appx', 'manner_latappx', 'backness_Z', 'backness_Y', 'backness_X', 'backness_W', 'backness_U', 'height_E', 'height_D', 'height_Q', 'height_B', 'height_A', 'rounding', 'cplace_B', 'cplace_F', 'cplace_D', 'cplace_T', 'cplace_S', 'cplace_R', 'cplace_J', 'cplace_K', 'cplace_H', 'art2', 'lat', 'EOW', 'start')
  } else {
    features = c('syll', 'cons', 'appx', 'son', 'nasal', 'cont', 'voice', 'strid', 'lat', 'sg', 'cg', 'dr', 'LAB', 'COR', 'GLOT', 'DOR', 'round', 'ant', 'dist', 'high', 'low', 'back', 'diph', 'wide', 'atr', 'EOW', 'start')
  }
  
  # pick alphabet
  if (dim(embs)[1] == 52) {
    alphabet = c('<s>', 'p', 'ɹ', 'ɔ1', 'm', 't', 'ə', '\n', 'g', 'ʊ', 'd', 'b', 'ʊ1', 'k', 'æ1', 'n', 's', 'æ', 'ɪ', 'z', 'oː', 'ɪ1', 'l', 'ɑ1ː', 'ɛ', 'v', 'ɛ1', 'f', 't͡ʃ', 'θ', 'w', 'j', 'u1ː', 'ʌ1', 'uː', 'i1ː', 'd͡ʒ', 'ɜ1ː', 'h', 'ə1', 'ɑː', 'o1ː', 'iː', 'ɔ', 'ŋ', 'ɜː', 'ʃ', 'ʌ', 'ʒ', 'ð', 'o', 'ːɪ1')
  } else if (dim(embs)[1] == 77) {
    alphabet = c('<s>', 'n', 'j', 'u1ː', 'z', 'ɪ', 's', '\n', 'b', 't', 'h', 'p', 'o1', 'ʊ', 'ɹ', 'e1', 'd', 'o', 'g', 'æ', 'm', 'ɛ1', 'ʃ', 'ɑ1ː', 'k', 'ɑ', 'ɛ2', 'ə', 'ɪ1', 'l', 'æ2', 'f', 'ɔ1ː', 'v', 'ɑ1', 'ʊ1', 'æ1', 'w', 'ʌ1', 'i1ː', 'ɪ2', 'uː', 'ɑː', 'ɜ1ː', 'ʌ2', 'a1', 'θ', 'ɔː', 'd͡ʒ', 'ŋ', 'o2', 'i2ː', 'a', 'ɛ', 'a2', 'e', 'ð', 'iː', 'ʊ2', 't͡ʃ', 'ɑ2ː', 'ɑ2', 'ɔ1', 'ʒ', 'e2', 'u2ː', 'ʌ', 'ɔ', 'ɜː', 'ɜ2ː', 'ɔ2ː', 'ɔ2', 'ɒ1ː', 'ɒː', 'ɒ2ː', 'æ1ː', 'x')
  } else {
    alphabet = c('<s>', 'EH1', 'N', 'ER0', '\n', 'S', 'OW1', 'B', 'R', 'K', 'W', 'AY1', 'AH0', 'T', 'P', 'OY1', 'IH0', 'D', 'UW1', 'M', 'Z', 'AE1', 'L', 'NG', 'AH1', 'HH', 'ER2', 'UW0', 'EY1', 'IH1', 'Y', 'SH', 'UH1', 'EY2', 'V', 'CH', 'AO1', 'G', 'JH', 'EH2', 'ER1', 'AA1', 'AA0', 'IY1', 'OW2', 'AW2', 'F', 'AY0', 'AH2', 'IY0', 'IY2', 'TH', 'AW1', 'OY2', 'DH', 'AE2', 'AO2', 'AY2', 'AE0', 'IH2', 'EH0', 'ZH', 'AA2', 'AO0', 'OW0', 'UW2', 'UH2', 'EY0', 'AW0', 'UH0')
  }

  rownames(embs) = alphabet
  colnames(embs) = lapply(embedding_files, function(x) {
    result <- tail(strsplit(x, "/")[[1]], n=2)[1]
    return(result)
  })

  return(embs)
}

# function that plots a dendrogram from a single embedding matrix
utils$plot_dendro <- function (embs, model_id, phone_cat_file,
                               dist_metric = "euclidean",
                               title = "") {
  model_ag = agnes(t(as.data.frame(embs[,model_id])), metric = dist_metric)
  dendro <- as.dendrogram(model_ag)
  ddata <- dendro_data(dendro, type = "rectangle")
  
  # switch around labels to make the resultant plot more human-intelligible
  labs <- label(ddata)
  labs$label = gsub("X.s.", "<s>", labs$label)
  labs$label = gsub("X.", "</s>", labs$label)
  labs <- labs[order(labs$label),]
  
  # import data about phone categories
  phone_cats <- read.csv(phone_cat_file)
  phone_cats$phone <- as.character(phone_cats$phone)
  row.names(phone_cats) <- phone_cats$phone
  
  if (nrow(labs) == nrow(phone_cats)) {
    
    for (row_idx in 1:nrow(labs)) {
      phone = labs[row_idx, 'label']
      labs[row_idx, "group"] = phone_cats[phone, "category"]
    }

    result_plot <- ggplot(segment(ddata)) + 
      geom_segment(aes(x = x, y = y, xend = xend, yend = yend)) + 
      coord_flip() + 
      theme_dendro() +
      # theme_set(theme_dendro(base_size = 18)) +
      scale_x_discrete(labels=labs$label) +
      scale_color_brewer(palette = "Dark2") +
      labs(color = "category",
           title = title) +
      theme(legend.position="bottom", text = element_text(size=16, family="Open Sans")) +
      NULL
    
    if (dist_metric == "manhattan") {
      result_plot <- result_plot +
        scale_y_reverse(expand = c(0, 5)) +
        geom_text(aes(x = x, y = y,
                      label = label, angle = 0,
                      color=labs$group), data= labs,
                  fontface = "bold",size=3.5,
                  nudge_y = 1,
                  hjust = 0) +
        NULL
    } else {
      result_plot <- result_plot +
        scale_y_reverse(expand = c(0, 2)) +
        geom_text(aes(x = x, y = y,
                      label = label, angle = 0,
                      color=labs$group), data= labs,
                  fontface = "bold",size=5,
                  nudge_y = 0.25,
                  hjust = 0) +
        NULL
    }
    
  } else {
    # don't bother with groups
    
    result_plot <- ggplot(segment(ddata)) + 
      geom_segment(aes(x = x, y = y, xend = xend, yend = yend)) + 
      # coord_flip() + 
      # scale_y_reverse(expand = c(0.4, 0)) +
      theme_dendro() +
      # theme_set(theme_dendro(base_size = 18)) +
      geom_text(aes(x = x, y = y,
                    label = label, angle = 0),
                data= labs,
                fontface = "bold",size=3.5,
                nudge_y = 1) +
      scale_x_discrete(labels=labs$label) +
      scale_color_brewer(palette = "Dark2") +
      theme(legend.position="bottom") +
      NULL
  }

  return(result_plot)
}

while("utils" %in% search())
  detach("utils")
attach(utils)