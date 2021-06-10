
# Importing the required libraries
library(quantmod)
library(magrittr)
library(visdat)
library(naniar)
library(stringr)
library(dplyr)
library(quanteda)
# library(tensorflow)
# library(keras)
library(reticulate)
library(purrr)
library(topicmodels)
library(tm)
library(doSNOW)
library(dplyr)
library(LDAvis)
library(lda)
library(pals)
library(pacman)
library(tidytext)
library(igraph)
library(servr)
library(plotly)
library(webshot)
library(htmlwidgets)
library(wordcloud)
library(caret)

# Reading in the data
Data <- read.csv(file = "C:/Users/Um Ar/R Projects/UN MINOR/Datasets/Nai/train.csv",
                 header = T)

# Data2 <- read.csv(file = "C:/Users/Um Ar/R Projects/UN MINOR/Datasets/data.csv")

# colnames(Data)
# colnames(Data2)

# colnames(Data2)[2] <- "title"
# colnames(Data2)[3] <- "text"
# colnames(Data2)[4] <- "label"

# Data2 <- Data2[,-1]
# Data <- Data[,-1]
# Data <- Data[,-2]

# FN_Data <- rbind(Data, Data2)


# Data <- FN_Data


# Exploratory Data Analysis
summary(Data)

head(Data)

colnames(Data)

# Creating a new feature  Text Length
Data$title <- as.character(Data$title)
Data$text <- as.character(Data$text)
Data$title_Text_Length <- nchar(Data$title)
Data$text_Text_Length <- nchar(Data$text)

library(ggplot2)
# Checking the distribution of data
ggplot(Data, aes(label)) +
  geom_bar(fill = "blue")


# Text Length of Main body
ggplot(Data, aes(x = text_Text_Length, fill = label)) + 
  geom_histogram(bins = 80) +
  labs(y = "Samples Count",
       x = "Text Length",
       title = "Main Body Text Length")
  


# Text Length of Headlines
ggplot(data = Data, mapping = aes(x = title_Text_Length)) + 
  geom_histogram(bins = 80) +
  theme_bw() + 
  labs(y = "Samples Count",
       x = "Text Length",
       title = "Head Line Text Length")



# Data$que <- sapply(Data$Headline, function(x)
#  length(unlist(strsplit(as.character(x), "\\?+"))))

# Count of exclamations in fake and real news
# Data %>% group_by(Label) %>% summarise(exclamations=sum(que))

# plotting histogram of title length
ggplot(Data ,aes(x = title_Text_Length, fill = label)) +
  geom_density(alpha=0.5) +
  guides(fill=guide_legend(title="News type")) + 
  xlab("Title length") + ylab("Density") + theme() + 
  ggtitle("Density distribuiton of title length") 


ggplot(Data ,aes(x = text_Text_Length, fill = label)) +
  geom_density(alpha=0.5) +
  guides(fill=guide_legend(title="News type")) + 
  xlab("Body length") + ylab("Density") + theme() + 
  ggtitle("Density distribuiton of Body length") 



# write.csv(Data, "FINAL_DATA.csv")

# Checking the missing values
# Heatmap of missing data
vis_miss(Data, cluster = T)

# Preprocessing
# Preprocessing the text files
corpus <- Corpus(VectorSource(Data$text))
english_words <- readLines("stopwords.txt",
                           encoding = "UTF-8")

processedCorpus <- tm_map(corpus, content_transformer(tolower))
processedCorpus <- tm_map(processedCorpus, removeWords, english_words)
processedCorpus <- tm_map(processedCorpus, removePunctuation, preserve_intra_word_dashes = TRUE)
processedCorpus <- tm_map(processedCorpus, removeNumbers)
processedCorpus <- tm_map(processedCorpus, stemDocument, language = "en")
processedCorpus <- tm_map(processedCorpus, stripWhitespace)


# Changing the corpus into a document term matrix
minimumFrequency <- 7
DTM <- DocumentTermMatrix(processedCorpus,
                          control = list(bounds = list(global = c(minimumFrequency, Inf))))

dim(DTM)


# Checking for missing values and removing them
raw.sum <- apply(DTM, 1, FUN = sum)
dfm_trimmed <- DTM[raw.sum!=0,]


# Hyperparameters for topic modelling
burnin <- 100
# Iterations
iter <- 100
# Taking every 100 iteration for further use
thin <- 10
# using 10 starting points
nstart <- 1
#Seeds for 10 starts
seed <- list(1103)
best <- T
# number of topicsrm

gc()
cl <- makeCluster(6, type = "SOCK")
registerDoSNOW(cl)

# TOpic modelling
topicModel <- LDA(dfm_trimmed, k = 50, method = "Gibbs", 
                  control = list(seed = seed, nstart = nstart,
                                 best = best, burnin = burnin, iter = iter,
                                 thin = thin, verbose = 25))

# Groping the topics
Topics <- tidy(topicModel, matrix = "beta")
top_terms <- Topics %>% group_by(topic) %>% top_n(20, beta) %>%
  ungroup() %>% arrange(topic, -beta)

# save.image("C:/Users/Um Ar/R Projects/UN MINOR/.RData")

# Taking out the top 5 terms
# TopicsT <- posterior(topicModel, dfm_trimmed)[["topics"]]
tmResult <- posterior(topicModel)
top5termsPerTopic <- terms(topicModel, 5)
topicNames <- apply(top5termsPerTopic, 2, paste, collapse=" ")

topicToViz <- 20 # change for topics of interest
# topicToViz <- grep("feel", topicNames)[1] # Or select a topic by a term contained in its name
# selecting to 80 most probable terms from the topic by sorting the term-topic-probability vector in decreasing order
top40terms <- sort(tmResult$terms[topicToViz,], decreasing=TRUE)[1:80]
words <- names(top40terms)
# extract the probabilites of each of the 40 terms
probabilities <- sort(tmResult$terms[topicToViz,], decreasing=TRUE)[1:80]
# visualize the terms as wordcloud
is.na(words)
mycolors <- brewer.pal(8, "Dark2")
wordcloud(words, probabilities, random.order = FALSE, color = mycolors)

# re-rank top topic terms for topic names
beta <- tmResult$terms
topicNames <- apply(lda::top.topic.words(beta, 10, by.score = T),
                    2, paste, collapse = " ")



# Plotting the top terms
top_terms %>% mutate(term = reorder(term, beta)) %>%
  mutate(topic = paste("Topic #", topic)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = F) +
  facet_wrap(~ topic, scales = "free") + 
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 18)) +
  labs(title = "Topic Model of suicide",
       caption = "Top Terms by Topic") +
  ylab("") +
  xlab("") +
  coord_flip()


# Heatmap of the topics terms
# heatmap(TopicsT)

# Network of the Word distribution over TOpics
post <- posterior(topicModel)

cor_mat <- cor(t(post[["terms"]]))
cor_mat[ cor_mat < .05 ] <- 0
diag(cor_mat) <- 0

graph <- graph.adjacency(cor_mat, weighted=TRUE, mode="lower")
graph <- delete.edges(graph, E(graph)[ weight < 0.05])

E(graph)$edge.width <- E(graph)$weight*20
V(graph)$label <- paste("Topic", V(graph))
V(graph)$size <- colSums(post[["topics"]]) * 0.05

par(mar=c(0, 0, 3, 0))
set.seed(110)
plot.igraph(graph, edge.width = E(graph)$edge.width, 
            edge.color = "orange", vertex.color = "orange", 
            vertex.frame.color = NA, vertex.label.color = "grey30")
title("Strength Between Topics Based On Word Probabilities", cex.main=.8)



# Network of topics over docuents
minval = .1
topic_mat <- posterior(topicModel)[["topics"]]
graph <- graph_from_incidence_matrix(topic_mat, weighted = T)
graph <- delete.edges(graph, E(graph)[weight < minval])

E(graph)$edge.width <- E(graph)$weight*0.2
E(graph)$color <- "blue"
V(graph)$color <- ifelse(grepl("^\\d+$", V(graph)$name), "grey75", "orange")
V(graph)$frame.color <- NA
V(graph)$label <- ifelse(grepl("^\\d+$", V(graph)$name),
                         paste("topic", V(graph)$name),
                         gsub("_", "\n", V(graph)$name))
V(graph)$size <- c(rep(10, nrow(topic_mat)), colSums(topic_mat) * 0.01)
V(graph)$label.color <- ifelse(grepl("^\\d+$", V(graph)$name), "red", "grey30")

par(mar=c(0, 0, 3, 0))
set.seed(365)
plot.igraph(graph, edge.width = E(graph)$edge.width, 
            vertex.color = adjustcolor(V(graph)$color, alpha.f = .4))
title("Topic & Document Relationships", cex.main=.8)
# Ldavis interactive plot
# topicModel %>%
#   topicmodels2LDAvis() %>%
#  LDAvis::serVis()
# save.image("C:/Users/Um Ar/R Projects/UN MINOR/.RData")
