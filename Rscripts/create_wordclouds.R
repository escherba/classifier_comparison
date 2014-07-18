### Alexander Rao
### create a word cloud

## first arg is imput csv where first column is words and second is a frequency mesure
## third arg is the number of words to appear in the word cloud
## giving args
args <- commandArgs(trailingOnly = TRUE)

## loading libraries
library(tm)
library(wordcloud)
library(rjson)
library(plyr)


dat <- read.csv(args[1],header=FALSE,colClasses=c("character","numeric"))
words <- dat[,1]
freqs <- dat[,2]

n <- length(freqs)
names(freqs) <- 1:n
sorted.freqs <- sort(freqs,decreasing = TRUE)
sorted.index <- as.numeric(names(sorted.freqs))
sorted.words <- words[sorted.index]

num.words <- as.numeric(args[3])

top.words <- sorted.words[1:num.words]
top.freqs <- sorted.freqs[1:num.words] 

pdf(args[2])
## wordcloud(words, freqs, min.freq=as.numeric(args[3]), colors=brewer.pal(6, "Dark2"))
wordcloud(top.words, top.freqs, colors=brewer.pal(6, "Dark2"))
dev.off()
system(paste("open",args[2]))

