### Alexander Rao
### create a word cloud

## first arg is imput csv where first column is words 
## second arg is pdf output
## third arg is the number of words to appear in the word cloud
args <- commandArgs(trailingOnly = TRUE)

library(wordcloud)


dat <- read.csv(args[1],header=FALSE,colClasses=c("character","numeric"))
words <- dat[,1]
freqs <- dat[,2]
n <- length(freqs)
num.words <- min(n, as.numeric(args[3]))


names(freqs) <- 1:n
sorted.freqs <- sort(freqs,decreasing = TRUE)
sorted.index <- as.numeric(names(sorted.freqs))
sorted.words <- words[sorted.index]


top.words <- sorted.words[1:num.words]
top.freqs <- sorted.freqs[1:num.words] 

pdf(args[2])

wordcloud(top.words, top.freqs, colors=c(brewer.pal(8,"Dark2"),brewer.pal(8,"Set2")), min.freq=0)
dev.off()
