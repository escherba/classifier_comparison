library(ggplot2)
tt.df = read.csv("tags_vs_topics.csv", header = TRUE)
bp <- ggplot(tt.df, aes(x=Topic1.Topic2, y=U.coefficient)) + geom_line(aes(y = user, colour= "user")) + geom_line(aes(y = spam, colour = "spam")) + geom_line(aes(y = bulk, colour = "bulk")) + geom_line(aes(y = profanity, colour = "profanity")) + geom_line(aes(y = insult, colour = "insult")) + theme(legend.title=element_blank(), text = element_text(size=8))
bp
ggsave("topic_clusters.png", bp, width=5.5, height=4.5)
