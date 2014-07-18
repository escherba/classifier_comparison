## Alexander Rao

train.week <- "weekly/week2013-12-28/"

weeks <- system(paste("ls",train.week) ,intern=TRUE)
indexs = 1:length(weeks)

scores <- matrix(nrow=15,ncol=length(weeks))
for(i in indexs){
    week <- weeks[i]
    dat <- read.csv(paste(train.week, week,"/results.csv",sep=""),colClasses = c("character",rep("numeric",times=3)))
    scores[,i] <- dat$Score
}
    
rownames(scores) <- dat$Classifier

file.name = "2013-12-28_original_c.pdf"
pdf(file.name)

par(mfrow=c(4,4))
par(bg="black")


for(i in 1:15){
    plot(x=indexs,y=scores[i,],main=rownames(scores)[i],ylab="score",xlab="date",type="b",        pch=16,
         col="blue",
         col.main="green",
         col.axis="green",
         col.sub="green",
         col.lab="green",
         fg="green"
         ## col="dark green",
         )

}

par(bg="white")
par(mfrow=c(1,1))
dev.off()
system(paste("open",file.name))
