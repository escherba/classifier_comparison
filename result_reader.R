## Alexander Rao

indexs <- 18:29
scores <- matrix(nrow=15,ncol=length(indexs))
for(i in indexs){
    dat <- read.csv(paste("dec",i,"/results.csv",sep=""),colClasses = c("character",rep("numeric",times=3)))
    scores[,i-17] <- dat$Score
}
    
rownames(scores) <- dat$Classifier

pdf("plots.pdf")

par(mfrow=c(4,4))
par(bg="black")

for(i in 1:15){
    plot(x=indexs,y=scores[i,],main=rownames(scores)[i],ylab="score",xlab="date",type="b",        pch=16,
         col="blue",
         col.main="green",
         col.axis="green",
         col.sub="green",
         col.lab="green",
         fg="green")
}

par(bg="white")
par(mfrow=c(1,1))
dev.off()
system("open plots.pdf")

