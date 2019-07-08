
# Get more colors!
palette(c("black", "green", "blue", "cyan", "magenta", "yellow", "grey", "purple", "brown", "violet", "pink", "orange", "darkgreen", "khaki", "darkorange", "red"))
# Black will be used for vectors classified as noise (cluster 0 for dbscan)

#### Start ####
# install.packages("BBmisc")
# install.packages("ppclust")
# install.packages("fclust")
# install.packages("dbscan")
# install.packages("EMCluster")
# install.packages("advclust")
library(EMCluster)
library(BBmisc)
library(ppclust)
library(fclust)
library(dbscan)
library(advclust)

#### Datasets ####

file <- "/Users/rafal/Documents/Studia/4GL/4GL_Zad2/Datasets/2d_k15_little_overlap.txt"
data <- read.table(file = file, sep = "" , header = F , nrows = 5000,
                   na.strings = "", stringsAsFactors = F)
data = normalize(data, method = "standardize", range = c(0, 1))
str(data)
summary(data)
plot(data, pch = 16, cex = .3)

# file <- "/Users/rafal/Documents/Studia/4GL/4GL_Zad2/Datasets/2d_k15_much_overlap.txt"
# data <- read.table(file = file, sep = "" , header = F , nrows = 5000,
#                    na.strings = "", stringsAsFactors = F)
# data = normalize(data, method = "standardize", range = c(0, 1))
# plot(data, pch = 16, cex = .3)
# 
# file <- "/Users/rafal/Documents/Studia/4GL/4GL_Zad2/Datasets/5d_k2_thyroid.txt"
# data <- read.table(file = file, sep = "" , header = F , nrows = 5000,
#                    na.strings = "", stringsAsFactors = F)
# data = normalize(data, method = "standardize", range = c(0, 1))
# plot(data, pch = 16, cex = .3)

#plot(data, col = res.pcm$cluster, pch = 16, cex = .5)


#### HCM ####


res.hcm <- hcm(data, centers = 15, iter.max = 100, dmetric = "euclidean")
hullplot(data, res.hcm, pch = 16, cex = .5, main = "HCM with dmetric = euclidean")

#### FCM ####

# res.fcm <- fcm(data, centers = 15, iter.max = 100, m = 2, dmetric = "sqeuclidean")
# hullplot(data, res.fcm, pch = 16, cex = .5, main = "")

res.fcm.m2dmetricsqeuclidean <- fcm(data, centers = 15, iter.max = 100, m = 2, dmetric = "sqeuclidean")
hullplot(data, res.fcm.m2dmetricsqeuclidean, pch = 16, cex = .5, main = "FCM with m = 2, dmetric = sqeuclidean")
res.fcm.m2dmetriceuclidean <- fcm(data, centers = 15, iter.max = 100, m = 2, dmetric = "euclidean")
hullplot(data, res.fcm.m2dmetriceuclidean, pch = 16, cex = .5, main = "FCM with m = 2, dmetric = euclidean")
res.fcm.m2dmetriccorrelation <- fcm(data, centers = 15, iter.max = 100, m = 2, dmetric = "correlation")
hullplot(data, res.fcm.m2dmetriccorrelation, pch = 16, cex = .5, main = "FCM with m = 2, dmetric = correlation")
res.fcm.m4dmetricsqeuclidean <- fcm(data, centers = 15, iter.max = 100, m = 4, dmetric = "sqeuclidean")
hullplot(data, res.fcm.m4dmetricsqeuclidean, pch = 16, cex = .5, main = "FCM with m = 4, dmetric = sqeuclidean")

res.fcm.m2dmetricdivergence <- fcm(data, centers = 15, iter.max = 100, m = 2, dmetric = "divergence")
hullplot(data, res.fcm.m2dmetricdivergence, pch = 16, cex = .5, main = "FCM with m = 2, dmetric = divergence")
res.fcm.m2dmetricmanhattan <- fcm(data, centers = 15, iter.max = 100, m = 2, dmetric = "manhattan")
hullplot(data, res.fcm.m2dmetricmanhattan, pch = 16, cex = .5, main = "FCM with m = 2, dmetric = manhattan")

#### PCM ####

# res.pcm <- pcm(data, centers = 15, iter.max = 100, eta = 2, K = 1, oftype = 1, dmetric = "sqeuclidean")
# hullplot(data, res.pcm, pch = 16, cex = .5, main = "")

res.pcm.eta2K1oftype1 <- pcm(data, centers = 15, iter.max = 100, eta = 2, K = 1, oftype = 1, dmetric = "sqeuclidean")
hullplot(data, res.pcm.eta2K1oftype1, pch = 16, cex = .5, main = "PCM with eta = 2, K = 1, oftype = 1")
res.pcm.eta4K1oftype1 <- pcm(data, centers = 15, iter.max = 100, eta = 4, K = 1, oftype = 1, dmetric = "sqeuclidean")
hullplot(data, res.pcm.eta4K1oftype1, pch = 16, cex = .5, main = "PCM with eta = 4, K = 1, oftype = 1")
res.pcm.eta2K2oftype1 <- pcm(data, centers = 15, iter.max = 100, eta = 2, K = 2, oftype = 1, dmetric = "sqeuclidean")
hullplot(data, res.pcm.eta2K2oftype1, pch = 16, cex = .5, main = "PCM with eta = 2, K = 2, oftype = 1")
res.pcm.eta2K1oftype2 <- pcm(data, centers = 15, iter.max = 100, eta = 2, K = 1, oftype = 2, dmetric = "sqeuclidean")
hullplot(data, res.pcm.eta2K1oftype2, pch = 16, cex = .5, main = "PCM with eta = 2, K = 1, oftype = 2")

#### FMLE ####

res.fmle.m2ggversionsimple <- gg(data, centers = 15, iter.max = 5, m = 2, ggversion = "simple")
hullplot(data, res.fmle.m2ggversionsimple, pch = 16, cex = .5, main = "GG-FMLE with m = 2, ggversion = simple")
res.fmle.m4ggversionsimple <- gg(data, centers = 15, iter.max = 5, m = 4, ggversion = "simple")
hullplot(data, res.fmle.m2ggversionsimple, pch = 16, cex = .5, main = "GG-FMLE with m = 4, ggversion = simple")
res.fmle.m2ggversionoriginal <- gg(data, centers = 15, iter.max = 5, m = 2, ggversion = "original")
hullplot(data, res.fmle.m2ggversionsimple, pch = 16, cex = .5, main = "GG-FMLE with m = 2, ggversion = original")

#### EM ####

set.seed(1234)
emobj <- simple.init(data, nclass = 15)

emobj <- emcluster(data, emobj, assign.class = TRUE)

plotem(emobj, data, color.pch=16, main="EM cluster")

#### GK ####

# res.gk <- gk(data, centers = 15, iter.max = 100, m = 2)
# hullplot(data, res.gk, pch = 16, cex = .5, main = "")
# plot(data, col = res.gk$cluster, pch = 16, cex = .5)

# ppclust implementation seems broken!

res.gk.ppclust.m2 <- gk(data, centers = 15, iter.max = 100, m = 2)
hullplot(data, res.gk.ppclust.m2, pch = 16, cex = .5, main = "GK with m = 2")
plot(data, col = res.gk.ppclust.m2$cluster, pch = 16, cex = .5)

# Use fclust instead of ppclust

res.gk.fclust.m2 = FKM.gk(data, k = 15, maxit = 100, m = 2)
# hullplot doesn't work because FKM.gk produces an object of class with a member called "clus" instead of "cluster"...
# hullplot(data, res.gk.fclust.m2, pch = 16, cex = .5, main = "")
plot(data, col = res.gk.fclust.m2$clus, pch = 16, cex = .5)

res.gk.fclust.m4 = FKM.gk(data, k = 15, maxit = 100, m = 4)
plot(data, col = res.gk.fclust.m4$clus, pch = 16, cex = .5)

#### DBSCAN ####

# Use k = minPts used later for dbscan. To find eps, find the knee on the following graph
kNNdistplot(data, k = 15)
# And so seems that eps = .08
abline(h = .08, col = "red", lty = 2)

res.dbscan.eps008minPts4 <- dbscan(data, borderPoints = TRUE, eps = .08, minPts = 4)
hullplot(data, res.dbscan.eps008minPts4, pch = 16, cex = .5, main = "DBSCAN with eps = .08, minPts = 4")
# pairs(data, col = res.dbscan.eps008minPts4$cluster + 1)

kNNdistplot(data, k = 15)
# And so seems that eps = .075
abline(h = .075, col = "red", lty = 2)

res.dbscan.eps008minPts12 <- dbscan(data, borderPoints = TRUE, eps = .08, minPts = 12)
hullplot(data, res.dbscan.eps008minPts12, pch = 16, cex = .5, main = "DBSCAN with eps = .08, minPts = 12")
# pairs() useful to scatterplot when dimentions > 2
# pairs(data, col = res.dbscan.eps0075minPts10$eps008minPts10 + 1)

# See what happens with eps = 0.15

res.dbscan.eps015minPts10 <- dbscan(data, borderPoints = TRUE, eps = .15, minPts = 10)
hullplot(data, res.dbscan.eps015minPts10, pch = 16, cex = .5, main = "DBSCAN with eps = .15, minPts = 10")

# Predicting new data's classification

newdata <- c(1.5, 2)
predict(res.dbscan.eps008minPts12, newdata, data = data)

#### BIRCH ####

######## CRASHES :-( ########

# To install a package from a local tar.
# A (deprecated) BIRCH implementation can be found here: https://cran.r-project.org/web/packages/birch/
# install.packages(path_to_tar, repos = NULL, type="source")

# Install Rtools so the downloaded package can be built: https://cran.r-project.org/bin/windows/Rtools/Rtools35.exe
# install.packages("ellipse")
# install.packages("C:\\Users\\PC\\Desktop\\Studia\\R\\4GL_Zad2\\birch_1.2-3.tar.gz", repos = NULL, type="source")

# datamatrix = as.matrix(data)
# res.birch.radius01compact01 <- birch(datamatrix, keeptree = TRUE, radius = .1, compact = .1)
# birchobj <- birch.getTree(res.birch.radius01compact01)


######## HCLUST ########

hclust.dist.methodeuclidean <- dist(data, method = "euclidean")
res.hclust.methodwardD <- hclust(hclust.dist.methodeuclidean, method = "ward.D")
plot(res.hclust.methodwardD)
rect.hclust(res.hclust.methodwardD, k = 15)
res.hclust.methodwardD.groups <- cutree(res.hclust.methodwardD, k = 15)

hullplot(data, res.hclust.methodwardD.groups, pch = 16, cex = .3, main = "Hclust Dendrogram with k = 15, method = ward.D")


hclust.dist.methodeuclidean <- dist(data, method = "euclidean")
res.hclust.methodmedian <- hclust(hclust.dist.methodeuclidean, method = "median")
plot(res.hclust.methodmedian)
rect.hclust(res.hclust.methodmedian, k = 15)
res.hclust.methodmedian.groups <- cutree(res.hclust.methodmedian, k = 15)

hullplot(data, res.hclust.methodmedian.groups, pch = 16, cex = .3, main = "Hclust Dendrogram with k = 15, method = median")

