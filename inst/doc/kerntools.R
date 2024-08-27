## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.width = 7, 
  fig.height = 6.5
)

## ----setup--------------------------------------------------------------------
library(kerntools)

## -----------------------------------------------------------------------------
iris_std <- scale(iris[,c( "Sepal.Length","Sepal.Width","Petal.Length", "Petal.Width")])
KL <- Linear(iris_std)
dim(KL)
class(KL)

## -----------------------------------------------------------------------------
histK(KL, vn = TRUE)

## -----------------------------------------------------------------------------
heatK(KL,cos.norm = FALSE)

## -----------------------------------------------------------------------------
iris_species <- factor(iris$Species)
kpca <- kPCA(KL,plot = 1:2,y = iris_species)
kpca$plot

## -----------------------------------------------------------------------------
library(kernlab)
set.seed(777)
## Training data
test_idx <- sample(1:150)[1:30] # 20% of samples
train_idx <- (1:150)[-test_idx]
KL_train <- KL[train_idx,train_idx]
## Model (training data)
linear_model <- kernlab::ksvm(x=KL_train, y=iris_species[train_idx], kernel="matrix")
linear_model

## -----------------------------------------------------------------------------
## Third model: Versicolor vs virginica
sv_index <- kernlab::alphaindex(linear_model)[[3]] # Vector with the SV indices
sv_coef <- kernlab::coef(linear_model)[[3]]  # Vector with the SV coefficients

feat_imp3 <- svm_imp(X=iris_std[train_idx,],svindx=sv_index,coeff=sv_coef)

## -----------------------------------------------------------------------------
plotImp(feat_imp3, leftmargin = 7, main="3rd model: versicolor vs virginica", color="steelblue1")

## -----------------------------------------------------------------------------
loadings <- kPCA_imp(iris_std)
pcs <- loadings$loadings[1:2,]
pcs

## -----------------------------------------------------------------------------
plotImp(pcs[1,], y=pcs[2,], ylegend="PC2",absolute=FALSE, main="PC1", leftmargin = 7,  color="rosybrown1")

## -----------------------------------------------------------------------------
kPCA_arrows(plot=kpca$plot,contributions=t(pcs),colour="grey15")

## -----------------------------------------------------------------------------
KL_test <- as.kernelMatrix(KL[test_idx,train_idx])
## Prediction (test data)
pred_class <- predict(linear_model,KL_test)
actual_class <- iris_species[test_idx]
pred_class
actual_class

## -----------------------------------------------------------------------------
ct <- table(actual_class,pred_class) # Confusion matrix
ct
Acc(ct) ## Accuracy
Acc_rnd(actual_class) ## Accuracy of the random model

## -----------------------------------------------------------------------------
Prec(ct,multi.class = "none") ## Precision or Positive Predictive Value
Rec(ct,multi.class = "none") ## Recall or True Positive Rate
Spe(ct,multi.class = "none") ## Specificity or True Negative Rate
F1(ct,multi.class = "none") ## F1 (harmonic mean of Precision and Recall)


## -----------------------------------------------------------------------------
Krbf <- RBF(iris_std,g=0.25)
histK(Krbf,col="aquamarine",vn = TRUE)

## -----------------------------------------------------------------------------
estimate_gamma(iris_std)

## -----------------------------------------------------------------------------
Krbf2 <- RBF(iris_std,g=0.1667)
histK(Krbf2,col="darkseagreen1",vn=TRUE)

## -----------------------------------------------------------------------------
Krbf_train <- Krbf2[train_idx,train_idx]
Krbf_test <- Krbf2[test_idx,train_idx]
rbf_kpca <- kPCA(K=Krbf_train, Ktest=Krbf_test, plot = 1:2, y = iris_species[train_idx], title = "RBF PCA")
rbf_kpca$plot

## -----------------------------------------------------------------------------
simK(list(linear=KL,rbf_0.166=Krbf, rbf_0.25=Krbf2))

## -----------------------------------------------------------------------------
rbf_kpca <- kPCA(K=Krbf)
proc <- Procrustes(kpca$projection,rbf_kpca)
proc$pro.cor # Procrustes correlation

## -----------------------------------------------------------------------------
####### Model (training data)
model <- kernlab::ksvm(x=Krbf_train, y=iris_species[train_idx], kernel="matrix", C=10)
model

## -----------------------------------------------------------------------------
Krbf_test <- as.kernelMatrix(Krbf_test)
####### Prediction (test data)
pred_class <- predict(model,Krbf_test)
actual_class <- iris_species[test_idx]
ct <- table(actual_class,pred_class) # Confusion matrix
Acc(ct) ## Accuracy

## -----------------------------------------------------------------------------
 ## Accuracy CI (95%)
Normal_CI(value = 0.5,ntest = 30) ## Accuracy CI (95%)
Boots_CI(target = actual_class, pred = pred_class, nboots = 2000,index = "acc") 

## -----------------------------------------------------------------------------
Prec(ct) ## Precision or Positive Predictive Value
Rec(ct) ## Recall or True Positive Rate
Spe(ct) ## Specificity or True Negative Rate
F1(ct) ## F1 (harmonic mean of Precision and Recall)


## -----------------------------------------------------------------------------
####### RBF versicolor vs virginica model:
sv_index <- kernlab::alphaindex(model)[[3]]
sv_coef <- kernlab::coef(model)[[3]]
svm_imp(X=iris_std[train_idx,],svindx=sv_index,coeff=sv_coef)

## -----------------------------------------------------------------------------
summary(showdata)

## -----------------------------------------------------------------------------
dummy_showdata <- dummy_data(showdata)
dummy_showdata[1:5,1:3]

## -----------------------------------------------------------------------------
KD <- Dirac(showdata, comp="sum")
histK(KD, col ="plum2")

## -----------------------------------------------------------------------------
KD <- Dirac(showdata[,1:4], comp="sum",feat_space=TRUE)
dirac_kpca <- kPCA(KD$K,plot=1:2,y=factor(showdata$Liked.new.show),ellipse=0.66,title="Dirac kernel PCA")
dirac_kpca$plot

## -----------------------------------------------------------------------------
pcs <- kPCA_imp(KD$feat_space)
pc1 <- plotImp(pcs$loadings[1,],leftmargin=15,nfeat=10,absolute = FALSE,  relative = FALSE,col ="bisque")
pc2 <- plotImp(pcs$loadings[2,],leftmargin=17,nfeat=10,absolute = FALSE, relative = FALSE, col="honeydew3")

## -----------------------------------------------------------------------------
features <- union(pc1$first_features,pc2$first_features)
kPCA_arrows(plot=dirac_kpca$plot,contributions=t(pcs$loadings[1:2,features]),colour="grey15")

## -----------------------------------------------------------------------------
KL1 <- Linear(cosnormX(iris[,1:4])) # important: by row
KL2 <- cosNorm(Linear(iris[,1:4])) # a third valid option is: Linear(iris[,1:4], cos.norm=TRUE)
simK(list(KL1=KL1,KL2=KL2))

## -----------------------------------------------------------------------------
center_iris <- scale(iris[,1:4],scale=FALSE,center=TRUE)
histK(RBF(center_iris,g=0.25),col="aquamarine")

## -----------------------------------------------------------------------------
histK(centerK(RBF(iris[,1:4],g=0.25)),col="aquamarine3")

## -----------------------------------------------------------------------------
dim(mtcars)
head(mtcars)

## -----------------------------------------------------------------------------
cat_feat_idx <- 8:9
MTCARS <- list(num=mtcars[,-cat_feat_idx], cat=mtcars[,cat_feat_idx])

## -----------------------------------------------------------------------------
K <- array(dim=c(32,32,2))
K[,,1] <- Linear(MTCARS[[1]]) ## Kernel for numeric data
K[,,2] <- Dirac(MTCARS[[2]]) ## Kernel for categorical data

## -----------------------------------------------------------------------------
Kcons <- MKC(K)

## -----------------------------------------------------------------------------
coeff <- sapply(MTCARS,ncol)
coeff #  K1 will weight 9/11 and K2 2/11.
Kcons_var <- MKC(K,coeff=coeff)

## -----------------------------------------------------------------------------
simK(list(Kcons=Kcons,K1=K[,,1],K2=K[,,2]))
simK(list(Kcons_var=Kcons_var,K1=K[,,1],K2=K[,,2]))

## -----------------------------------------------------------------------------
histK(K[,,1], col="khaki1")
histK(K[,,2], col="hotpink")

## -----------------------------------------------------------------------------
K[,,1] <- Linear(minmax(MTCARS[[1]])) ## Kernel for numeric data
K[,,2] <- Dirac(MTCARS[[2]],comp="sum") ## Kernel for categorical data
Kcons <- MKC(K)
Kcons_var <- MKC(K,coeff=coeff)
simK(list(Kcons=Kcons,K1=K[,,1],K2=K[,,2]))
simK(list(Kcons=Kcons_var,K1=K[,,1],K2=K[,,2]))

