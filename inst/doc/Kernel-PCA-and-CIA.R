## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup, include = FALSE---------------------------------------------------
library(kerntools)

## -----------------------------------------------------------------------------
iris_feat <- iris[,c( "Sepal.Length","Sepal.Width","Petal.Length","Petal.Width")]

# The variables are centered but not standardized:
iris_prcomp <- prcomp(iris_feat,center = TRUE,scale. = FALSE)
iris_princomp <- princomp(iris_feat, cor = FALSE)
iris_kerntools <- kPCA(Linear(iris_feat),center=TRUE)

head(iris_prcomp$x)
head(iris_princomp$scores)
head(iris_kerntools[,1:4])

## -----------------------------------------------------------------------------
iris_prcomp$rotation
iris_princomp$loadings

## -----------------------------------------------------------------------------
iris_pcs <- kPCA_imp(iris_feat,center = TRUE, projected = iris_kerntools)
t(iris_pcs$loadings)

## -----------------------------------------------------------------------------
colnames(showdata)
categdata <- Dirac(showdata,feat_space = TRUE)
K <- categdata$K
FS <- categdata$feat_space

## -----------------------------------------------------------------------------
dirac_kpca <- kPCA(K, center = TRUE)
dirac_pcs <- kPCA_imp(FS,center = TRUE, projected = dirac_kpca)
head(t(dirac_pcs$loadings[1:3,]) )

## -----------------------------------------------------------------------------
## Example using random datasets
data1 <- matrix(rnorm(50),ncol=5,nrow=10)
data2 <- matrix(rnorm(50),ncol=5,nrow=10)
data3 <- matrix(rnorm(50),ncol=5,nrow=10)

K1 <- Linear(data1)
K2 <- Linear(data2)
K3 <- Linear(data3)

K1 <- centerK(K1)
K2 <- centerK(K2)
K3 <- centerK(K3)


simK(list(data1=K1,data2=K2,data3=K3))

## -----------------------------------------------------------------------------
## Example using random datasets
data1 <- matrix(rnorm(50),ncol=5,nrow=10)
data2 <- c("flowing","flower","cauliflower","thing","water","think","float","ink","wait","deaf")
data3 <- matrix(sample(LETTERS[1:5],50,replace=TRUE),ncol=5,nrow=10)

K1 <- Linear(data1)
K2 <- Spectrum(data2,alphabet = letters,l = 2)
K3 <- Dirac(data3,comp = "sum")

simK(list(Real=K1,String=K2,Categorical=K3))

