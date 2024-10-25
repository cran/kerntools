## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup, include = FALSE---------------------------------------------------
library(kerntools)

## ----eval=FALSE---------------------------------------------------------------
#  Linear(iris[,1:4])

## ----eval=FALSE---------------------------------------------------------------
#  Linear(iris[,1:4], coeff = c(1/2, 1/4, 1/8, 1/8))

## ----eval=FALSE---------------------------------------------------------------
#  Linear(iris[,1:4], cos.norm = TRUE)

## ----eval=FALSE---------------------------------------------------------------
#  RBF(iris[,1:4], g= 0.1)

## ----eval=FALSE---------------------------------------------------------------
#  estimate_gamma(iris[,1:4])

## ----eval=FALSE---------------------------------------------------------------
#  Laplace(iris[,1:4], g= 0.1)

## ----eval=FALSE---------------------------------------------------------------
#  ### Let's suppose our samples come from three different matrix tables:
#  setosa <- iris[iris$Species == "setosa", 1:4]
#  versicolor <- iris[iris$Species == "versicolor", 1:4]
#  virginica <- iris[iris$Species == "virginica", 1:4]
#  matrices <- list(setosa,versicolor,virginica)
#  Frobenius(matrices)

## ----eval=FALSE---------------------------------------------------------------
#  Frobenius(matrices, cos.norm = TRUE)

## ----eval=FALSE---------------------------------------------------------------
#  Frobenius(matrices, feat_space = TRUE)

## ----eval=FALSE---------------------------------------------------------------
#  ### Our example dataset contains the bacterial abundance of *D* species in *N* soil samples.
#  data <- soil$abund
#  BrayCurtis(data)

## ----eval=FALSE---------------------------------------------------------------
#  ### Our example dataset contains the bacterial abundance of *D* species in *N* soil samples.
#  data <- soil$abund
#  Ruzicka(data)

## -----------------------------------------------------------------------------
cat_feat <- data.frame(var=factor(sample(LETTERS[1:3],10,replace = TRUE)))
rownames(cat_feat) <- 1:10
cat_feat

## -----------------------------------------------------------------------------
dummy_data(cat_feat)

## ----eval=FALSE---------------------------------------------------------------
#  Dirac(showdata)

## ----eval=FALSE---------------------------------------------------------------
#  Dirac(showdata,feat_space = TRUE)

## ----eval=FALSE---------------------------------------------------------------
#  coeffs <- c(1/8,1/4,1/4,1/4,1/8)
#  Dirac(showdata, comp = "weighted",  coeff=coeffs)

## ----eval=FALSE---------------------------------------------------------------
#  coeffs <- rep(1/ncol(showdata),length(showdata))
#  Dirac(showdata, comp = "weighted",  coeff=coeffs)

## -----------------------------------------------------------------------------
universe <-  c("blue","green","lightblue","orange","purple","red","white","yellow")
person1 <- c(2, 6, 8)
person2 <- c(1, 8)
person3 <- c(2, 3,  6)
list(person1=universe[person1],person2=universe[person2],person3=universe[person3])

## -----------------------------------------------------------------------------
colors <- matrix(0,nrow=3,ncol=length(universe))
rownames(colors) <- c("person1","person2","person3")
colnames(colors) <- universe
colors[1,person1] <- 1
colors[2,person2] <- 1
colors[3,person3] <- 1
colors

## -----------------------------------------------------------------------------
cosnormX(colors)

## -----------------------------------------------------------------------------
universe <- c("b","g","l","o","p","r","w","y")
colors <- matrix("",ncol=1,nrow=3)
rownames(colors) <-  c("person1","person2","person3")
colors[1,] <-paste(universe[person1],collapse="")
colors[2,] <-paste(universe[person2],collapse="")
colors[3,] <-paste(universe[person3],collapse="")
colors

## ----eval=FALSE---------------------------------------------------------------
#  Intersect(colors,elements = universe)

## ----eval=FALSE---------------------------------------------------------------
#  Intersect(colors, elements = universe, feat_space = TRUE)

## -----------------------------------------------------------------------------
universe2 <-  LETTERS[1:10]
person1 <- c(1,2,9,10)
person2 <- c(1,4,6,7)
person3 <- c(2,3,5,9,10)
person1=universe2[person1]
person2=universe2[person2]
person3=universe2[person3]
list(person1=person3,person2=person3,person3=person3)
friends <- colors
friends[1,] <- paste(person1,collapse="")
friends[2,] <- paste(person2,collapse="")
friends[3,] <- paste(person3,collapse="")
friends

## -----------------------------------------------------------------------------
set_data <- data.frame(colors=colors,friends=friends)
set_data

## ----eval=FALSE---------------------------------------------------------------
#  Intersect(set_data,elements = c(universe,universe2),comp="sum")

## ----eval=FALSE---------------------------------------------------------------
#  ### We will make that "friends" has 3 times more importance than "colors":
#  coeffs <- c(1/4,3/4)
#  Intersect(set_data, elements = c(universe,universe2),comp = "weighted",  coeff=coeffs)

## ----eval=FALSE---------------------------------------------------------------
#  Jaccard(set_data,elements = c(universe,universe2),comp="sum")

## ----eval=FALSE---------------------------------------------------------------
#  cosNorm(Jaccard(set_data,elements = c(universe,universe2),comp="sum"))

## ----eval=FALSE---------------------------------------------------------------
#  D <- ncol(set_data)
#  Jaccard(set_data,elements = c(universe,universe2),comp="sum")/D

## ----eval=FALSE---------------------------------------------------------------
#  Jaccard(set_data,elements = c(universe,universe2),comp="mean")

## -----------------------------------------------------------------------------
color_list <-  c("black","blue","green","grey","lightblue","orange","purple",
"red","white","yellow")
survey1 <- 1:10
survey2 <- 10:1
survey3 <- sample(10)
color <- cbind(survey1,survey2,survey3) # Samples in columns
rownames(color) <- color_list
color

## ----eval=FALSE---------------------------------------------------------------
#  Kendall(color)

## ----eval=FALSE---------------------------------------------------------------
#  color <- t(color)
#  Kendall(color,samples.in.rows=TRUE)

## -----------------------------------------------------------------------------
food <- matrix(c(10, 1,18, 25,30, 7, 5,20, 5, 12, 7,20, 20, 3,22),ncol=3,nrow=5,byrow = TRUE)
colnames(food) <- colnames(color)
rownames(food) <- c("spinach", "chicken", "beef" , "salad","lentils")
food

## ----eval=FALSE---------------------------------------------------------------
#  Kendall(food)

## ----eval=FALSE---------------------------------------------------------------
#  ordinal_data <- list(color=color,food=food) #All samples in columns
#  Kendall(ordinal_data)

## -----------------------------------------------------------------------------
LETTERS

## -----------------------------------------------------------------------------
length(LETTERS)

## -----------------------------------------------------------------------------
alphabet <- c(letters,"_")
strings <- c("hello_world","hello_word","hola_mon","kaixo_mundua",
"saluton_mondo","ola_mundo", "bonjour_le_monde")
names(strings) <- c("english1","english_typo","catalan","basque",
"esperanto","galician","french")
strings
alphabet

## ----eval=FALSE---------------------------------------------------------------
#  Spectrum(strings,l=2,alphabet=alphabet)

## ----eval=FALSE---------------------------------------------------------------
#  Spectrum(strings,l=2,alphabet=alphabet, feat_space = TRUE)

## ----eval=FALSE---------------------------------------------------------------
#  Spectrum(strings,l=2,alphabet=alphabet,cos.norm = TRUE)

