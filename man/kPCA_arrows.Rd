% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/kernel_pca.R
\name{kPCA_arrows}
\alias{kPCA_arrows}
\title{Plot the original variables' contribution to a PCA plot}
\usage{
kPCA_arrows(plot, contributions, colour = "steelblue", size = 4, ...)
}
\arguments{
\item{plot}{A kernel PCA plot generated by `kPCA()`.}

\item{contributions}{The variables contributions, for instance obtained via `kPCA_imp()`.
It is not mandatory to draw all the original variables; a subset of interest
can be passed on to this argument.}

\item{colour}{Color of arrows and labels. (Defaults: "steelblue").}

\item{size}{Size of the labels. (Defaults: 4).}

\item{...}{Additional parameters passed on to geom_segments() and geom_text().}
}
\value{
The PCA plot with the arrows (`ggplot2` object).
}
\description{
`kPCA_arrows()` draws arrows on a (kernel) PCA plot to represent the contribution
of the original variables to the two displayed Principal Components (PCs).
}
\details{
It is important to note that the arrows are scaled to match the samples' projection
plot. Thus, arrows' directions are correct, but do not expect that their magnitudes
match the output of `kPCA_imp()` or other functions(`prcomp`, `princomp...`).
(Nevertheless, they should at least be proportional to the real magnitudes.)
}
\examples{
dat <- matrix(rnorm(500),ncol=10,nrow=50)
K <- Linear(dat)

## Computing the kernel PCA. The plot represents PC1 and PC2:
kpca <- kPCA(K,plot=1:2)

## Computing the contributions to all the PCS:
pcs <- kPCA_imp(dat,secure=FALSE)

## We will draw the arrows for PC1 and PC2.
contributions <- t(pcs$loadings[1:2,])
rownames(contributions) <- 1:10
kPCA_arrows(plot=kpca$plot,contributions=contributions)
}
