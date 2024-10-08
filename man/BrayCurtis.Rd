% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/kernel_functions.R
\name{BrayCurtis}
\alias{BrayCurtis}
\alias{Ruzicka}
\title{Kernels for count data}
\usage{
BrayCurtis(X)

Ruzicka(X)
}
\arguments{
\item{X}{Matrix or data.frame that contains absolute or relative frequencies.}
}
\value{
Kernel matrix (dimension: \emph{NxN}).
}
\description{
Ruzicka and Bray-Curtis are kernel functions for absolute or relative
frequencies and count data. Both kernels have as input a matrix or data.frame
with dimension \emph{NxD} and \emph{N}>1, \emph{D}>1, containing strictly non-negative real numbers.
Samples should be in the rows. Thus, when working with relative frequencies,
`rowSums(X)` should be 1 (or 100, or another arbitrary number) for \emph{all} rows
(samples) of the dataset.
}
\details{
For more info about these measures, please check Details in
?vegan::vegdist(). Note that, in the vegan help page, "Ruzicka" corresponds to
"quantitative Jaccard". `BrayCurtis(X)` gives the same result than
 `1-vegan::vegdist(X,method="bray")`, and the same with `Ruzicka(data)` and
 `1-vegan::vegdist(data,method="jaccard")`.
}
\examples{
data <- matrix(rpois(5000,lambda=3),ncol=100,nrow=50)
Kruz <- Ruzicka(data)
Kbray <- BrayCurtis(data)
Kruz[1:5,1:5]
Kbray[1:5,1:5]
}
